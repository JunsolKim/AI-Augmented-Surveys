#!/usr/bin/env python
import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roper_build_data as rbd
import roper_gemma_verify as rgv

GEMMA_DIR = rgv.GEMMA_DIR
VERIFY_OUTPUT  = GEMMA_DIR / 'verify_output.csv'
BIN_PARQ_MAIN  = GEMMA_DIR / 'binarize_input.parquet'
BIN_OUT_MAIN   = GEMMA_DIR / 'binarize_output.csv'
BIN_PARQ_FILL  = GEMMA_DIR / 'binarize_input_fill.parquet'
BIN_OUT_FILL   = GEMMA_DIR / 'binarize_output_fill.csv'
BIN_OUT_FINAL  = GEMMA_DIR / 'binarize_output.csv'  # overwrite when merged
JOBS_STATE     = GEMMA_DIR / 'binarize_fill_submitted_jobs.json'

SUBMIT_RETRIES = 8
DOWNLOAD_RETRIES = 6


def setup_logging():
    rbd.BASE_DIR.joinpath('logs').mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    fh = rbd.BASE_DIR / 'logs' / f'{ts}_roper_gemma_binarize_fill.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(fh)],
    )
    logging.getLogger('roper_build').info(f'Log → {fh}')


def build_fill_parquet():
    """Identify Yes pairs in verify_output.csv NOT in binarize_input.parquet
    (these are the fill-derived new Yes pairs), expand to year-level."""
    log = rbd.log
    v = pd.read_csv(VERIFY_OUTPUT)
    yes = v[v['llm_same'].astype(str) == 'Yes'][
        ['gss_variable', 'roper_question']
    ].drop_duplicates()
    log.info(f'current Yes pairs: {len(yes):,d}')

    main = pd.read_parquet(BIN_PARQ_MAIN)
    main_keys = set(zip(main['gss_variable'].astype(str),
                         main['roper_question'].astype(str)))
    log.info(f'main binarize_input pairs: {len(main_keys):,d}')

    yes['key'] = yes.apply(
        lambda r: (str(r['gss_variable']), str(r['roper_question'])), axis=1)
    new_yes = yes[~yes['key'].isin(main_keys)][
        ['gss_variable', 'roper_question']
    ].reset_index(drop=True)
    log.info(f'NEW Yes pairs (fill-derived): {len(new_yes):,d}')
    if len(new_yes) == 0:
        log.info('Nothing to do.')
        return None

    # Pull gss metadata + cos_sim from verify_input.csv (already includes
    # gss_question, gss_binarized, cos_sim).
    cands = pd.read_csv(GEMMA_DIR / 'verify_input.csv')
    new_pairs = new_yes.merge(cands, on=['gss_variable', 'roper_question'],
                               how='left')
    log.info(f'attached gss metadata: {len(new_pairs):,d}')

    # Expand year-level via raw Roper meta
    df_roper = rbd._load_roper_questions()
    log.info(f'loaded raw Roper: {len(df_roper):,d}')
    roper_meta = df_roper[['question', 'responses', 'year',
                            'roper_question_id']
                          ].drop_duplicates(subset=['question', 'year',
                                                     'roper_question_id'])
    df_bin = new_pairs[
        ['gss_question_id', 'gss_variable', 'gss_question', 'gss_binarized',
         'roper_question', 'cos_sim']
    ].merge(roper_meta, left_on='roper_question', right_on='question',
             how='left').drop(columns=['question'])
    df_bin['responses'] = df_bin['responses'].apply(
        lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list)
                  else (v if isinstance(v, str) else None))
    log.info(f'expanded rows (year-level): {len(df_bin):,d}')
    df_bin.to_parquet(BIN_PARQ_FILL, index=False)
    log.info(f'wrote {BIN_PARQ_FILL}')
    return df_bin


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    setup_logging()
    log = rbd.log

    if BIN_OUT_FINAL.exists() and not args.force:
        existing = pd.read_csv(BIN_OUT_FINAL)
        log.info(f'existing binarize_output.csv: {len(existing):,d} rows')

    # Step 1 — build fill parquet (or skip if exists)
    if not BIN_PARQ_FILL.exists() or args.force:
        df_bin = build_fill_parquet()
        if df_bin is None:
            return
    else:
        df_bin = pd.read_parquet(BIN_PARQ_FILL)
        log.info(f'reloaded existing fill parquet: {len(df_bin):,d} rows')

    if len(df_bin) == 0:
        log.info('No new pairs to binarize. Done.')
        return

    # Step 2 — build prompts
    def _resp(v):
        if isinstance(v, list): return v
        if isinstance(v, str):
            try: return json.loads(v)
            except: return []
        return []
    prompts = [
        rbd._build_binarization_prompt(
            row['gss_question'], row['gss_binarized'],
            row['roper_question'], _resp(row['responses']),
        )
        for _, row in df_bin.iterrows()
    ]
    log.info(f'built {len(prompts):,d} binarization prompts')

    # Step 3 — submit Gemini inline batches
    from google.genai import types
    client = rbd._get_gemini_client()
    if client is None:
        log.error('no Gemini client; abort')
        sys.exit(2)

    gen_cfg = types.GenerateContentConfig(
        response_mime_type='application/json', temperature=0.2,
    )
    log.info(f'config: model=gemini-2.5-pro  '
             f'cfg={gen_cfg.model_dump(exclude_none=True)}')

    n_batches = math.ceil(len(prompts) / rbd.BATCH_SIZE)
    log.info(f'submitting {n_batches} inline batch(es)')

    state = {}
    if JOBS_STATE.exists():
        try:
            state = json.loads(JOBS_STATE.read_text())
        except Exception:
            state = {}
    log.info(f'resumable state: {len(state)}/{n_batches}')

    for b in range(n_batches):
        if str(b) in state:
            log.info(f'  [{b+1}/{n_batches}] cached → {state[str(b)]}')
            continue
        s = b * rbd.BATCH_SIZE
        e = min(s + rbd.BATCH_SIZE, len(prompts))
        inlined = [
            types.InlinedRequest(
                contents=[types.Content(
                    parts=[types.Part(text=prompts[j])], role='user')],
                config=gen_cfg,
                metadata={'key': f'binarize_fill-{b}-{j-s}'},
            )
            for j in range(s, e)
        ]
        for attempt in range(1, SUBMIT_RETRIES + 1):
            try:
                job = client.batches.create(
                    model='gemini-2.5-pro',
                    src=types.BatchJobSource(inlined_requests=inlined),
                    config={'display_name': f'binarize_fill_{b:03d}'},
                )
                state[str(b)] = job.name
                JOBS_STATE.write_text(json.dumps(state, indent=2))
                log.info(f'  [{b+1}/{n_batches}] submitted → {job.name}')
                break
            except Exception as ex:
                wait = min(5 * (2 ** (attempt - 1)), 120)
                log.warning(f'  submit attempt {attempt} failed: '
                            f'{type(ex).__name__}: {ex} — sleep {wait}s')
                emsg = str(ex).lower()
                if any(w in emsg for w in ('quota', 'limit', 'maximum')):
                    try:
                        rbd._wait_for_any_batch_to_complete(client)
                    except Exception:
                        pass
                time.sleep(wait)
        else:
            log.error(f'  [{b+1}/{n_batches}] gave up')
            sys.exit(3)

    # Step 4 — wait for completion
    job_set = set(state.values())
    log.info(f'Waiting for {len(job_set)} batch(es)…')
    while True:
        try:
            all_b = list(client.batches.list())
        except Exception as ex:
            log.warning(f'  poll error: {ex} — sleep {rbd.POLL_SEC}s')
            time.sleep(rbd.POLL_SEC); continue
        ours = [b for b in all_b if b.name in job_set]
        cnt = Counter(b.state.name for b in ours)
        active = [b for b in ours if b.state.name not in rbd.COMPLETED_STATES]
        log.info(f'  states: {dict(cnt)}  ({len(active)}/{len(ours)} active)')
        if ours and not active:
            break
        time.sleep(rbd.POLL_SEC)

    # Step 5 — download + parse
    parsed = {}
    for b_obj in sorted(ours, key=lambda x: x.display_name or x.name):
        if b_obj.state.name != 'JOB_STATE_SUCCEEDED':
            log.warning(f'[skip] {b_obj.display_name} — {b_obj.state.name}')
            continue
        label = b_obj.display_name or b_obj.name.split('/')[-1]
        raw_path = GEMMA_DIR / f'binarize_output_{label}.json'
        if raw_path.exists():
            try:
                cached = json.loads(raw_path.read_text())
                for r in cached:
                    md = r.get('metadata') or {}
                    key = md.get('key', '') if isinstance(md, dict) else ''
                    try:
                        text = r['response']['candidates'][0]['content']['parts'][0]['text']
                        d = json.loads(text)
                    except Exception:
                        d = {}
                    parsed[key] = d
                log.info(f'[cached] {label}: {len(cached)} responses')
                continue
            except Exception:
                pass
        full = None
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                full = client.batches.get(name=b_obj.name); break
            except Exception as ex:
                wait = min(10 * (2 ** (attempt - 1)), 120)
                log.warning(f'[download {label}] attempt {attempt} failed: '
                            f'{ex} — sleep {wait}s')
                time.sleep(wait)
        if full is None or full.dest is None or not full.dest.inlined_responses:
            log.error(f'[skip] {label} — no responses')
            continue
        responses = full.dest.inlined_responses
        try:
            raw_path.write_text(json.dumps(
                [r.model_dump() if hasattr(r, 'model_dump') else dict(r)
                 for r in responses], indent=2, default=str))
        except Exception as ex:
            log.warning(f'[raw write failed] {label}: {ex}')
        n_ok = 0
        for r in responses:
            md = r.metadata or {}
            key = md.get('key', '') if isinstance(md, dict) else ''
            try:
                text = r.response.candidates[0].content.parts[0].text
                d = json.loads(text); n_ok += 1
            except Exception:
                d = {}
            parsed[key] = d
        log.info(f'[done] {label}: {n_ok}/{len(responses)} parsed')

    # Step 6 — map results back to df_bin
    n = len(df_bin)
    mapping = [None] * n
    yes_pct = [None] * n
    n_assigned = 0
    for b in range(n_batches):
        s = b * rbd.BATCH_SIZE
        e = min(s + rbd.BATCH_SIZE, n)
        for local in range(e - s):
            key = f'binarize_fill-{b}-{local}'
            d = parsed.get(key)
            if not d: continue
            mapping[s + local] = json.dumps(d.get('mapping') or {},
                                             ensure_ascii=False)
            yes_pct[s + local] = d.get('roper_yes_pct')
            n_assigned += 1
    log.info(f'assigned binarization to {n_assigned:,d}/{n:,d} rows '
             f'(missing: {n - n_assigned:,d})')

    df_bin = df_bin.copy()
    df_bin['llm_mapping'] = mapping
    df_bin['roper_yes_pct'] = yes_pct
    df_bin.to_csv(BIN_OUT_FILL, index=False)
    log.info(f'wrote {BIN_OUT_FILL}')

    # Step 7 — concat with main binarize output
    if BIN_OUT_MAIN.exists():
        main_out = pd.read_csv(BIN_OUT_MAIN)
        log.info(f'main binarize_output.csv: {len(main_out):,d} rows')
        combined = pd.concat([main_out, df_bin], ignore_index=True)
        combined.to_csv(BIN_OUT_FINAL, index=False)
        log.info(f'wrote merged {BIN_OUT_FINAL}: {len(combined):,d} rows '
                 f'(main {len(main_out):,d} + fill {len(df_bin):,d})')
    else:
        log.warning(f'{BIN_OUT_MAIN} missing — wrote only fill output')
    log.info('Done.')


if __name__ == '__main__':
    main()

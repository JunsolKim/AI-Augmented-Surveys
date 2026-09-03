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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roper_build_data as rbd
import roper_gemma_verify as rgv

GEMMA_DIR = rgv.GEMMA_DIR
BINARIZE_INPUT  = GEMMA_DIR / 'binarize_input.parquet'
VERIFY_OUTPUT   = GEMMA_DIR / 'verify_output.csv'
OUT_PATH        = GEMMA_DIR / 'binarize_output.csv'
JOBS_STATE      = GEMMA_DIR / 'binarize_submitted_jobs.json'


def build_gemma_binarize_input():
    """Build binarize_input.parquet for ONLY the verify-passed (Yes) gemma
    pairs by joining with raw Roper metadata (year-level rows + responses)."""
    log = rbd.log
    log.info('Building gemma binarize_input.parquet (Yes-only)…')

    df_cands = pd.read_csv(GEMMA_DIR / 'verify_input.csv')
    log.info(f'  gemma candidates (total): {len(df_cands):,d}')

    verify = pd.read_csv(VERIFY_OUTPUT)
    yes_keys = verify[verify['llm_same'].astype(str) == 'Yes'][
        ['gss_variable', 'roper_question']
    ].drop_duplicates()
    log.info(f'  verified Yes pairs: {len(yes_keys):,d}')

    df_yes = df_cands.merge(yes_keys, on=['gss_variable', 'roper_question'],
                             how='inner')
    log.info(f'  Yes-pair candidates: {len(df_yes):,d}')

    df_roper = rbd._load_roper_questions()
    log.info(f'  loaded raw Roper: {len(df_roper):,d}')
    roper_meta = df_roper[
        ['question', 'responses', 'year', 'roper_question_id']
    ].drop_duplicates(subset=['question', 'year', 'roper_question_id'])

    df_bin = df_yes[
        ['gss_question_id', 'gss_variable', 'gss_question', 'gss_binarized',
         'roper_question', 'cos_sim']
    ].merge(roper_meta, left_on='roper_question', right_on='question',
             how='left').drop(columns=['question'])
    df_bin['responses'] = df_bin['responses'].apply(
        lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list)
                  else (v if isinstance(v, str) else None))
    log.info(f'  expanded rows (year-level): {len(df_bin):,d}')
    df_bin.to_parquet(BINARIZE_INPUT, index=False)
    log.info(f'  wrote {BINARIZE_INPUT}')
    return df_bin

SUBMIT_RETRIES = 8
DOWNLOAD_RETRIES = 6


def setup_logging():
    rbd.BASE_DIR.joinpath('logs').mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    fh = rbd.BASE_DIR / 'logs' / f'{ts}_roper_gemma_binarize.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(fh)],
    )
    logging.getLogger('roper_build').info(f'Log → {fh}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    setup_logging()
    log = rbd.log

    if not VERIFY_OUTPUT.exists():
        log.error(f'Missing {VERIFY_OUTPUT}')
        sys.exit(2)

    if OUT_PATH.exists() and not args.force:
        log.info(f'{OUT_PATH} already exists — pass --force to recompute')
        return

    if not BINARIZE_INPUT.exists() or args.force:
        build_gemma_binarize_input()
    df_bin = pd.read_parquet(BINARIZE_INPUT)
    log.info(f'binarize_input.parquet rows (Yes-only): {len(df_bin):,d}')

    # Each row -> one prompt. responses is stored as JSON string in parquet;
    # deserialize back to list-of-[ord, text, pct] triples for the prompt.
    def _resp(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return []
        return []
    prompts = [
        rbd._build_binarization_prompt(
            row['gss_question'], row['gss_binarized'],
            row['roper_question'], _resp(row['responses']),
        )
        for _, row in df_bin.iterrows()
    ]
    log.info(f'built {len(prompts):,d} binarization prompts')

    # Submit inline batches with retry + resumable state
    from google.genai import types
    client = rbd._get_gemini_client()
    if client is None:
        log.error('no Gemini client; abort')
        sys.exit(2)

    gen_cfg = types.GenerateContentConfig(
        response_mime_type='application/json',
        temperature=0.2,
    )
    log.info(f'config: model=gemini-2.5-pro  '
             f'cfg={gen_cfg.model_dump(exclude_none=True)}')

    n_batches = math.ceil(len(prompts) / rbd.BATCH_SIZE)
    log.info(f'submitting {n_batches} inline batch(es) of size '
             f'{rbd.BATCH_SIZE}')

    state = {}
    if JOBS_STATE.exists():
        try:
            state = json.loads(JOBS_STATE.read_text())
        except Exception:
            state = {}
    log.info(f'resumable state: {len(state)}/{n_batches} already submitted')

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
                metadata={'key': f'binarize-{b}-{j-s}'},
            )
            for j in range(s, e)
        ]
        for attempt in range(1, SUBMIT_RETRIES + 1):
            try:
                job = client.batches.create(
                    model='gemini-2.5-pro',
                    src=types.BatchJobSource(inlined_requests=inlined),
                    config={'display_name': f'binarize_{b:03d}'},
                )
                state[str(b)] = job.name
                JOBS_STATE.write_text(json.dumps(state, indent=2))
                log.info(f'  [{b+1}/{n_batches}] submitted → {job.name}')
                break
            except Exception as ex:
                wait = min(5 * (2 ** (attempt - 1)), 120)
                log.warning(f'  submit attempt {attempt}/{SUBMIT_RETRIES} '
                            f'failed: {type(ex).__name__}: {ex} — '
                            f'sleep {wait}s')
                emsg = str(ex).lower()
                if any(w in emsg for w in ('quota', 'limit', 'maximum')):
                    try:
                        rbd._wait_for_any_batch_to_complete(client)
                    except Exception:
                        pass
                time.sleep(wait)
        else:
            log.error(f'  [{b+1}/{n_batches}] gave up after '
                      f'{SUBMIT_RETRIES} attempts')
            sys.exit(3)

    # Wait for completion
    job_set = set(state.values())
    log.info(f'Waiting for {len(job_set)} batch(es) to complete …')
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

    # Download + parse with retries; cache raw .json per batch for resume
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
                full = client.batches.get(name=b_obj.name)
                break
            except Exception as ex:
                wait = min(10 * (2 ** (attempt - 1)), 120)
                log.warning(f'[download {label}] attempt {attempt}/'
                            f'{DOWNLOAD_RETRIES} failed: '
                            f'{type(ex).__name__}: {ex} — sleep {wait}s')
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
                d = json.loads(text)
                n_ok += 1
            except Exception:
                d = {}
            parsed[key] = d
        log.info(f'[done] {label}: {n_ok}/{len(responses)} parsed')

    # Map results back to df_bin rows by key
    n = len(df_bin)
    mapping = [None] * n
    yes_pct = [None] * n
    n_assigned = 0
    for b in range(n_batches):
        s = b * rbd.BATCH_SIZE
        e = min(s + rbd.BATCH_SIZE, n)
        for local in range(e - s):
            key = f'binarize-{b}-{local}'
            d = parsed.get(key)
            if not d:
                continue
            mapping[s + local] = json.dumps(d.get('mapping') or {},
                                            ensure_ascii=False)
            yes_pct[s + local] = d.get('roper_yes_pct')
            n_assigned += 1
    log.info(f'assigned binarization to {n_assigned:,d}/{n:,d} rows '
             f'(missing: {n - n_assigned:,d})')

    df_bin = df_bin.copy()
    df_bin['llm_mapping'] = mapping
    df_bin['roper_yes_pct'] = yes_pct
    df_bin.to_csv(OUT_PATH, index=False)
    log.info(f'wrote {OUT_PATH}')
    log.info('Done.')


if __name__ == '__main__':
    main()

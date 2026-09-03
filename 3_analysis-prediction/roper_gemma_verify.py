#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roper_build_data as rbd

SUBMIT_RETRIES = 8
SUBMIT_BACKOFF_BASE = 5.0


def _wait_for_jobs(client, job_names):
    """Poll until all submitted batch jobs reach a terminal state.
    Tolerant to transient 503/timeout errors during polling."""
    from collections import Counter
    log = rbd.log
    job_set = set(job_names)
    log.info(f'Waiting for {len(job_names)} batch(es) to complete …')
    last_ours = []
    while True:
        try:
            all_batches = list(client.batches.list())
        except Exception as e:
            log.warning(f'  poll error: {type(e).__name__}: {e} — sleep '
                        f'{rbd.POLL_SEC}s and retry')
            time.sleep(rbd.POLL_SEC)
            continue
        ours = [b for b in all_batches if b.name in job_set]
        last_ours = ours
        active = [b for b in ours
                  if b.state.name not in rbd.COMPLETED_STATES]
        cnt = Counter(b.state.name for b in ours)
        log.info(f'  states: {dict(cnt)}  '
                 f'({len(active)}/{len(ours)} active, '
                 f'tracked={len(ours)}/{len(job_names)})')
        if ours and not active:
            return ours
        time.sleep(rbd.POLL_SEC)


def _download_inline_results(client, batches, prefix, work_dir):
    """For inline batches, results are exposed at batch.dest.inlined_responses.
    Per-batch retries on transient 5xx; skips batches that persistently fail.
    Caches per-batch raw JSON to disk so reruns can resume.
    """
    log = rbd.log
    out = []
    DOWNLOAD_RETRIES = 6
    DOWNLOAD_BACKOFF = 10.0
    for b in sorted(batches, key=lambda x: x.display_name or x.name):
        if b.state.name != 'JOB_STATE_SUCCEEDED':
            log.warning(f'[skip] {b.display_name or b.name} — {b.state.name}')
            continue
        label = b.display_name or b.name.split('/')[-1]
        raw_path = work_dir / f'{prefix}_output_{label}.json'
        # Resume: if we already cached this batch, parse it from disk
        if raw_path.exists():
            try:
                cached = json.loads(raw_path.read_text())
                log.info(f'[cached] {label}: {len(cached)} responses')
                for r in cached:
                    md = r.get('metadata') or {}
                    key = md.get('key', '') if isinstance(md, dict) else ''
                    try:
                        text = r['response']['candidates'][0]['content']['parts'][0]['text']
                        parsed = json.loads(text)
                    except Exception:
                        parsed = {}
                    out.append({'key': key, 'result': parsed})
                continue
            except Exception as e:
                log.warning(f'[cached read failed] {label}: {e} — re-fetching')

        full = None
        last_err = None
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                full = client.batches.get(name=b.name)
                break
            except Exception as e:
                last_err = e
                wait = min(DOWNLOAD_BACKOFF * (2 ** (attempt - 1)), 120.0)
                log.warning(f'[download {label}] attempt {attempt}/'
                            f'{DOWNLOAD_RETRIES} failed: '
                            f'{type(e).__name__}: {e} — sleep {wait:.0f}s')
                time.sleep(wait)
        if full is None:
            log.error(f'[skip] {label} — could not fetch detail '
                      f'after {DOWNLOAD_RETRIES} retries: {last_err}')
            continue

        dest = full.dest
        responses = []
        if dest is not None and getattr(dest, 'inlined_responses', None):
            responses = dest.inlined_responses
        else:
            log.warning(f'[no responses] {label}')
            continue

        # Persist raw to disk for audit + resume
        try:
            raw_path.write_text(json.dumps(
                [r.model_dump() if hasattr(r, 'model_dump') else dict(r)
                 for r in responses], indent=2, default=str))
        except Exception as e:
            log.warning(f'[raw write failed] {label}: {e}')

        n_ok = 0
        for resp in responses:
            metadata = getattr(resp, 'metadata', None) or {}
            key = (metadata.get('key') if isinstance(metadata, dict)
                   else getattr(metadata, 'key', '')) or ''
            try:
                inner = resp.response
                text = inner.candidates[0].content.parts[0].text
                parsed = json.loads(text)
                n_ok += 1
            except Exception:
                parsed = {}
            out.append({'key': key, 'result': parsed})
        log.info(f'[done] {label}: {n_ok}/{len(responses)} parsed')
    return out


def _submit_inline_batches(client, prefix, prompts, work_dir, model):
    """Submit Gemini batch jobs INLINE (no Files API). Returns list of job
    names. Persistent state file maps batch index -> job name so reruns skip
    already-submitted batches.
    """
    from google.genai import types
    from google.genai import errors as genai_errors
    log = rbd.log

    n_batches = (len(prompts) + rbd.BATCH_SIZE - 1) // rbd.BATCH_SIZE
    log.info(f'{prefix} (inline): {len(prompts)} prompts → {n_batches} batches')

    state_path = work_dir / f'{prefix}_submitted_jobs.json'
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            state = {}
    else:
        state = {}
    log.info(f'  resumable state: {len(state)}/{n_batches} already submitted '
             f'({state_path.name})')

    gen_cfg = types.GenerateContentConfig(
        response_mime_type='application/json',
        max_output_tokens=2000,
        temperature=0.2,
    )

    for b in range(n_batches):
        if str(b) in state:
            log.info(f'  [{b+1}/{n_batches}] cached → {state[str(b)]}')
            continue
        start = b * rbd.BATCH_SIZE
        chunk = prompts[start:start + rbd.BATCH_SIZE]
        inlined = [
            types.InlinedRequest(
                contents=[types.Content(
                    parts=[types.Part(text=p)], role='user')],
                config=gen_cfg,
                metadata={'key': f'{prefix}-{b}-{i}'},
            )
            for i, p in enumerate(chunk)
        ]
        src = types.BatchJobSource(inlined_requests=inlined)

        last_err = None
        for attempt in range(1, SUBMIT_RETRIES + 1):
            try:
                job = client.batches.create(
                    model=model, src=src,
                    config={'display_name': f'{prefix}_batch_{b:03d}'},
                )
                state[str(b)] = job.name
                state_path.write_text(json.dumps(state, indent=2))
                log.info(f'  [{b+1}/{n_batches}] submitted → {job.name}  '
                         f'state={job.state.name}')
                break
            except Exception as e:
                last_err = e
                wait = min(SUBMIT_BACKOFF_BASE * (2 ** (attempt - 1)), 120.0)
                log.warning(f'  [{b+1}/{n_batches}] submit attempt {attempt}/'
                            f'{SUBMIT_RETRIES} failed: '
                            f'{type(e).__name__}: {e} — sleep {wait:.0f}s')
                # If we hit a per-account active-batch limit, wait for
                # any running batch to clear before retry.
                emsg = str(e).lower()
                if 'maximum' in emsg or 'limit' in emsg or 'quota' in emsg:
                    try:
                        rbd._wait_for_any_batch_to_complete(client)
                    except Exception:
                        pass
                time.sleep(wait)
        else:
            log.error(f'  [{b+1}/{n_batches}] gave up after '
                      f'{SUBMIT_RETRIES} attempts. last={last_err}')
            raise RuntimeError(f'Submit failed for batch {b}: {last_err}')

    return [state[str(b)] for b in range(n_batches)]

# Keep gemma outputs separate from the mpnet outputs
GEMMA_DIR = rbd.SAVE_DIR / 'colab_llm_gemma'
GEMMA_DIR.mkdir(parents=True, exist_ok=True)

GSS_EMB_PATH   = rbd.GSS_DIR / 'gss_gemma_embeddings.npy'
ROPER_EMB_PATH = rbd.GSS_DIR / 'roper_gemma_embeddings.npy'
ROPER_QSV_PATH = rbd.GSS_DIR / 'roper_unique_questions.csv'

CACHED_VERIFY = rbd.SAVE_DIR / 'colab_llm' / 'verify_output.csv'

THRESHOLD = 0.80
CHUNK = 1000


def setup_logging():
    rbd.BASE_DIR.joinpath('logs').mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    fh = rbd.BASE_DIR / 'logs' / f'{ts}_roper_gemma_verify.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(fh)],
    )
    logging.getLogger('roper_build').info(f'Log → {fh}')
    return fh


def build_candidate_pairs(threshold):
    """Compute cosine matrix in chunks; return DataFrame of candidate pairs."""
    log = rbd.log
    log.info(f'Loading cached gemma embeddings (threshold={threshold})')
    gss = pd.read_csv(rbd.GSS_DIR / 'gss_questions_for_matching.csv')
    rop_q = pd.read_csv(ROPER_QSV_PATH)['question'].tolist()
    gss_emb = np.load(GSS_EMB_PATH)
    rop_emb = np.load(ROPER_EMB_PATH)
    assert gss_emb.shape[0] == len(gss), 'GSS row count mismatch'
    assert rop_emb.shape[0] == len(rop_q), 'Roper row count mismatch'
    log.info(f'  gss_emb={gss_emb.shape}, rop_emb={rop_emb.shape}')

    rows = []
    for start in range(0, len(rop_q), CHUNK):
        end = min(start + CHUNK, len(rop_q))
        sims = gss_emb @ rop_emb[start:end].T   # (n_gss, c)
        if sims.max() < threshold:
            continue
        gi, ri = np.where(sims >= threshold)
        for g, r in zip(gi, ri):
            rows.append({
                'gss_question_id': int(gss.iloc[g]['question_id']),
                'gss_variable':    gss.iloc[g]['variable'],
                'gss_question':    gss.iloc[g]['question'],
                'gss_binarized':   gss.iloc[g]['binarized'],
                'roper_question':  rop_q[start + int(r)],
                'roper_question_idx': start + int(r),
                'cos_sim': float(sims[g, r]),
            })
        if (start // CHUNK) % 50 == 0:
            log.info(f'  chunk {start}/{len(rop_q)}  pairs so far: {len(rows)}')
    df = pd.DataFrame(rows)
    log.info(f'Total candidate pairs at threshold {threshold}: {len(df):,d}')
    return df


def dedup_against_cache(df_cands):
    """Drop (gss_variable, roper_question) pairs already in cached
    verify_output.csv."""
    log = rbd.log
    if not CACHED_VERIFY.exists():
        log.info(f'No cached verify_output.csv at {CACHED_VERIFY} — all '
                 f'pairs are new')
        return df_cands.copy(), pd.DataFrame()

    cached = pd.read_csv(CACHED_VERIFY)
    log.info(f'Cached verify_output.csv rows: {len(cached):,d}')
    cached_keys = set(zip(cached['gss_variable'].astype(str),
                          cached['roper_question'].astype(str)))
    log.info(f'  unique (gss_var, roper_q) keys in cache: {len(cached_keys):,d}')

    cand_keys = list(zip(df_cands['gss_variable'].astype(str),
                         df_cands['roper_question'].astype(str)))
    is_new = [k not in cached_keys for k in cand_keys]
    df_new    = df_cands[is_new].reset_index(drop=True)
    df_cached = df_cands[[not x for x in is_new]].reset_index(drop=True)
    log.info(f'  → {len(df_new):,d} NEW pairs (need Gemini)')
    log.info(f'  → {len(df_cached):,d} cached pairs (reuse labels)')
    return df_new, df_cached


def cost_estimate(n_pairs):
    """Rough $ estimate for Gemini-2.5-Pro batch verification."""
    in_tok_per = 320
    out_tok_per = 50
    in_rate  = 0.625e-6   # $/token, batch
    out_rate = 5.0e-6     # $/token, batch
    cost = n_pairs * (in_tok_per * in_rate + out_tok_per * out_rate)
    return cost


def stage_b_verify_new(df_new, force=False):
    """Submit Gemini batch on NEW pairs; return DataFrame with labels."""
    log = rbd.log
    out_path = GEMMA_DIR / 'verify_output_new.csv'
    if not force and out_path.exists():
        log.info(f'[verify] {out_path.name} exists; loading')
        return pd.read_csv(out_path)

    in_path = GEMMA_DIR / 'verify_input_new.csv'
    df_new.to_csv(in_path, index=False)
    log.info(f'[verify] wrote {in_path} with {len(df_new):,d} pairs')

    client = rbd._get_gemini_client()
    if client is None:
        log.error('GEMINI_API_KEY missing — cannot run batch. '
                  'Wrote verify_input_new.csv only.')
        return None

    prompts = [
        rbd._build_verification_prompt(row['gss_question'],
                                        row['roper_question'])
        for _, row in df_new.iterrows()
    ]
    job_names = _submit_inline_batches(client, 'verify', prompts, GEMMA_DIR,
                                        rbd.GEMINI_MODEL)
    batches   = _wait_for_jobs(client, job_names)
    results   = _download_inline_results(client, batches, 'verify',
                                          GEMMA_DIR)

    # Map results back to df_new rows by their (batch_idx, pos_in_batch) key.
    # Missing batches/responses → row stays NaN.
    key_to_row = {}
    n = len(df_new)
    for b in range((n + rbd.BATCH_SIZE - 1) // rbd.BATCH_SIZE):
        start = b * rbd.BATCH_SIZE
        for i in range(min(rbd.BATCH_SIZE, n - start)):
            key_to_row[f'verify-{b}-{i}'] = start + i

    same = [None] * n
    conf = [None] * n
    n_assigned = 0
    for r in results:
        idx = key_to_row.get(r.get('key', ''))
        if idx is None:
            continue
        same[idx] = r['result'].get('answer')
        conf[idx] = r['result'].get('confidence')
        n_assigned += 1
    log.info(f'[verify] assigned labels to {n_assigned:,d}/{n:,d} rows '
             f'(missing: {n - n_assigned:,d})')

    df_new = df_new.copy()
    df_new['llm_same']       = same
    df_new['llm_confidence'] = conf
    yes = int((df_new.llm_same == 'Yes').sum())
    no  = int((df_new.llm_same == 'No').sum())
    log.info(f'[verify] Yes={yes:,d}  No={no:,d}  '
             f'(parsed {yes+no:,d}/{len(df_new):,d})')
    df_new[['gss_variable', 'roper_question',
            'llm_same', 'llm_confidence']].to_csv(out_path, index=False)
    log.info(f'[verify] wrote {out_path}')
    return df_new


def merge_labels(df_cands, df_cached_pairs, df_new_labeled):
    """Combine cached + new labels into final verify_output.csv."""
    log = rbd.log
    cached_lab = pd.read_csv(CACHED_VERIFY) if CACHED_VERIFY.exists() else \
                  pd.DataFrame(columns=['gss_variable','roper_question',
                                        'llm_same','llm_confidence'])
    cached_lookup = dict(zip(
        zip(cached_lab['gss_variable'].astype(str),
            cached_lab['roper_question'].astype(str)),
        zip(cached_lab['llm_same'], cached_lab['llm_confidence']),
    ))
    new_lookup = {}
    if df_new_labeled is not None and len(df_new_labeled):
        new_lookup = dict(zip(
            zip(df_new_labeled['gss_variable'].astype(str),
                df_new_labeled['roper_question'].astype(str)),
            zip(df_new_labeled['llm_same'], df_new_labeled['llm_confidence']),
        ))

    out = df_cands.copy()
    same, conf, src = [], [], []
    for _, row in out.iterrows():
        k = (str(row['gss_variable']), str(row['roper_question']))
        if k in cached_lookup:
            s, c = cached_lookup[k]; src.append('cached')
        elif k in new_lookup:
            s, c = new_lookup[k]; src.append('new')
        else:
            s, c = None, None;     src.append('unlabeled')
        same.append(s); conf.append(c)
    out['llm_same'] = same
    out['llm_confidence'] = conf
    out['label_source'] = src

    out_path = GEMMA_DIR / 'verify_output.csv'
    out.to_csv(out_path, index=False)
    log.info(f'[merge] wrote {out_path} '
             f'(cached={src.count("cached"):,d}  new={src.count("new"):,d}  '
             f'unlabeled={src.count("unlabeled"):,d})')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threshold', type=float, default=THRESHOLD)
    p.add_argument('--dry-run', action='store_true',
                   help='Build candidates + show cost; do NOT call Gemini.')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    log_file = setup_logging()
    log = rbd.log
    log.info(f'roper_gemma_verify starting (threshold={args.threshold}, '
             f'dry_run={args.dry_run})')

    # 1. candidate pairs at threshold
    cand_path = GEMMA_DIR / 'verify_input.csv'
    if cand_path.exists() and not args.force:
        log.info(f'Loading existing candidate pairs from {cand_path}')
        df_cands = pd.read_csv(cand_path)
    else:
        df_cands = build_candidate_pairs(args.threshold)
        df_cands.to_csv(cand_path, index=False)
        log.info(f'wrote {cand_path}')

    # 2. dedup against cache
    df_new, df_cached = dedup_against_cache(df_cands)
    new_path = GEMMA_DIR / 'verify_input_new.csv'
    df_new.to_csv(new_path, index=False)
    log.info(f'wrote {new_path}')

    cost = cost_estimate(len(df_new))
    log.info(f'Estimated Gemini batch cost for {len(df_new):,d} new pairs: '
             f'~${cost:.2f}')

    # 3. submit batch (unless dry-run)
    if args.dry_run:
        log.info('--dry-run: stopping before Gemini batch submit.')
        log.info('To proceed: rerun without --dry-run with GEMINI_API_KEY set.')
        return

    df_new_labeled = stage_b_verify_new(df_new, force=args.force)
    if df_new_labeled is None:
        log.error('Batch did not run; merged output not produced.')
        return

    # 4. merge cached + new labels into unified verify_output.csv
    merge_labels(df_cands, df_cached, df_new_labeled)
    log.info('Done.')


if __name__ == '__main__':
    main()

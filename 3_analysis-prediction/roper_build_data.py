#!/usr/bin/env python
import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE_DIR / 'data'
GSS_DIR    = DATA_DIR / 'roper_prep'
ROPER_DIR  = DATA_DIR / 'roper'
SAVE_DIR   = DATA_DIR / 'roper_prep' / 'results'
COLAB_DIR  = SAVE_DIR / 'colab_llm'

SBERT_MODEL   = 'sentence-transformers/all-mpnet-base-v2'
GEMINI_MODEL  = 'gemini-2.5-pro'
COS_THRESHOLD = 0.8
CHUNK_SIZE    = 500

BATCH_SIZE      = 1000
POLL_SEC        = 30
MAX_RETRIES     = 10
RETRY_DELAY_SEC = 60
US_COUNTRY_ID = 'da81340e-5e20-454e-ab85-052e698c2f57'
COMPLETED_STATES = {
    'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED',
}

log = logging.getLogger('roper_build')


# --- Stage A: local matching -------------------------------------------------

def _load_gss_questions():
    return pd.read_csv(GSS_DIR / 'gss_questions_for_matching.csv')


def _load_roper_questions():
    """~507k Roper questions from per-year JSONL + CSV metadata."""
    question_json_dir = ROPER_DIR / 'question' / 'json'
    year_files = sorted(glob.glob(str(question_json_dir / 'all_questions_*.json')))
    log.info(f'Found {len(year_files)} Roper year files')

    dataset = []
    n_skipped = 0
    for fpath in tqdm(year_files, desc='Loading Roper questions'):
        year = int(Path(fpath).stem.split('_')[-1])
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    q = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if not isinstance(q, dict):
                    n_skipped += 1
                    continue

                pt = q.get('processedText')
                question_text = pt.get('value') if isinstance(pt, dict) else None

                responses = []
                for r in q.get('responses', []) or []:
                    if not isinstance(r, dict):
                        continue
                    display = r.get('displayText')
                    text = display.get('value') if isinstance(display, dict) else None
                    if text is None and isinstance(display, dict):
                        nf = display.get('normalizedForm')
                        text = nf.get('value') if isinstance(nf, dict) else None
                    if text is not None:
                        responses.append([
                            r.get('ordinal'), text, r.get('percentage'),
                        ])

                study = q.get('study') or {}
                if not isinstance(study, dict):
                    study = {}
                dataset.append({
                    'year': year,
                    'question': question_text,
                    'responses': responses,
                    'roper_question_id': q.get('id'),
                    'roper_study_id': study.get('id'),
                    'study_title': study.get('title'),
                    'survey_by': study.get('depositor'),
                    'conducted_by': study.get('collectionMode'),
                    'topline': q.get('topline'),
                })

    df_roper = pd.DataFrame(dataset)
    log.info(f'Total Roper questions: {len(df_roper)} (skipped non-dict: {n_skipped})')

    csv_dir = ROPER_DIR / 'csv' / 'question'
    csv_files = sorted(glob.glob(str(csv_dir / 'ipoll_question_results_*.csv')))
    csv_dfs = [pd.read_csv(f, low_memory=False)
               for f in tqdm(csv_files, desc='Loading CSVs')]
    df_csv = pd.concat(csv_dfs, ignore_index=True)
    df_csv['united_states'] = df_csv['countryIds'].apply(
        lambda x: int(US_COUNTRY_ID in str(x)))

    merge_cols = ['id', 'studyTitle', 'surveyBy', 'conductedBy', 'united_states']
    merge_cols = [c for c in merge_cols if c in df_csv.columns]
    df_roper = df_roper.merge(
        df_csv[merge_cols].drop_duplicates('id'),
        left_on='roper_question_id', right_on='id', how='left'
    ).drop(columns=['id'], errors='ignore')

    df_roper['gss'] = df_roper['studyTitle'].apply(
        lambda x: 'general social survey' in str(x).lower())
    df_roper['national_adult'] = df_roper['topline'].apply(
        lambda x: int('national adult' in str(x).lower()) if pd.notna(x) else 0)

    df_roper = df_roper[
        (df_roper['united_states'] == 1) & (~df_roper['gss'])
    ].reset_index(drop=True)

    log.info(f'After filters: {len(df_roper)} '
             f'(national_adult={int(df_roper["national_adult"].sum())})')
    return df_roper


def _compute_or_load_embeddings(gss_questions, roper_unique_questions):
    import torch
    from sentence_transformers import SentenceTransformer

    gss_emb_path   = GSS_DIR / 'gss_sbert_embeddings.npy'
    roper_emb_path = GSS_DIR / 'roper_sbert_embeddings.npy'

    if gss_emb_path.exists() and roper_emb_path.exists():
        log.info('Cached SBERT embeddings found; skipping model load.')
        return np.load(gss_emb_path), np.load(roper_emb_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'Loading SBERT ({SBERT_MODEL}) on {device}')
    sbert = SentenceTransformer(SBERT_MODEL, device=device)

    if gss_emb_path.exists():
        gss_emb = np.load(gss_emb_path)
    else:
        gss_texts = gss_questions['question'].fillna('').tolist()
        gss_emb = sbert.encode(gss_texts, batch_size=128,
                               show_progress_bar=True,
                               normalize_embeddings=True)
        np.save(gss_emb_path, gss_emb)
        log.info(f'Saved GSS embeddings → {gss_emb_path}')

    if roper_emb_path.exists():
        roper_emb = np.load(roper_emb_path)
    else:
        roper_emb = sbert.encode(roper_unique_questions.tolist(),
                                 batch_size=128,
                                 show_progress_bar=True,
                                 normalize_embeddings=True)
        np.save(roper_emb_path, roper_emb)
        log.info(f'Saved Roper embeddings → {roper_emb_path}')

    log.info(f'GSS emb {gss_emb.shape}, Roper emb {roper_emb.shape}')
    return gss_emb, roper_emb


def _cosine_match(gss_questions, roper_unique_questions, gss_emb, roper_emb):
    gss_norm   = gss_emb / np.linalg.norm(gss_emb, axis=1, keepdims=True)
    roper_norm = roper_emb / np.linalg.norm(roper_emb, axis=1, keepdims=True)

    matches = []
    for i in tqdm(range(0, len(gss_norm), CHUNK_SIZE), desc='Cosine similarity'):
        chunk = gss_norm[i:i + CHUNK_SIZE]
        sim = chunk @ roper_norm.T
        gi_arr, ri_arr = np.where(sim >= COS_THRESHOLD)
        for gi, ri in zip(gi_arr, ri_arr):
            row = gss_questions.iloc[i + gi]
            matches.append({
                'gss_question_id': int(row['question_id']),
                'gss_variable': row['variable'],
                'gss_question': row['question'],
                'gss_binarized': row.get('binarized', ''),
                'roper_question': roper_unique_questions[ri],
                'roper_question_idx': int(ri),
                'cos_sim': float(sim[gi, ri]),
            })

    df = pd.DataFrame(matches).sort_values('cos_sim', ascending=False)
    log.info(f'Matches with cos_sim >= {COS_THRESHOLD}: {len(df)}')
    return df


def stage_a_matching(force=False):
    df_full_path   = COLAB_DIR / 'df_matches_full.pkl'
    verify_in_path = COLAB_DIR / 'verify_input.csv'
    bin_in_path    = COLAB_DIR / 'binarize_input.parquet'

    if (not force and df_full_path.exists()
            and verify_in_path.exists() and bin_in_path.exists()):
        log.info('[stage A] outputs exist — skipping (--force to recompute).')
        return

    log.info('[stage A] loading GSS + Roper …')
    gss_questions = _load_gss_questions()
    df_roper = _load_roper_questions()

    df_roper_unique = (
        df_roper.dropna(subset=['question'])
                .drop_duplicates('question')
                .reset_index(drop=True)
    )
    roper_unique_questions = df_roper_unique['question'].values
    log.info(f'Unique Roper questions: {len(roper_unique_questions)}')

    gss_emb, roper_emb = _compute_or_load_embeddings(
        gss_questions, roper_unique_questions)

    df_matches = _cosine_match(
        gss_questions, roper_unique_questions, gss_emb, roper_emb)

    roper_meta = df_roper[['question', 'responses', 'year', 'roper_question_id',
                           'roper_study_id', 'studyTitle', 'surveyBy',
                           'conductedBy', 'national_adult']].copy()
    df_matches_full = df_matches.merge(
        roper_meta, left_on='roper_question', right_on='question', how='left',
        suffixes=('', '_roper')
    ).drop(columns=['question'], errors='ignore')
    log.info(f'Matches with metadata: {len(df_matches_full)}')

    # (a) verify_input.csv — unique (gss_variable, roper_question)
    df_verify_in = (
        df_matches.drop_duplicates(subset=['gss_variable', 'roper_question'])
                  [['gss_variable', 'gss_question', 'gss_binarized',
                    'roper_question', 'cos_sim']]
                  .reset_index(drop=True)
    )
    df_verify_in.to_csv(verify_in_path, index=False)
    log.info(f'verify_input.csv {df_verify_in.shape} → {verify_in_path}')

    # (b) binarize_input.parquet — match rows with per-year Roper responses
    df_bin_in = df_matches_full[
        ['gss_variable', 'gss_question', 'gss_binarized', 'roper_question',
         'responses', 'year', 'roper_question_id', 'cos_sim']
    ].copy()
    df_bin_in = df_bin_in[df_bin_in['responses'].apply(
        lambda x: isinstance(x, list) and len(x) > 0)]
    df_bin_in['responses'] = df_bin_in['responses'].apply(json.dumps)
    df_bin_in.to_parquet(bin_in_path, index=False)
    log.info(f'binarize_input.parquet {df_bin_in.shape} → {bin_in_path}')

    # (c) df_matches_full.pkl
    df_matches_full.to_pickle(df_full_path)
    log.info(f'df_matches_full.pkl {df_matches_full.shape} → {df_full_path}')

    # Canonical CSV artifact
    df_matches_full.to_csv(SAVE_DIR / 'roper_cosine_matches.csv', index=False)
    log.info(f'roper_cosine_matches.csv → {SAVE_DIR}')


# --- Stage B/C: Gemini batch API --------------------------------------------

def _get_gemini_client():
    try:
        from google import genai  # noqa
    except ImportError:
        log.error('google-genai not installed. `pip install google-genai`.')
        return None
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        log.error('GEMINI_API_KEY env var is not set; cannot run LLM stages.')
        return None
    from google import genai
    return genai.Client(api_key=api_key)


def _build_verification_prompt(gss_q, roper_q):
    return f'''Task:
You are given two survey questions. Determine whether a positive response (e.g., "yes", "agree") to Survey Question 1 would generally imply a positive response to Survey Question 2.

Output Format:
Return ONLY a valid JSON object in the following form:
{{"answer": "Yes" or "No", "confidence": <float between 0.0 and 1.0>}}

Survey Question 1: "{gss_q}"
Survey Question 2: "{roper_q}"
'''


def _build_binarization_prompt(gss_q, gss_binarized, roper_q, roper_responses):
    resp_str = ', '.join([f'"{r[1]}" ({r[2]}%)' for r in roper_responses])
    return f'''Task:
You are given a GSS survey question with its binarization mapping, and a matched Roper survey question with response options and their percentages.

Map each Roper response to 1 (positive/agree) or 0 (negative/disagree) following the GSS binarization logic.

GSS Question: "{gss_q}"
GSS Binarization: {gss_binarized}

Roper Question: "{roper_q}"
Roper Responses: {resp_str}

Output Format:
Return ONLY a valid JSON object:
{{"mapping": {{"<response_text>": 0 or 1 or null, ...}}, "roper_yes_pct": <sum of percentages for responses mapped to 1>}}

Rules:
- 1 = aligns with GSS binarized=1 (positive/yes/agree)
- 0 = aligns with GSS binarized=0 (negative/no/disagree)
- null = "Don't know", "No opinion", "Refused", "Not sure", etc.
- roper_yes_pct = sum of percentage for all responses mapped to 1
'''


def _upload_jsonl_batches(client, prompts, prefix, work_dir):
    from google.genai import types
    n_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    log.info(f'{prefix}: {len(prompts)} prompts → {n_batches} batches')

    uploaded = []
    for b in range(n_batches):
        start = b * BATCH_SIZE
        chunk = prompts[start:start + BATCH_SIZE]
        jsonl_path = work_dir / f'{prefix}_batch_{b:03d}.jsonl'
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for i, prompt in enumerate(chunk):
                req = {
                    'key': f'{prefix}-{b}-{i}',
                    'request': {
                        'contents': [{'parts': [{'text': prompt}]}],
                        'generation_config': {
                            'response_mime_type': 'application/json',
                            'max_output_tokens': 2000,
                            'temperature': 0.2,
                        },
                    }
                }
                f.write(json.dumps(req, ensure_ascii=False) + '\n')
        up = client.files.upload(
            file=str(jsonl_path),
            config=types.UploadFileConfig(
                display_name=f'{prefix}_batch_{b:03d}',
                mime_type='jsonl',
            )
        )
        uploaded.append(up.name)
        log.info(f'  [{b+1}/{n_batches}] uploaded → {up.name}')
        time.sleep(1)
    return uploaded


def _wait_for_any_batch_to_complete(client):
    while True:
        batches = list(client.batches.list())
        active = [b for b in batches if b.state.name not in COMPLETED_STATES]
        if not active:
            return
        time.sleep(POLL_SEC)
        new_active = [b for b in list(client.batches.list())
                      if b.state.name not in COMPLETED_STATES]
        if len(new_active) < len(active):
            return


def _submit_and_wait(client, uploaded_files, prefix):
    from google import genai
    submitted, failed = [], []
    for idx, file_id in enumerate(uploaded_files):
        log.info(f'Submitting {prefix} batch {idx+1}/{len(uploaded_files)}')
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                job = client.batches.create(
                    model=GEMINI_MODEL, src=file_id,
                    config={'display_name': f'{prefix}_batch_{idx:03d}'},
                )
                submitted.append(job.name)
                log.info(f'  → {job.name}')
                break
            except genai.errors.ClientError as e:
                log.warning(f'  [attempt {attempt}] client error: {e}')
                if attempt < MAX_RETRIES:
                    _wait_for_any_batch_to_complete(client)
                    time.sleep(RETRY_DELAY_SEC)
                else:
                    failed.append(idx)

    log.info(f'Waiting for {len(submitted)} {prefix} batch(es) to complete …')
    while True:
        batches = list(client.batches.list())
        active = [b for b in batches if b.state.name not in COMPLETED_STATES
                  and b.name in submitted]
        if not active:
            break
        log.info(f'  {len(active)} batch(es) still running …')
        time.sleep(POLL_SEC)

    log.info(f'{prefix} batches complete. Failed submissions: {len(failed)}')
    all_batches = list(client.batches.list())
    return [b for b in all_batches if b.name in submitted]


def _download_batch_results(client, batches, prefix, work_dir):
    results = []
    for b in tqdm(sorted(batches, key=lambda x: x.display_name),
                  desc=f'Downloading {prefix}'):
        if b.state.name != 'JOB_STATE_SUCCEEDED':
            log.warning(f'[SKIP] {b.display_name} — {b.state.name}')
            continue
        content = client.files.download(file=b.dest.file_name).decode('utf-8')
        (work_dir / f'{prefix}_output_{b.display_name}.txt').write_text(content)
        for line in content.strip().split('\n'):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get('key', '')
            try:
                text = row['response']['candidates'][0]['content']['parts'][0]['text']
                parsed = json.loads(text)
            except (KeyError, IndexError, json.JSONDecodeError):
                parsed = {}
            results.append({'key': key, 'result': parsed})
    return results


def stage_b_verify(client, force=False):
    out_path = COLAB_DIR / 'verify_output.csv'
    if not force and out_path.exists():
        log.info('[stage B] verify_output.csv exists — skipping.')
        return

    in_path = COLAB_DIR / 'verify_input.csv'
    if not in_path.exists():
        log.error(f'[stage B] missing {in_path}. Run stage A first.')
        sys.exit(2)

    df_verify_in = pd.read_csv(in_path)
    log.info(f'[stage B] verify_input: {df_verify_in.shape}')

    prompts = [
        _build_verification_prompt(row['gss_question'], row['roper_question'])
        for _, row in df_verify_in.iterrows()
    ]
    uploaded = _upload_jsonl_batches(client, prompts, 'verify', COLAB_DIR)
    batches = _submit_and_wait(client, uploaded, 'verify')
    results = _download_batch_results(client, batches, 'verify', COLAB_DIR)

    results.sort(key=lambda x: (
        int(x['key'].split('-')[1]), int(x['key'].split('-')[2]),
    ))
    df_verify_in['llm_same'] = [r['result'].get('answer') for r in results]
    df_verify_in['llm_confidence'] = [
        r['result'].get('confidence', 0.0) for r in results]

    yes_cnt = int((df_verify_in.llm_same == 'Yes').sum())
    no_cnt  = int((df_verify_in.llm_same == 'No').sum())
    log.info(f'[stage B] Yes={yes_cnt}, No={no_cnt}')

    df_verify_in[[
        'gss_variable', 'roper_question', 'llm_same', 'llm_confidence',
    ]].to_csv(out_path, index=False)
    log.info(f'[stage B] wrote {out_path}')


def stage_c_binarize(client, force=False):
    out_path = COLAB_DIR / 'binarize_output.csv'
    if not force and out_path.exists():
        log.info('[stage C] binarize_output.csv exists — skipping.')
        return

    bin_path    = COLAB_DIR / 'binarize_input.parquet'
    verify_path = COLAB_DIR / 'verify_output.csv'
    if not bin_path.exists() or not verify_path.exists():
        log.error(f'[stage C] need {bin_path} and {verify_path}. Abort.')
        sys.exit(2)

    df_bin_in = pd.read_parquet(bin_path)
    df_bin_in['responses'] = df_bin_in['responses'].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x)
    log.info(f'[stage C] binarize_input: {df_bin_in.shape}')

    df_verify_out = pd.read_csv(verify_path)
    verified_pairs = set(
        df_verify_out.loc[df_verify_out.llm_same == 'Yes',
                          ['gss_variable', 'roper_question']]
                     .apply(tuple, axis=1)
    )
    df_bin_in = df_bin_in[
        df_bin_in[['gss_variable', 'roper_question']]
                 .apply(tuple, axis=1)
                 .isin(verified_pairs)
    ].copy()
    log.info(f'After verification filter: {df_bin_in.shape}')

    df_bin_in['resp_key'] = df_bin_in['responses'].apply(
        lambda rs: '|'.join(sorted(r[1] for r in rs))
    )
    df_bin_unique = df_bin_in.drop_duplicates(
        subset=['gss_variable', 'resp_key']).copy()
    log.info(f'Unique binarization prompts: {len(df_bin_unique)}')

    prompts = [
        _build_binarization_prompt(
            row['gss_question'], row['gss_binarized'],
            row['roper_question'], row['responses'],
        )
        for _, row in df_bin_unique.iterrows()
    ]
    uploaded = _upload_jsonl_batches(client, prompts, 'binarize', COLAB_DIR)
    batches = _submit_and_wait(client, uploaded, 'binarize')
    results = _download_batch_results(client, batches, 'binarize', COLAB_DIR)

    results.sort(key=lambda x: (
        int(x['key'].split('-')[1]), int(x['key'].split('-')[2]),
    ))
    df_bin_unique['bin_mapping']   = [r['result'].get('mapping') for r in results]
    df_bin_unique['roper_yes_pct'] = [r['result'].get('roper_yes_pct') for r in results]
    log.info(f'Successfully binarized: '
             f'{int(df_bin_unique.roper_yes_pct.notna().sum())}')

    df_bin_unique['bin_mapping'] = df_bin_unique['bin_mapping'].apply(
        lambda m: json.dumps(m) if m is not None else None)
    df_bin_unique[[
        'gss_variable', 'resp_key', 'bin_mapping', 'roper_yes_pct',
    ]].to_csv(out_path, index=False)
    log.info(f'[stage C] wrote {out_path}')


# --- Entry point -------------------------------------------------------------

def _setup_logging():
    logs_dir = BASE_DIR / 'logs'
    logs_dir.mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_file = logs_dir / f'{ts}_roper_build.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logging.getLogger().info(f'Log → {log_file}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true',
                        help='Recompute a stage even if its outputs exist.')
    parser.add_argument('--stages', default='a,b,c',
                        help='Comma-separated stages to run (a|b|c).')
    parser.add_argument('--skip-llm', action='store_true',
                        help='Shortcut: only run stage A.')
    args = parser.parse_args()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    COLAB_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging()

    stages = {s.strip() for s in args.stages.split(',')}
    if args.skip_llm:
        stages -= {'b', 'c'}

    if 'a' in stages:
        stage_a_matching(force=args.force)

    if stages & {'b', 'c'}:
        client = _get_gemini_client()
        if client is None:
            log.info('Skipping LLM stages (no API key or SDK missing).')
            return
        if 'b' in stages:
            stage_b_verify(client, force=args.force)
        if 'c' in stages:
            stage_c_binarize(client, force=args.force)

    log.info('Done.')


if __name__ == '__main__':
    main()

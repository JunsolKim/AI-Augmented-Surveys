#!/usr/bin/env python
import glob
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

BASE = Path(__file__).resolve().parents[1]
JSON_DIR = BASE / 'data' / 'roper' / 'question' / 'json'
CSV_DIR = BASE / 'data' / 'roper' / 'csv' / 'question'
OUT = BASE / 'data' / 'roper_prep' / 'roper_question_meta.parquet'

US_COUNTRY_ID = 'da81340e-5e20-454e-ab85-052e698c2f57'


def main():
    files = sorted(glob.glob(str(JSON_DIR / 'all_questions_*.json')))
    if not files:
        raise SystemExit(
            f'No Roper question JSON found under {JSON_DIR}.\n'
            'This one-time builder needs the raw Roper iPoll scrape, which '
            'the Roper license does not allow the replication archive to '
            'include. Its output ships in the OSF data as '
            f'{OUT.relative_to(BASE)} and is what downstream scripts read; '
            'rerunning this script is only needed to rebuild the Roper '
            'matching from scratch.')
    rows = []
    for fp in tqdm(files, desc='Loading Roper JSON'):
        with open(fp) as fh:
            txt = fh.read().strip()
        if not txt:
            continue
        try:
            qs = json.loads(txt)
            if not isinstance(qs, list):
                continue
        except json.JSONDecodeError:
            qs = [json.loads(line) for line in txt.splitlines() if line.strip()]
        for q in qs:
            if not isinstance(q, dict):
                continue
            topline = q.get('topline') or ''
            study = q.get('study') or {}
            if not isinstance(study, dict):
                study = {}
            rows.append({
                'roper_question_id': q.get('id'),
                'national_adult': int('national adult' in str(topline).lower()),
                'roper_study_id': study.get('id'),
                'study_title': study.get('title'),
            })
    df = pd.DataFrame(rows).drop_duplicates('roper_question_id').reset_index(drop=True)

    df['gss'] = df['study_title'].fillna('').str.lower().str.contains(
        'general social survey').astype(int)

    # Merge in per-question CSV metadata (surveyBy, conductedBy, country)
    csv_files = sorted(glob.glob(str(CSV_DIR / 'ipoll_question_results_*.csv')))
    csv_dfs = []
    for f in tqdm(csv_files, desc='Loading Roper CSVs'):
        try:
            csv_dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            continue
    if csv_dfs:
        df_csv = pd.concat(csv_dfs, ignore_index=True)
        df_csv['united_states'] = df_csv.get('countryIds', '').apply(
            lambda x: int(US_COUNTRY_ID in str(x)))
        keep_cols = [c for c in
                     ['id', 'studyTitle', 'surveyBy', 'conductedBy', 'united_states']
                     if c in df_csv.columns]
        df_csv = df_csv[keep_cols].drop_duplicates('id')
        df = df.merge(df_csv, left_on='roper_question_id', right_on='id',
                      how='left').drop(columns=['id'], errors='ignore')
        # Prefer CSV studyTitle when present
        if 'studyTitle' in df.columns:
            df['study_title'] = df['studyTitle'].fillna(df['study_title'])
            df = df.drop(columns=['studyTitle'])
        df = df.rename(columns={'surveyBy': 'survey_by',
                                'conductedBy': 'conducted_by'})
    else:
        df['survey_by'] = None
        df['conducted_by'] = None
        df['united_states'] = None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f'rows: {len(df):,}, national_adult==1: {int(df.national_adult.sum()):,}, '
          f'gss==1: {int(df.gss.sum()):,}, '
          f'united_states==1: {int((df.get("united_states", pd.Series([])) == 1).sum()):,}')
    print(f'saved: {OUT}')


if __name__ == '__main__':
    main()

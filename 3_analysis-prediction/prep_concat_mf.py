#!/usr/bin/env python
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / 'data'
OUT_DIR    = DATA_DIR / 'fig_table_gen'
PRED_DIR   = DATA_DIR / 'predictions'

N_FOLDS_TOT      = 10
FOLDS_USE        = [0, 1, 2]
RANDOM_STATE     = 42
DIM              = 50
R_LAMBDA         = 10.0
N_ITER           = 10
SCENARIOS        = ['impute', 'partial']
EMB_TYPES        = ['tfidf', 'sbert']

log = logging.getLogger('prep_concat_mf')


def _setup_logging():
    logs_dir = BASE_DIR / 'logs'
    logs_dir.mkdir(exist_ok=True)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_file = logs_dir / f'{ts}_prep_concat_mf.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    log.info(f'Log -> {log_file}')


def load_mf_long(scenario, fold):
    path = PRED_DIR / f'mf_{scenario}_{fold}_{N_FOLDS_TOT}_128_50__resample1_long.parquet'
    return pd.read_parquet(path)


def load_alpaca_long(scenario, fold):
    path = PRED_DIR / (
        f'maicomputer_alpaca-native_{scenario}_{fold}_{N_FOLDS_TOT}_128_50'
        f'__resample1_long.parquet')
    return pd.read_parquet(path)


def load_question_texts(df_slim):
    vq = pickle.load(open(DATA_DIR / 'var_question_dict.pkl', 'rb'))
    qid_to_var = (df_slim[['question_id', 'variable']].drop_duplicates()
                  .set_index('question_id')['variable'].to_dict())
    qids = sorted(qid_to_var.keys())
    texts = [str(vq.get(qid_to_var[q], qid_to_var[q])) for q in qids]
    return qids, texts


def build_F_tfidf(qids, texts):
    """Raw TF-IDF features (no dim reduction)."""
    tfv = TfidfVectorizer(max_features=4000, min_df=3,
                          stop_words='english', ngram_range=(1, 2))
    X_tfidf = tfv.fit_transform(texts)
    F = X_tfidf.toarray().astype(np.float32)
    qid_to_row = {q: i for i, q in enumerate(qids)}
    log.info(f'F (TF-IDF raw): {F.shape}')
    return F, qid_to_row


def build_F_sbert(qids, texts):
    """Raw SBERT-384 features (no dim reduction)."""
    cache_path = OUT_DIR / '_cache_sbert_q_embed.parquet'
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if list(cached['question_id']) == list(qids):
            emb_cols = [c for c in cached.columns if c != 'question_id']
            F = cached[emb_cols].to_numpy(dtype=np.float32)
            log.info(f'SBERT cache hit: {F.shape}')
            qid_to_row = {q: i for i, q in enumerate(qids)}
            return F, qid_to_row
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    F = model.encode(texts, show_progress_bar=False, batch_size=256,
                     normalize_embeddings=True).astype(np.float32)
    qid_to_row = {q: i for i, q in enumerate(qids)}
    log.info(f'F (SBERT raw-384): {F.shape}')
    return F, qid_to_row


def concat_als(R, F, dim=DIM, n_iter=N_ITER, r_lambda=R_LAMBDA, seed=RANDOM_STATE):
    """Weighted ALS on augmented [R; F^T]."""
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape
    assert F.shape[0] == n_items, f'F shape {F.shape} items != {n_items}'
    n_feat = F.shape[1]

    Rc = R.tocsr(); Rc.sort_indices()
    Rcc = R.tocsc(); Rcc.sort_indices()
    F_T = np.asfortranarray(F.T)

    X_users = rng.normal(scale=1.0 / dim, size=(n_users, dim)).astype(np.float32)
    X_feat  = rng.normal(scale=1.0 / dim, size=(n_feat, dim)).astype(np.float32)
    Y       = rng.normal(scale=1.0 / dim, size=(n_items, dim)).astype(np.float32)
    lam_I = (r_lambda * np.eye(dim)).astype(np.float32)

    for epoch in range(n_iter):
        t0 = time.time()
        for u in range(n_users):
            s, e = Rc.indptr[u], Rc.indptr[u + 1]
            if e == s:
                continue
            obs = Rc.indices[s:e]
            p = Rc.data[s:e].astype(np.float32)
            Yo = Y[obs]
            A = Yo.T @ Yo + lam_I
            b = Yo.T @ p
            X_users[u] = np.linalg.solve(A, b)
        t_u = time.time() - t0

        t0 = time.time()
        A_feat = (Y.T @ Y + lam_I).astype(np.float32)
        B_feat = Y.T @ F_T.T
        X_feat = np.linalg.solve(A_feat, B_feat).T.astype(np.float32)
        t_f = time.time() - t0

        t0 = time.time()
        Xfeat_YY = (X_feat.T @ X_feat).astype(np.float32)
        for i in range(n_items):
            s, e = Rcc.indptr[i], Rcc.indptr[i + 1]
            if e == s:
                Xo_YY = 0.0
                bu = np.zeros(dim, dtype=np.float32)
            else:
                u_obs = Rcc.indices[s:e]
                p_u = Rcc.data[s:e].astype(np.float32)
                Xo = X_users[u_obs]
                Xo_YY = Xo.T @ Xo
                bu = Xo.T @ p_u
            A = Xo_YY + Xfeat_YY + lam_I
            b = bu + X_feat.T @ F_T[:, i]
            Y[i] = np.linalg.solve(A, b)
        t_i = time.time() - t0

        log.info(f'  epoch {epoch+1}/{n_iter}  X_u={t_u:.1f}s  '
                 f'X_f={t_f:.1f}s  Y={t_i:.1f}s')

    return X_users, X_feat, Y


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging()

    df_slim = pd.read_pickle(DATA_DIR / 'df_analysis_slim.pkl')
    gt_map = (
        df_slim.loc[df_slim['binarized'].notna(),
                    ['yearid_id', 'question_id', 'binarized']]
        .drop_duplicates(['yearid_id', 'question_id'])
    )
    log.info(f'df_slim ground truth rows: {len(gt_map):,}')

    qids, qtexts = load_question_texts(df_slim)
    F_by_emb = {
        'tfidf': build_F_tfidf(qids, qtexts),
        'sbert': build_F_sbert(qids, qtexts),
    }

    n_users = int(df_slim['yearid_id'].max()) + 1
    n_items = int(df_slim['question_id'].max()) + 1
    log.info(f'n_users={n_users}, n_items={n_items}')

    records = []
    for scenario in SCENARIOS:
        log.info(f'=== scenario: {scenario} ===')
        for k in FOLDS_USE:
            log.info(f'--- fold {k} ---')
            mf_long = load_mf_long(scenario, k)
            val_col, resp_col = f'validation_{k}', f'response_{k}'
            mf_long = mf_long.merge(gt_map,
                                    on=['yearid_id', 'question_id'], how='inner')

            train_cells = mf_long.loc[mf_long[val_col] == 0,
                                      ['yearid_id', 'question_id',
                                       'binarized']].reset_index(drop=True)
            test_cells  = mf_long.loc[mf_long[val_col] == 1,
                                      ['yearid_id', 'question_id', resp_col,
                                       'binarized']].reset_index(drop=True)
            log.info(f'  train cells {len(train_cells):,}, '
                     f'test cells {len(test_cells):,}')

            rows = train_cells['yearid_id'].to_numpy()
            cols = train_cells['question_id'].to_numpy()
            data = train_cells['binarized'].astype(np.float32).to_numpy()
            R = csr_matrix((data, (rows, cols)),
                           shape=(n_users, n_items))
            log.info(f'  R nnz={R.nnz:,}, density={R.nnz/(n_users*n_items):.2e}')

            alp = load_alpaca_long(scenario, k)
            alp = alp.loc[alp[val_col] == 1,
                          ['yearid_id', 'question_id', resp_col]]
            alp_j = test_cells[['yearid_id', 'question_id']].merge(
                alp, on=['yearid_id', 'question_id'], how='left')
            pred_al = alp_j[resp_col].to_numpy(dtype=np.float32)

            pred_mf = test_cells[resp_col].to_numpy(dtype=np.float32)

            y_te = test_cells['binarized'].to_numpy(dtype=np.int32)
            u_te = test_cells['yearid_id'].to_numpy()
            q_te = test_cells['question_id'].to_numpy()

            for emb_type in EMB_TYPES:
                F, _ = F_by_emb[emb_type]
                log.info(f'  concat_{emb_type} MF: training ({N_ITER} iters) ...')
                t0 = time.time()
                X_users, X_feat, Y = concat_als(R, F)
                dt_train = time.time() - t0
                log.info(f'    train done in {dt_train:.1f}s')

                pred_cm = np.einsum('ij,ij->i', X_users[u_te], Y[q_te])
                pred_cm = pred_cm.astype(np.float32)

                ok = (~np.isnan(pred_al) & ~np.isnan(pred_mf) & ~np.isnan(pred_cm))
                y_ok = y_te[ok]

                def metr(p):
                    p = p[ok]
                    try:
                        auc = roc_auc_score(y_ok, p)
                    except ValueError:
                        auc = np.nan
                    acc = accuracy_score(y_ok, (p >= 0.5).astype(int))
                    return auc, acc

                auc_cm, acc_cm = metr(pred_cm)

                log.info(f'    concat_{emb_type} AUC={auc_cm:.4f} '
                         f'acc={acc_cm:.4f}  n={int(ok.sum()):,}')

                records.append({
                    'scenario': scenario, 'fold': k,
                    'model':    f'concat_mf_{emb_type}',
                    'auc':      auc_cm, 'acc': acc_cm,
                    'n':        int(ok.sum()),
                })

            auc_mf, acc_mf = roc_auc_score(y_te, pred_mf), \
                             accuracy_score(y_te, (pred_mf >= 0.5).astype(int))
            auc_al, acc_al = roc_auc_score(y_te, pred_al), \
                             accuracy_score(y_te, (pred_al >= 0.5).astype(int))
            records += [
                {'scenario': scenario, 'fold': k, 'model': 'mf',
                 'auc': auc_mf, 'acc': acc_mf, 'n': len(y_te)},
                {'scenario': scenario, 'fold': k, 'model': 'alpaca',
                 'auc': auc_al, 'acc': acc_al, 'n': len(y_te)},
            ]

    out = pd.DataFrame(records)
    out_path = OUT_DIR / 'concat_mf.parquet'
    out.to_parquet(out_path, index=False)
    log.info(f'Saved {out.shape} -> {out_path}')

    log.info('Per-scenario means:')
    summ = (out.groupby(['scenario', 'model'])[['auc', 'acc']]
            .mean().round(4))
    log.info('\n' + summ.to_string())


if __name__ == '__main__':
    main()

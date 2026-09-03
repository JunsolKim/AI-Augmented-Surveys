import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = DATA_DIR / "weights"
MODELS_DIR = BASE_DIR / "data" / "models"

sys.path.insert(0, str(BASE_DIR / "1_data-preprocessing"))
sys.path.insert(0, str(BASE_DIR / "2_model-finetuning"))


def load_question_dict():
    """Re-export step9's loader to guarantee parity."""
    import step9_generate_embeddings as s9
    return s9._load_question_dict()


def build_prompt(question_text: str) -> str:
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction: {question_text}\n\n### Response: "
    )


def build_all_prompts():
    """Build prompts for every question_id in order (matches step9._build_prompts)."""
    import step9_generate_embeddings as s9
    qdict, n_questions = s9._load_question_dict()
    return s9._build_prompts(qdict, n_questions), n_questions


def load_intersection_train(k: int = 0, n_splits: int = 10):
    """Strict-intersection training set across {impute, partial, total} splits.

    A row (yearid_id, question_id) is kept iff it is in the TRAINING split of
    all three split_types simultaneously. This guarantees Stage 0/1 never sees
    any cell that is held-out for ANY of the three downstream evaluations, so
    the resulting embeddings are leakage-free for all of impute/partial/total.

    The monitoring validation set is the INTERSECTION of the three val sets —
    rows held out under ALL three split schemes simultaneously.

    Returns
    -------
    df_analysis        : pd.DataFrame  (same df_analysis used by all 3 splits)
    train              : pd.DataFrame  (intersection train, ~73% of df_analysis)
    val_intersection   : pd.DataFrame  (val_impute ∩ val_partial ∩ val_total)
    """
    from utils import load_data  # local to avoid TF init at module load

    splits = ("total", "partial", "impute")
    train_sets = {}
    val_sets = {}
    df_analysis = None
    for s in splits:
        df_a, tr, va = load_data(
            split_type=s, k=k, n_splits=n_splits,
            resample=1, use_demo=False, use_poli=True,
        )
        train_sets[s] = tr
        val_sets[s] = va
        if df_analysis is None:
            df_analysis = df_a

    # Intersect train rows on (yearid_id, question_id) — unique key after drop_duplicates.
    train_keys = train_sets["total"][["yearid_id", "question_id"]] \
        .merge(train_sets["partial"][["yearid_id", "question_id"]],
               on=["yearid_id", "question_id"]) \
        .merge(train_sets["impute"][["yearid_id", "question_id"]],
               on=["yearid_id", "question_id"])
    intersection_train = train_sets["total"].merge(
        train_keys, on=["yearid_id", "question_id"], how="inner",
    ).reset_index(drop=True)

    # Intersect val rows the same way — strict held-out under ALL three schemes.
    val_keys = val_sets["total"][["yearid_id", "question_id"]] \
        .merge(val_sets["partial"][["yearid_id", "question_id"]],
               on=["yearid_id", "question_id"]) \
        .merge(val_sets["impute"][["yearid_id", "question_id"]],
               on=["yearid_id", "question_id"])
    val_intersection = val_sets["total"].merge(
        val_keys, on=["yearid_id", "question_id"], how="inner",
    ).reset_index(drop=True)

    return df_analysis, intersection_train, val_intersection


BASE_LLM = "maicomputer/alpaca-native"

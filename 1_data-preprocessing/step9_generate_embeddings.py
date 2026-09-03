import os
import pickle
import random
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = DATA_DIR / "weights"

os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_question_dict():
    """Load question_id -> question text mapping (survey + demographic questions)."""
    df_analysis = pd.read_pickle(DATA_DIR / "df_analysis.pkl")
    df_analysis = df_analysis.loc[
        df_analysis.binarized.notna() & df_analysis.yearid_id.notna() & df_analysis.question_id.notna()
    ]

    # Append demographic questions (they get question_ids after the survey questions)
    demo_path = DATA_DIR / "demographics_long.csv"
    if not demo_path.exists():
        demo_path = DATA_DIR / "demographics.parquet"
    if demo_path.suffix == ".csv":
        demographics_long = pd.read_csv(demo_path)
    else:
        demographics_long = pd.read_parquet(demo_path)
    demographics_long['question_id'] = demographics_long.question_id + df_analysis.question_id.max() + 1

    df_all = pd.concat([df_analysis, demographics_long], axis=0, ignore_index=True)
    question_dict = dict(zip(df_all['question_id'], df_all['question']))
    n_questions = max(df_all['question_id']) + 1
    return question_dict, n_questions


def _build_prompts(question_dict, n_questions):
    """Build instruction prompts for each question_id."""
    prompts = []
    for i in range(n_questions):
        q = question_dict.get(i, "")
        prompts.append(
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction: {q}\n\n### Response: "
        )
    return prompts


def _save(embeddings, model_name):
    """Save embeddings as pickle."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    out_path = WEIGHTS_DIR / f"{safe_name}.pkl"
    pickle.dump(embeddings, open(out_path, 'wb'))
    print(f"Saved {embeddings.shape} -> {out_path}")


def alpaca(model_name="chavinlo/alpaca-native", cache_dir=None, batch_size=32):
    """Extract last-token hidden states from a causal LM (Alpaca/LLaMA).

    Uses: outputs.hidden_states[-1][:, -1, :]
    """
    question_dict, n_questions = _load_question_dict()
    prompts = _build_prompts(question_dict, n_questions)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(DEVICE)
    model.eval()
    print(f"Using device: {DEVICE}")

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Embedding ({model_name})"):
            batch = prompts[i:i + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            outputs = model(**encoded, output_hidden_states=True)
            # Get last-token embedding per sequence (use attention mask to find last real token)
            seq_lens = encoded["attention_mask"].sum(dim=1) - 1
            hidden = outputs.hidden_states[-1]
            emb = hidden[torch.arange(hidden.size(0)), seq_lens].cpu().numpy()
            embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    _save(embeddings, model_name)


def gpt(model_name="EleutherAI/gpt-j-6B", cache_dir=None, batch_size=32):
    """Extract last-token hidden states from a GPT-style model.

    Uses: outputs.last_hidden_state[:, -1, :]
    """
    question_dict, n_questions = _load_question_dict()
    prompts = _build_prompts(question_dict, n_questions)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(DEVICE)
    model.eval()
    print(f"Using device: {DEVICE}")

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Embedding ({model_name})"):
            batch = prompts[i:i + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            outputs = model(**encoded)
            seq_lens = encoded["attention_mask"].sum(dim=1) - 1
            hidden = outputs.last_hidden_state
            emb = hidden[torch.arange(hidden.size(0)), seq_lens].cpu().numpy()
            embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    _save(embeddings, model_name)


def bert(model_name="roberta-large", cache_dir=None, batch_size=64):
    """Extract pooler output from an encoder model (BERT/RoBERTa).

    Uses: outputs[1]  (pooler_output)
    """
    question_dict, n_questions = _load_question_dict()
    prompts = _build_prompts(question_dict, n_questions)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(DEVICE)
    model.eval()
    print(f"Using device: {DEVICE}")

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Embedding ({model_name})"):
            batch = prompts[i:i + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            outputs = model(**encoded)
            emb = outputs[1].cpu().numpy()
            embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    _save(embeddings, model_name)


if __name__ == "__main__":
    fire.Fire()

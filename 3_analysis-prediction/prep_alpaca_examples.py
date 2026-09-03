from pathlib import Path

import fire
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = DATA / "fig_table_gen"

TARGET_VARS = ["homosex", "marhomo1", "busing", "nomeat"]

PROMPT_TMPL = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction: {q}\n\n### Response: "
)


def _load_question_text(target_vars):
    vm = pd.read_parquet(DATA / "gss_train_vars_corrected_binarized_again.parquet")
    vm = vm[vm["variable_name"].isin(target_vars)].drop_duplicates("variable_name")
    qmap = dict(zip(vm["variable_name"], vm["question"].fillna("")))
    missing = [v for v in target_vars if not qmap.get(v, "").strip()]
    if missing:
        raise RuntimeError(f"missing question text for {missing}")
    return qmap


def main(
    model_name: str = "maicomputer/alpaca-native",
    cache_dir: str | None = None,
    max_new_tokens: int = 80,
):
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    qmap = _load_question_text(TARGET_VARS)

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    rows = []
    for var in TARGET_VARS:
        question = qmap[var]
        prompt = PROMPT_TMPL.format(q=question)
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        response = gen.strip()
        rows.append({"variable": var, "question": question, "response": response})
        print(f"[{var}] {response[:120]}")

    df = pd.DataFrame(rows)
    out = OUT / "alpaca_examples.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved: {out} ({len(df)} rows)")


if __name__ == "__main__":
    fire.Fire(main)

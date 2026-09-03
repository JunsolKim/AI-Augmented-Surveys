import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_VARS_FILE = DATA_DIR / "gss_train_vars_corrected_binarized_again.parquet"


def strip_code_prefix(key: str) -> str:
    """Remove the leading code number from a binarized key (e.g. '1: YES' -> 'YES')."""
    parts = key.split(":", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return key.strip()


def build_prompt(processed_question: str, binarized_json: str) -> str:
    """Build a survey prompt from a processed question and its binarized mapping."""
    binarized = json.loads(binarized_json)

    # Strip code prefixes from labels
    labels = [strip_code_prefix(k) for k in binarized.keys()]
    values = list(binarized.values())

    # Response options list
    options_lines = "\n".join(f"- {label}" for label in labels)

    # Group by binarized value
    ones = [label for label, val in zip(labels, values) if val == 1]
    zeros = [label for label, val in zip(labels, values) if val == 0]

    ones_str = ", ".join(ones)
    zeros_str = ", ".join(zeros)

    prompt = (
        f"Assume you are a survey participant and answer the following survey question.\n"
        f"{processed_question}\n"
        f"\n"
        f"Response options:\n"
        f"{options_lines}\n"
        f"\n"
        f"Respond with only 1 or 0. Do not output anything else.\n"
        f"Respond with 1 if your response is one of the following: {ones_str}.\n"
        f"Respond with 0 if your response is one of the following: {zeros_str}."
    )
    return prompt


def main():
    train_vars = pd.read_parquet(TRAIN_VARS_FILE)
    train_vars = train_vars[train_vars["use_variable_final"] == 1].copy()

    prompts = []
    for _, row in train_vars.iterrows():
        prompt = build_prompt(row["processed_question"], row["binarized"])
        prompts.append({
            "variable_name": row["variable_name"].lower(),
            "processed_question": row["processed_question"],
            "prompt": prompt,
        })

    df_prompts = pd.DataFrame(prompts)
    df_prompts.to_parquet(DATA_DIR / "variable_prompts.parquet", index=False)

    print(f"Generated {len(df_prompts)} prompts -> {DATA_DIR / 'variable_prompts.parquet'}")

    # Print examples
    examples = [0, 5, 50, 100, 500, 1000]
    for i in examples:
        if i < len(df_prompts):
            row = df_prompts.iloc[i]
            print(f"\n{'='*80}")
            print(f"[{i}] variable: {row['variable_name']}")
            print(f"{'='*80}")
            print(row["prompt"])


if __name__ == "__main__":
    main()

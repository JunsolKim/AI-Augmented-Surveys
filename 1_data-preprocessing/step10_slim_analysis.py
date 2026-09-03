from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

KEEP_COLS = [
    'yearid_id',
    'question_id',
    'year_order',
    'binarized',
    'module',
    'variable',
    'yearid',
    'year',
]


def main():
    print("Loading df_analysis.pkl ...")
    df = pd.read_pickle(DATA_DIR / "df_analysis.pkl")
    print(f"  Original: {df.shape}, {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    df_slim = df[KEEP_COLS].copy()

    # Downcast numeric columns to save memory
    for col in ['yearid_id', 'question_id', 'year_order', 'year']:
        df_slim[col] = pd.to_numeric(df_slim[col], downcast='integer')
    df_slim['binarized'] = pd.to_numeric(df_slim['binarized'], downcast='float')

    out_path = DATA_DIR / "df_analysis_slim.pkl"
    df_slim.to_pickle(out_path)
    print(f"  Slim:     {df_slim.shape}, {df_slim.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()

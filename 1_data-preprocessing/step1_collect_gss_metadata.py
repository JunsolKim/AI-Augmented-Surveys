import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm.auto import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://3ilfsaj2lj.execute-api.us-east-1.amazonaws.com/prod/variables/guest-details/{vid}?workspace_id=NaN"
HEADERS = {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:32.0) Gecko/20100101 Firefox/32.0"}
NUM_VARIABLES = 7735
MAX_WORKERS = 40


def fetch_variable(vid: int) -> dict | None:
    """Fetch metadata for a single GSS variable by ID."""
    try:
        resp = requests.get(API_URL.format(vid=vid), headers=HEADERS).json()
        if resp["response"]["result"] == "success":
            return resp["response"]["response"]
    except Exception:
        return None


def main():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        meta_results = list(
            tqdm(executor.map(fetch_variable, range(1, NUM_VARIABLES + 1)), total=NUM_VARIABLES)
        )

    with open(DATA_DIR / "meta_results.pkl", "wb") as f:
        pickle.dump(meta_results, f)

    n_success = sum(1 for r in meta_results if r is not None)
    print(f"Collected {n_success}/{NUM_VARIABLES} variables successfully.")


if __name__ == "__main__":
    main()

import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    with open(DATA_DIR / "meta_results.pkl", "rb") as f:
        meta_results = [r for r in pickle.load(f) if r is not None]

    print(pd.DataFrame(meta_results).groupby("variable_type")["variable_id"].count())

    var_question_dict = {}
    var_description_dict = {}
    online_value_label_dict = {}
    module_dict = {}
    subject_to_var_dict = {}
    var_to_subject_dict = {}

    for item in tqdm(meta_results):
        vname = item["variable_name"]

        # Module
        if item.get("module"):
            module_dict[vname] = item["module"][0]

        # Subject mappings
        if item["subject"] is not None:
            var_to_subject_dict[vname] = item["subject"]
            for subj in item["subject"]:
                subject_to_var_dict.setdefault(subj, [])
                if vname not in subject_to_var_dict[subj]:
                    subject_to_var_dict[subj].append(vname)

        # Variable description
        if item["variable_label"] is not None:
            var_description_dict[vname] = item["variable_label"]

        # Survey question text
        has_question = (
            vname is not None
            and item["survey_questions"] is not None
            and len(item["survey_questions"]) > 0
            and item["survey_questions"][0] is not None
        )
        if has_question:
            var_question_dict[vname] = item["survey_questions"][0]

        # Value labels (discrete variables only)
        if item["variable_type"] == "discrete":
            if "quart_3" in item["summary_by_year"]:
                continue

            current_labels = {}
            for entry in item["summary_by_year"]:
                if float(entry["code_id"]) >= 0 and entry["code_label"] is not None:
                    current_labels[entry["code_id"]] = entry["code_label"].lower()

            if current_labels:
                online_value_label_dict[vname] = current_labels

    # Save all dictionaries
    dicts_to_save = {
        "var_description_dict.pkl": var_description_dict,
        "var_question_dict.pkl": var_question_dict,
        "online_value_label_dict.pkl": online_value_label_dict,
        "module_dict.pkl": module_dict,
        "subject_to_var_dict.pkl": subject_to_var_dict,
        "var_to_subject_dict.pkl": var_to_subject_dict,
    }
    for filename, data in dicts_to_save.items():
        with open(DATA_DIR / filename, "wb") as f:
            pickle.dump(data, f)

    print(f"Available variables with labels: {len(online_value_label_dict)}")
    if online_value_label_dict:
        label_counts = [len(v) for v in online_value_label_dict.values()]
        print(f"Min options per variable: {min(label_counts)}")
        print(f"Max options per variable: {max(label_counts)}")


if __name__ == "__main__":
    main()

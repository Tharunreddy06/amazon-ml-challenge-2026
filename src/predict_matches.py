import csv
import joblib
import sqlite3
from pathlib import Path

import pandas as pd

from matching_features import generate_matching_features
from test_candidate_lookup import generate_candidates

DB_FILE = Path("ML_Challange_datasets/test_source1_index.db")
MODEL_FILE = Path("models/matching_model.pkl")

TEST_FILES = [
    Path("ML_Challange_datasets/test_source2.tsv"),
    Path("ML_Challange_datasets/test_source3.tsv"),
]

OUTPUT_FILE = Path("predictions.csv")


def load_model():
    saved = joblib.load(MODEL_FILE)

    return (
        saved["model"],
        saved["feature_columns"],
        saved["best_threshold"],
    )


def predict_record(
    conn,
    record,
    model,
    feature_columns,
    threshold,
):
    candidates = generate_candidates(
        conn,
        record,
        max_candidates=50,
    )

    if not candidates:
        return None

    feature_rows = []

    for candidate in candidates:
        features = generate_matching_features(
            candidate,
            record,
        )

        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)
    feature_df = feature_df[feature_columns]

    probs = model.predict_proba(feature_df)[:, 1]

    best_idx = probs.argmax()

    if probs[best_idx] < threshold:
        return None

    return {
        "query_entity_id": record["entity_id"],
        "matched_entity_id":
            candidates[best_idx]["entity_id"],
        "probability":
            float(probs[best_idx]),
    }


def process_file(
    conn,
    file_path,
    writer,
    model,
    feature_columns,
    threshold,
):
    print(f"Processing {file_path.name}")

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        count = 0

        for row in reader:

            prediction = predict_record(
                conn,
                row,
                model,
                feature_columns,
                threshold,
            )

            if prediction:
                writer.writerow(prediction)

            count += 1
            if count >= 1000:
                break

            if count % 10000 == 0:
                print(
                    f"Processed {count:,}"
                )


def main():

    model, feature_columns, threshold = load_model()

    conn = sqlite3.connect(DB_FILE)

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_entity_id",
                "matched_entity_id",
                "probability",
            ],
        )

        writer.writeheader()

        for file_path in TEST_FILES:

            process_file(
                conn,
                file_path,
                writer,
                model,
                feature_columns,
                threshold,
            )

    conn.close()

    print("Done")


if __name__ == "__main__":
    main()
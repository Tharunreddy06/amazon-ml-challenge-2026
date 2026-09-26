import csv
import joblib
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import preprocess_record
from matching_features import generate_matching_features
from test_candidate_lookup import generate_candidates


DB_FILE = Path("data/test_source1_index.db")
MODEL_FILE = Path("models/matching_model.pkl")

TEST_SOURCE2 = Path("data/test_source2.tsv")
TEST_SOURCE3 = Path("data/test_source3.tsv")

SAMPLE_SIZE = 100


def load_model():

    saved = joblib.load(MODEL_FILE)

    model = saved["model"]
    feature_columns = saved["feature_columns"]
    threshold = saved["best_threshold"]

    print("Model loaded")
    print(f"Features: {len(feature_columns)}")
    print(f"Threshold: {threshold}")

    return model, feature_columns, threshold


def load_sample(file_path, sample_size):

    records = []

    with file_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t"
        )

        for row in reader:

            records.append(row)

            if len(records) >= sample_size:
                break

    return records


def predict_query(
    conn,
    query_record,
    model,
    feature_columns,
    threshold
):

    candidates = generate_candidates(
        conn,
        query_record,
        max_candidates=50
    )

    if not candidates:
        return []

    feature_rows = []
    candidate_records = []

    for candidate in candidates:

        features = generate_matching_features(
            candidate,
            query_record
        )

        feature_rows.append(features)
        candidate_records.append(candidate)

    feature_df = pd.DataFrame(
        feature_rows
    )

    # Make sure feature order exactly matches training.
    feature_df = feature_df[
        feature_columns
    ]

    probabilities = model.predict_proba(
        feature_df
    )[:, 1]

    predictions = []

    for candidate, probability in zip(
        candidate_records,
        probabilities
    ):

        predictions.append(
            {
                "candidate_entity_id":
                    candidate["entity_id"],

                "candidate_name":
                    candidate["business_name"],

                "candidate_address":
                    candidate["business_address"],

                "candidate_country":
                    candidate["country"],

                "probability":
                    float(probability),

                "is_match":
                    bool(probability >= threshold),
            }
        )

    predictions.sort(
        key=lambda x: x["probability"],
        reverse=True
    )

    return predictions


def main():

    model, feature_columns, threshold = load_model()

    conn = sqlite3.connect(
        DB_FILE
    )

    print("\nLoading test samples...")

    source2_records = load_sample(
        TEST_SOURCE2,
        SAMPLE_SIZE
    )

    source3_records = load_sample(
        TEST_SOURCE3,
        SAMPLE_SIZE
    )

    print(
        f"Source2 samples: {len(source2_records)}"
    )

    print(
        f"Source3 samples: {len(source3_records)}"
    )

    # --------------------------------------------------
    # Test Source2
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("SOURCE2 INFERENCE TEST")
    print("=" * 80)

    for query in source2_records[:10]:

        predictions = predict_query(
            conn,
            query,
            model,
            feature_columns,
            threshold
        )

        print(
            f"\nQuery: {query['entity_id']}"
        )

        print(
            f"Name: {query['business_name']}"
        )

        print(
            f"Country: {query['country']}"
        )

        print(
            f"Candidates: {len(predictions)}"
        )

        for prediction in predictions[:5]:

            print(
                f"  {prediction['candidate_entity_id']} "
                f"| probability="
                f"{prediction['probability']:.4f} "
                f"| match="
                f"{prediction['is_match']}"
            )

            print(
                f"    {prediction['candidate_name']}"
            )

    # --------------------------------------------------
    # Test Source3
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("SOURCE3 INFERENCE TEST")
    print("=" * 80)

    for query in source3_records[:10]:

        predictions = predict_query(
            conn,
            query,
            model,
            feature_columns,
            threshold
        )

        print(
            f"\nQuery: {query['entity_id']}"
        )

        print(
            f"Name: {query['business_name']}"
        )

        print(
            f"Country: {query['country']}"
        )

        print(
            f"Candidates: {len(predictions)}"
        )

        for prediction in predictions[:5]:

            print(
                f"  {prediction['candidate_entity_id']} "
                f"| probability="
                f"{prediction['probability']:.4f} "
                f"| match="
                f"{prediction['is_match']}"
            )

            print(
                f"    {prediction['candidate_name']}"
            )

    conn.close()

    print("\nInference test complete.")


if __name__ == "__main__":
    main()
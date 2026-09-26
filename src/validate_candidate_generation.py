"""
Validate candidate generation against the training ground truth.

This script:
1. Loads a sample of Source1 records with known matches.
2. Loads the corresponding Source2 and Source3 records.
3. Builds the candidate index from Source1.
4. Generates candidates for matched source records.
5. Measures whether the known Source1 entity is recovered.
"""

from pathlib import Path
import pandas as pd

from preprocessing import preprocess_record
from candidate_generation import (
    build_candidate_index,
    generate_candidates
)


DATA_DIR = Path("data")

GROUND_TRUTH_FILE = DATA_DIR / "train_ground_truth.tsv"
SOURCE1_FILE = DATA_DIR / "train_source1.tsv"
SOURCE2_FILE = DATA_DIR / "train_source2.tsv"
SOURCE3_FILE = DATA_DIR / "train_source3.tsv"

SAMPLE_SIZE = 1000
MAX_CANDIDATES = 200


def load_data():
    """Load the required training datasets."""

    print("Loading ground truth...")

    ground_truth = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print("Loading Source1...")

    source1 = pd.read_csv(
        SOURCE1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print("Loading Source2...")

    source2 = pd.read_csv(
        SOURCE2_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print("Loading Source3...")

    source3 = pd.read_csv(
        SOURCE3_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    return ground_truth, source1, source2, source3


def preprocess_dataframe(df):
    """Preprocess all records in a dataframe."""

    records = []

    for record in df.to_dict("records"):
        records.append(preprocess_record(record))

    return records


def build_record_lookup(records):
    """Create entity_id -> record lookup."""

    return {
        record["entity_id"]: record
        for record in records
    }


def main():

    ground_truth, source1, source2, source3 = load_data()

    print("\nDataset sizes:")
    print("Source1:", len(source1))
    print("Source2:", len(source2))
    print("Source3:", len(source3))
    print("Ground truth:", len(ground_truth))

    # Keep only Source1 records that have known matches.
    matched_ground_truth = ground_truth[
        ground_truth["matched_entity_ids"].str.strip() != ""
    ].copy()

    # Use a deterministic sample.
    sample = matched_ground_truth.head(SAMPLE_SIZE)

    print("\nValidation sample:", len(sample))

    print("\nPreprocessing Source1...")

    source1_records = preprocess_dataframe(source1)

    print("Preprocessing Source2...")

    source2_records = preprocess_dataframe(source2)

    print("Preprocessing Source3...")

    source3_records = preprocess_dataframe(source3)

    # Create lookups.
    source1_lookup = build_record_lookup(source1_records)
    source2_lookup = build_record_lookup(source2_records)
    source3_lookup = build_record_lookup(source3_records)

    print("\nBuilding candidate index from Source1...")

    candidate_index = build_candidate_index(source1_records)

    total_true_matches = 0
    recovered_matches = 0

    total_queries = 0
    total_candidates = 0

    missed_matches = []

    print("\nStarting candidate validation...")

    for _, gt_row in sample.iterrows():

        source1_id = gt_row["source1_entity_id"]

        matched_ids = [
            value.strip()
            for value in gt_row["matched_entity_ids"].split(",")
            if value.strip()
        ]

        if source1_id not in source1_lookup:
            continue

        source1_record = source1_lookup[source1_id]

        for matched_id in matched_ids:

            total_true_matches += 1

            # Determine whether the matched entity belongs to Source2 or Source3.
            if matched_id in source2_lookup:
                query_record = source2_lookup[matched_id]

            elif matched_id in source3_lookup:
                query_record = source3_lookup[matched_id]

            else:
                missed_matches.append(
                    (source1_id, matched_id, "not_found_in_sources")
                )
                continue

            total_queries += 1

            candidates = generate_candidates(
                query_record=query_record,
                reference_records=source1_records,
                candidate_index=candidate_index,
                max_candidates=MAX_CANDIDATES
            )

            total_candidates += len(candidates)

            candidate_ids = {
                candidate["entity_id"]
                for candidate in candidates
            }

            if source1_id in candidate_ids:
                recovered_matches += 1
            else:
                missed_matches.append(
                    (
                        source1_id,
                        matched_id,
                        query_record.get("business_name", "")
                    )
                )

    print("\n" + "=" * 60)
    print("CANDIDATE GENERATION VALIDATION")
    print("=" * 60)

    print(f"Source1 sample records: {len(sample)}")
    print(f"True matches checked: {total_true_matches}")
    print(f"Queries evaluated: {total_queries}")
    print(f"Matches recovered: {recovered_matches}")
    print(f"Matches missed: {total_true_matches - recovered_matches}")

    if total_true_matches > 0:
        recall = recovered_matches / total_true_matches

        print(f"\nCandidate Recall: {recall:.4%}")

    if total_queries > 0:
        average_candidates = total_candidates / total_queries

        print(
            f"Average candidates per query: "
            f"{average_candidates:.2f}"
        )

    print(f"\nMissed match examples: {len(missed_matches)}")

    for example in missed_matches[:20]:
        print(example)


if __name__ == "__main__":
    main()
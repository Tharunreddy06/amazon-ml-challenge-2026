"""
Build labeled candidate pairs for the entity matching model.

For every query record:
    Source1 reference + candidate record
                         ↓
                  matching features
                         ↓
                    label 0/1

Positive:
    Candidate Source1 entity is in ground truth.

Negative:
    Candidate Source1 entity is not in ground truth.
"""

from pathlib import Path

import pandas as pd

from preprocessing import preprocess_record
from candidate_generation import (
    build_candidate_index,
    generate_candidates,
)
from matching_features import generate_matching_features


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data")

GROUND_TRUTH_FILE = DATA_DIR / "train_ground_truth.tsv"
SOURCE1_FILE = DATA_DIR / "train_source1.tsv"
SOURCE2_FILE = DATA_DIR / "train_source2.tsv"
SOURCE3_FILE = DATA_DIR / "train_source3.tsv"

SAMPLE_SIZE = 5000
CHUNK_SIZE = 200_000

OUTPUT_FILE = DATA_DIR / "training_pairs_sample.csv"


# --------------------------------------------------
# LOAD GROUND TRUTH
# --------------------------------------------------

print("Loading ground truth...")

ground_truth = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

ground_truth = ground_truth[
    ground_truth["matched_entity_ids"].str.strip() != ""
].head(SAMPLE_SIZE)

print("Ground truth records:", len(ground_truth))


# --------------------------------------------------
# COLLECT REQUIRED IDS
# --------------------------------------------------

source1_ids = set(
    ground_truth["source1_entity_id"]
)

matched_ids = set()

for value in ground_truth["matched_entity_ids"]:

    for entity_id in value.split(","):

        entity_id = entity_id.strip()

        if entity_id:
            matched_ids.add(entity_id)

print("Required Source1 IDs:", len(source1_ids))
print("Required matched IDs:", len(matched_ids))


# --------------------------------------------------
# LOAD REQUIRED RECORDS
# --------------------------------------------------

USE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def load_records_by_ids(file_path, required_ids):
    """
    Load only records whose entity_id appears in required_ids.
    """

    records = []

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=USE_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):

        filtered = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        if not filtered.empty:

            records.extend(
                filtered.to_dict("records")
            )

    return records


print("\nLoading Source1...")

source1_raw = load_records_by_ids(
    SOURCE1_FILE,
    source1_ids,
)

print("Source1 records:", len(source1_raw))


print("\nLoading Source2...")

source2_raw = load_records_by_ids(
    SOURCE2_FILE,
    matched_ids,
)

print("Source2 records:", len(source2_raw))


print("\nLoading Source3...")

source3_raw = load_records_by_ids(
    SOURCE3_FILE,
    matched_ids,
)

print("Source3 records:", len(source3_raw))


# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------

print("\nPreprocessing...")

source1_records = [
    preprocess_record(record)
    for record in source1_raw
]

source2_records = [
    preprocess_record(record)
    for record in source2_raw
]

source3_records = [
    preprocess_record(record)
    for record in source3_raw
]


# --------------------------------------------------
# BUILD CANDIDATE INDEX
# --------------------------------------------------

print("\nBuilding candidate index...")

candidate_index = build_candidate_index(
    source1_records
)


# --------------------------------------------------
# QUERY LOOKUP
# --------------------------------------------------

query_records = {}

for record in source2_records:
    query_records[record["entity_id"]] = record

for record in source3_records:
    query_records[record["entity_id"]] = record


# --------------------------------------------------
# SOURCE1 LOOKUP
# --------------------------------------------------

source1_lookup = {
    record["entity_id"]: record
    for record in source1_records
}


# --------------------------------------------------
# BUILD TRAINING PAIRS
# --------------------------------------------------

print("\nGenerating training pairs...")

training_rows = []

positive_pairs = 0
negative_pairs = 0

queries_processed = 0
queries_with_candidates = 0


for row_number, (_, row) in enumerate(
    ground_truth.iterrows(),
    start=1
):

    source1_id = row["source1_entity_id"]

    reference_record = source1_lookup.get(
        source1_id
    )

    if reference_record is None:
        continue

    true_matches = {
        entity_id.strip()
        for entity_id in
        row["matched_entity_ids"].split(",")
        if entity_id.strip()
    }

    # --------------------------------------------------
    # Process every true query record
    # --------------------------------------------------

    for matched_id in true_matches:

        query_record = query_records.get(
            matched_id
        )

        if query_record is None:
            continue

        queries_processed += 1

        # Generate Source1 candidates for this query.
        candidates = generate_candidates(
            query_record=query_record,
            reference_records=source1_records,
            candidate_index=candidate_index,
            max_candidates=200,
        )

        if candidates:
            queries_with_candidates += 1

        # --------------------------------------------------
        # IMPORTANT:
        # Compare Source1 reference with EACH CANDIDATE
        # --------------------------------------------------

        for candidate in candidates:

            candidate_id = candidate["entity_id"]

            # Correct feature calculation:
            #
            # reference Source1 record
            #            +
            # candidate Source1 record
            #
            features = generate_matching_features(
                candidate,
                query_record,
            )

            # The candidate is positive only when its
            # Source1 ID is the true Source1 entity.
            label = int(
                candidate_id == source1_id
            )

            training_row = {
                "source1_entity_id": source1_id,
                "query_entity_id": matched_id,
                "candidate_entity_id": candidate_id,
                "label": label,
            }

            training_row.update(features)

            training_rows.append(
                training_row
            )

            if label == 1:
                positive_pairs += 1
            else:
                negative_pairs += 1

    if row_number % 100 == 0:

        print(
            f"Processed {row_number}/"
            f"{len(ground_truth)} ground-truth records..."
        )


# --------------------------------------------------
# SAVE DATASET
# --------------------------------------------------

training_df = pd.DataFrame(
    training_rows
)

training_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 50)
print("TRAINING PAIR GENERATION COMPLETE")
print("=" * 50)

print(
    "Total training pairs:",
    len(training_df)
)

print(
    "Positive pairs:",
    positive_pairs
)

print(
    "Negative pairs:",
    negative_pairs
)

print(
    "Queries processed:",
    queries_processed
)

print(
    "Queries with candidates:",
    queries_with_candidates
)

if len(training_df) > 0:

    positive_percentage = (
        positive_pairs /
        len(training_df)
    ) * 100

    print(
        f"Positive percentage: "
        f"{positive_percentage:.2f}%"
    )

print(
    "\nSaved to:",
    OUTPUT_FILE
)
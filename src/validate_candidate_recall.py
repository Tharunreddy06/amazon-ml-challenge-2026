
from pathlib import Path
import pandas as pd

from preprocessing import preprocess_record
from candidate_generation import (
    build_candidate_index,
    generate_candidates,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data")

GROUND_TRUTH_FILE = DATA_DIR / "train_ground_truth.tsv"
SOURCE1_FILE = DATA_DIR / "train_source1.tsv"
SOURCE2_FILE = DATA_DIR / "train_source2.tsv"
SOURCE3_FILE = DATA_DIR / "train_source3.tsv"

SAMPLE_SIZE = 1000
CHUNK_SIZE = 200_000


# --------------------------------------------------
# LOAD GROUND TRUTH SAMPLE
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

print("Ground truth sample size:", len(ground_truth))


# --------------------------------------------------
# COLLECT REQUIRED IDS
# --------------------------------------------------

source1_ids = set(ground_truth["source1_entity_id"])

matched_ids = set()

for value in ground_truth["matched_entity_ids"]:
    for entity_id in value.split(","):
        entity_id = entity_id.strip()

        if entity_id:
            matched_ids.add(entity_id)

print("Required Source1 records:", len(source1_ids))
print("Required matched records:", len(matched_ids))


# --------------------------------------------------
# LOAD ONLY REQUIRED RECORDS
# --------------------------------------------------

USE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def load_records_by_ids(file_path, required_ids):
    """
    Read a TSV file in chunks and retain only
    records whose entity_id is required.
    """

    collected_records = []

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
            collected_records.extend(
                filtered.to_dict("records")
            )

    return collected_records


print("\nLoading required Source1 records...")

source1_records_raw = load_records_by_ids(
    SOURCE1_FILE,
    source1_ids,
)

print(
    "Source1 records loaded:",
    len(source1_records_raw)
)


print("\nLoading required Source2 records...")

source2_records_raw = load_records_by_ids(
    SOURCE2_FILE,
    matched_ids,
)

print(
    "Source2 records loaded:",
    len(source2_records_raw)
)


print("\nLoading required Source3 records...")

source3_records_raw = load_records_by_ids(
    SOURCE3_FILE,
    matched_ids,
)

print(
    "Source3 records loaded:",
    len(source3_records_raw)
)


# --------------------------------------------------
# PREPROCESS RECORDS
# --------------------------------------------------

print("\nPreprocessing records...")

source1_records = [
    preprocess_record(record)
    for record in source1_records_raw
]

source2_records = [
    preprocess_record(record)
    for record in source2_records_raw
]

source3_records = [
    preprocess_record(record)
    for record in source3_records_raw
]


# --------------------------------------------------
# BUILD CANDIDATE INDEX
# --------------------------------------------------

print("\nBuilding candidate index...")

candidate_index = build_candidate_index(
    source1_records
)


# --------------------------------------------------
# CREATE QUERY RECORD LOOKUP
# --------------------------------------------------

query_records = {}

for record in source2_records:
    query_records[record["entity_id"]] = record

for record in source3_records:
    query_records[record["entity_id"]] = record


# --------------------------------------------------
# VALIDATE CANDIDATE RECALL
# --------------------------------------------------

print("\nValidating candidate recall...")

source1_lookup = {
    record["entity_id"]: record
    for record in source1_records
}

total_matches = 0
retrieved_matches = 0
missed_matches = 0

total_queries = 0
queries_with_candidates = 0

missed_examples = []


for _, row in ground_truth.iterrows():

    source1_id = row["source1_entity_id"]

    matched_entity_ids = [
        entity_id.strip()
        for entity_id in row["matched_entity_ids"].split(",")
        if entity_id.strip()
    ]

    reference_record = source1_lookup.get(source1_id)

    if reference_record is None:
        continue

    for matched_id in matched_entity_ids:

        query_record = query_records.get(matched_id)

        if query_record is None:
            continue

        total_queries += 1
        total_matches += 1

        candidates = generate_candidates(
            query_record=query_record,
            reference_records=source1_records,
            candidate_index=candidate_index,
            max_candidates=200,
        )

        candidate_ids = {
            candidate["entity_id"]
            for candidate in candidates
        }

        if source1_id in candidate_ids:
            retrieved_matches += 1
        else:
            missed_matches += 1

            if len(missed_examples) < 10:
                missed_examples.append({
                    "source1_id": source1_id,
                    "matched_id": matched_id,
                    "query_name": query_record.get(
                        "business_name", ""
                    ),
                    "reference_name": reference_record.get(
                        "business_name", ""
                    ),
                })

        if candidates:
            queries_with_candidates += 1


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 50)
print("CANDIDATE RECALL VALIDATION RESULTS")
print("=" * 50)

print("Total evaluated matches:", total_matches)
print("Retrieved matches:", retrieved_matches)
print("Missed matches:", missed_matches)

if total_matches > 0:
    recall = (
        retrieved_matches / total_matches
    ) * 100

    print(f"Candidate recall: {recall:.2f}%")
else:
    print("Candidate recall: Cannot calculate")

if total_queries > 0:
    candidate_coverage = (
        queries_with_candidates / total_queries
    ) * 100

    print(
        f"Queries with candidates: "
        f"{candidate_coverage:.2f}%"
    )

print("\nMissed match examples:")

for example in missed_examples:
    print(example)

print("\nValidation completed.")
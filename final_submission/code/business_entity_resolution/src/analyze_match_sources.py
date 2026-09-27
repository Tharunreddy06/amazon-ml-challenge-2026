
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

GROUND_TRUTH_FILE = DATA_DIR / "train_ground_truth.tsv"

print("Loading ground truth...")

df = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

def get_source_counts(value):
    value = value.strip()

    if not value:
        return 0, 0

    ids = value.split(",")

    source2_count = sum(
        entity_id.startswith("S2-")
        for entity_id in ids
    )

    source3_count = sum(
        entity_id.startswith("S3-")
        for entity_id in ids
    )

    return source2_count, source3_count


counts = df["matched_entity_ids"].map(get_source_counts)

df["source2_count"] = counts.map(lambda x: x[0])
df["source3_count"] = counts.map(lambda x: x[1])

print("\nSource 2 match statistics:")
print(df["source2_count"].describe())

print("\nSource 3 match statistics:")
print(df["source3_count"].describe())

print("\nTotal Source 2 matches:")
print(df["source2_count"].sum())

print("\nTotal Source 3 matches:")
print(df["source3_count"].sum())

print("\nRecords containing Source 2 matches:")
print((df["source2_count"] > 0).sum())

print("\nRecords containing Source 3 matches:")
print((df["source3_count"] > 0).sum())

print("\nRecords containing both Source 2 and Source 3 matches:")
print(
    (
        (df["source2_count"] > 0) &
        (df["source3_count"] > 0)
    ).sum()
)

print("\nRecords containing only Source 2 matches:")
print(
    (
        (df["source2_count"] > 0) &
        (df["source3_count"] == 0)
    ).sum()
)

print("\nRecords containing only Source 3 matches:")
print(
    (
        (df["source2_count"] == 0) &
        (df["source3_count"] > 0)
    ).sum()
)

print("\nRecords with no matches:")
print(
    (
        (df["source2_count"] == 0) &
        (df["source3_count"] == 0)
    ).sum()
)
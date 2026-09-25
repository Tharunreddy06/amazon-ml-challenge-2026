
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

print("\nDataset shape:", df.shape)
print("Columns:", list(df.columns))

# Empty matched_entity_ids means no known match
df["has_match"] = df["matched_entity_ids"].str.strip() != ""

# Count the number of matched entities per Source 1 record
def count_matches(value):
    value = value.strip()

    if not value:
        return 0

    return len(value.split(","))


df["match_count"] = df["matched_entity_ids"].map(count_matches)

print("\nRecords with at least one match:")
print(df["has_match"].value_counts())

print("\nMatch count distribution:")
print(df["match_count"].value_counts().sort_index().head(30))

print("\nBasic statistics:")
print(df["match_count"].describe())

print("\nRecords with zero matches:")
print((df["match_count"] == 0).sum())

print("\nRecords with one match:")
print((df["match_count"] == 1).sum())

print("\nRecords with more than one match:")
print((df["match_count"] > 1).sum())

print("\nTop match counts:")
print(df["match_count"].value_counts().sort_index(ascending=False).head(20))
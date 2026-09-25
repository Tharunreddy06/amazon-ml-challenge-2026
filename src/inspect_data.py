
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
]

for filename in files:
    path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")

    if not path.exists():
        print("STATUS: File not found")
        continue

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Shape:", df.shape)

    print("\nEmpty values:")
    print((df == "").sum())

    if "entity_id" in df.columns:
        print(
            "Duplicate entity IDs:",
            df["entity_id"].duplicated().sum()
        )

    if "country" in df.columns:
        print("\nCountry distribution:")
        print(df["country"].value_counts())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
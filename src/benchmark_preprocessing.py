
import sys
import time
from pathlib import Path

import pandas as pd

# Allow imports from the src directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import preprocess_record


def main():
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "train_source1.tsv"

    print(f"Loading dataset: {dataset_path}")

    df = pd.read_csv(
        dataset_path,
        sep="\t",
        nrows=10_000,
        dtype=str,
        keep_default_na=False,
    )

    print(f"Records loaded: {len(df):,}")

    records = df.to_dict(orient="records")

    start_time = time.perf_counter()

    processed_records = [
        preprocess_record(record)
        for record in records
    ]

    elapsed_time = time.perf_counter() - start_time

    print(f"Records processed: {len(processed_records):,}")
    print(f"Time taken: {elapsed_time:.4f} seconds")

    if elapsed_time > 0:
        records_per_second = len(records) / elapsed_time
        print(f"Processing speed: {records_per_second:,.2f} records/second")

    print("\nSample processed record:")
    print(processed_records[0])


if __name__ == "__main__":
    main()
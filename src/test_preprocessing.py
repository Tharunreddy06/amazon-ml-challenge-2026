
from pathlib import Path

import pandas as pd

from preprocessing import preprocess_record


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "train_source1.tsv"

    # Load only 10 rows for testing.
    df = pd.read_csv(
        data_path,
        sep="\t",
        nrows=10,
        dtype=str,
        keep_default_na=False,
    )

    print("Original dataset:")
    print(df.head(3).to_string(index=False))

    print("\nProcessed records:")
    print("=" * 60)

    for _, row in df.iterrows():
        processed = preprocess_record(row.to_dict())

        print("Entity ID:", processed["entity_id"])
        print("Original name:", processed["business_name"])
        print("Normalized name:", processed["normalized_name"])
        print("Original address:", processed["business_address"])
        print("Normalized address:", processed["normalized_address"])
        print("Country:", processed["normalized_country"])
        print("-" * 60)


if __name__ == "__main__":
    main()
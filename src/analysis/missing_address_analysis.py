import pandas as pd

for file in [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv"
]:

    df = pd.read_csv(
        f"ML_Challange_datasets/{file}",
        sep="\t"
    )

    missing = df["business_address"].isna().sum()

    print(f"\n{file}")
    print(f"Rows: {len(df)}")
    print(f"Missing addresses: {missing}")
    print(f"Percentage: {100*missing/len(df):.2f}%")
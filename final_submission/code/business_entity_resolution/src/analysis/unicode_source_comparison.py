import pandas as pd

def count_unicode(file_path, source_name):
    df = pd.read_csv(
        file_path,
        sep="\t",
        usecols=["business_name"],
        dtype=str,
        nrows=100000
    )

    count = 0

    for name in df["business_name"].fillna(""):
        try:
            name.encode("ascii")
        except UnicodeEncodeError:
            count += 1

    print(f"\n{source_name}")
    print("-" * 40)
    print(f"Records Checked: {len(df):,}")
    print(f"Unicode Names: {count:,}")
    print(f"Percentage: {100*count/len(df):.2f}%")

count_unicode(
    "ML_Challange_datasets/train_source1.tsv",
    "SOURCE 1"
)

count_unicode(
    "ML_Challange_datasets/train_source2.tsv",
    "SOURCE 2"
)

count_unicode(
    "ML_Chllange_datasets/train_source3.tsv",
    "SOURCE 3"
)
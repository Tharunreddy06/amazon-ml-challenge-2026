import pandas as pd

print("Loading sample data...")

s1 = pd.read_csv(
    "ML_Challange_datasets/train_source1.tsv",
    sep="\t",
    usecols=["business_name"],
    dtype=str,
    nrows=100000
)

total = len(s1)
unicode_count = 0

for name in s1["business_name"].fillna(""):

    try:
        name.encode("ascii")
    except UnicodeEncodeError:
        unicode_count += 1

print("\nUNICODE ANALYSIS")
print("=" * 50)

print(f"Records Checked: {total:,}")
print(f"Unicode Names: {unicode_count:,}")
print(
    f"Percentage: {(unicode_count/total)*100:.2f}%"
)
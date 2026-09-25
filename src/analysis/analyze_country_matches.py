import pandas as pd

print("Loading train_source1...")

df = pd.read_csv(
    "ML_Challange_datasets/train_source1.tsv",
    sep="\t"
)

print("\nCountry Distribution")
print(df["country"].value_counts())

print("\nPercentages")
print(
    (df["country"].value_counts(normalize=True) * 100)
    .round(2)
)
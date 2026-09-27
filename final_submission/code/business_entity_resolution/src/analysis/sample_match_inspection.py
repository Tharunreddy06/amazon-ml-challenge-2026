import pandas as pd

print("Loading data...")

s1 = pd.read_csv(
    "ML_Challange_datasets/train_source1.tsv",
    sep="\t",
    nrows=10
)

gt = pd.read_csv(
    "ML_Challange_datasets/train_ground_truth.tsv",
    sep="\t",
    nrows=10
)

print("\n=== SOURCE 1 SAMPLE ===")
print(s1[["entity_id", "business_name", "business_address"]])

print("\n=== GROUND TRUTH SAMPLE ===")
print(gt)
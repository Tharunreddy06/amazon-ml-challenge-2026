import pandas as pd

print("Loading Ground Truth...")

gt = pd.read_csv(
    "ML_Challange_datasets/train_ground_truth.tsv",
    sep="\t"
)

print("Rows:", len(gt))

print("\nGround Truth Sample:")
print(gt.head())

print("\nRecords With Matches:")

matches = gt["matched_entity_ids"].notna().sum()

print(matches)

print("\nRecords Without Matches:")

print(len(gt) - matches)
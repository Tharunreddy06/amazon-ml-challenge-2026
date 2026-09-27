import pandas as pd

gt = pd.read_csv(
    "ML_Challange_datasets/train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

match_counts = gt["matched_entity_ids"].fillna("").apply(
    lambda x: len(str(x).split(",")) if x.strip() else 0
)

max_matches = match_counts.max()

print(f"Maximum Matches = {max_matches}")

rows = gt[match_counts == max_matches]

print("\nExamples:\n")

for _, row in rows.head(5).iterrows():
    print("Source1:", row["source1_entity_id"])
    print("Matches:", row["matched_entity_ids"])
    print("-" * 80)
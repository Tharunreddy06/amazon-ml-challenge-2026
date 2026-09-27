import pandas as pd

gt = pd.read_csv(
    "ML_Challange_datasets/train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

s2 = pd.read_csv(
    "ML_Challange_datasets/train_source2.tsv",
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    "ML_Challange_datasets/train_source3.tsv",
    sep="\t",
    dtype=str
)

target = "S1-765235386"

row = gt[gt["source1_entity_id"] == target].iloc[0]

matches = row["matched_entity_ids"].split(",")

print("TOTAL MATCHES:", len(matches))
print("=" * 80)

for match_id in matches:

    if match_id.startswith("S2"):
        rec = s2[s2["entity_id"] == match_id]

    else:
        rec = s3[s3["entity_id"] == match_id]

    if not rec.empty:
        r = rec.iloc[0]

        print("\nID:", r["entity_id"])
        print("NAME:", r["business_name"])
        print("ADDRESS:", r["business_address"])
        print("-" * 80)
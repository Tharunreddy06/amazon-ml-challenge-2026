import pandas as pd

print("Loading datasets...")

gt = pd.read_csv(
    "ML_Challange_datasets/train_ground_truth.tsv",
    sep="\t"
)

s1 = pd.read_csv(
    "ML_Challange_datasets/train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "country"]
)

s2 = pd.read_csv(
    "ML_Challange_datasets/train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "country"]
)

s3 = pd.read_csv(
    "ML_Challange_datasets/train_source3.tsv",
    sep="\t",
    usecols=["entity_id", "country"]
)

print("\nChecking first 1000 matched records...\n")

same_country = 0
different_country = 0

sample_gt = gt.head(1000)

for _, row in sample_gt.iterrows():

    s1_id = row["source1_entity_id"]

    s1_country_row = s1[s1["entity_id"] == s1_id]

    if len(s1_country_row) == 0:
        continue

    s1_country = s1_country_row.iloc[0]["country"]

    matches = str(row["matched_entity_ids"]).split(",")

    for match_id in matches:

        if match_id.startswith("S2"):
            match_row = s2[s2["entity_id"] == match_id]

        elif match_id.startswith("S3"):
            match_row = s3[s3["entity_id"] == match_id]

        else:
            continue

        if len(match_row) == 0:
            continue

        match_country = match_row.iloc[0]["country"]

        if s1_country == match_country:
            same_country += 1
        else:
            different_country += 1

print("Same Country Matches:", same_country)
print("Different Country Matches:", different_country)
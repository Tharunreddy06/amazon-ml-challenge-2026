import pandas as pd

print("Loading datasets...")

s1 = pd.read_csv(
    "ML_Challange_datasets/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "ML_Challange_datasets/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "ML_Challange_datasets/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    "ML_Challange_datasets/train_ground_truth.tsv",
    sep="\t"
)

samples = gt.sample(10, random_state=42)

for idx, sample in samples.iterrows():

    print("\n" + "="*80)

    s1_id = sample["source1_entity_id"]

    source1_row = s1[s1["entity_id"] == s1_id]

    print("\nSOURCE 1:")
    print(
        source1_row[
            ["entity_id", "business_name", "business_address"]
        ]
    )

    matched_ids = str(sample["matched_entity_ids"]).split(",")

    print("\nMATCHES:")

    for mid in matched_ids:

        if mid.startswith("S2"):
            row = s2[s2["entity_id"] == mid]

        elif mid.startswith("S3"):
            row = s3[s3["entity_id"] == mid]

        else:
            continue

        print("\n------------------")
        print(
            row[
                ["entity_id", "business_name", "business_address"]
            ]
        )
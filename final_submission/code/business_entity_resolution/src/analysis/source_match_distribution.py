import pandas as pd

print("Loading ground truth...")

gt = pd.read_csv(
    "ML_Challange_datasets/train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

total_s2 = 0
total_s3 = 0

for matches in gt["matched_entity_ids"].fillna(""):

    if not matches.strip():
        continue

    ids = matches.split(",")

    s2_count = sum(
        1 for x in ids
        if x.startswith("S2-")
    )

    s3_count = sum(
        1 for x in ids
        if x.startswith("S3-")
    )

    total_s2 += s2_count
    total_s3 += s3_count

print("\nSOURCE MATCH ANALYSIS")
print("=" * 50)

print(f"Total S2 Matches: {total_s2:,}")
print(f"Total S3 Matches: {total_s3:,}")

print(
    f"\nAverage S2 Matches Per Source1: "
    f"{total_s2 / len(gt):.2f}"
)

print(
    f"Average S3 Matches Per Source1: "
    f"{total_s3 / len(gt):.2f}"
)

print(
    f"\nPercentage S2: "
    f"{100 * total_s2 / (total_s2 + total_s3):.2f}%"
)

print(
    f"Percentage S3: "
    f"{100 * total_s3 / (total_s2 + total_s3):.2f}%"
)
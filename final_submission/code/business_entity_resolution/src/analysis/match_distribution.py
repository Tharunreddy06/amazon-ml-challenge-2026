import pandas as pd

print("Loading ground truth...")

ground_truth = pd.read_csv(
    "ML_Challange_datasets/train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

print(f"Total Records: {len(ground_truth):,}")

# Count matches per source1 record
match_counts = ground_truth["matched_entity_ids"].fillna("").apply(
    lambda x: len(str(x).split(",")) if x.strip() else 0
)

print("\nMATCH DISTRIBUTION")
print("=" * 50)

print(f"Average Matches Per Record: {match_counts.mean():.2f}")
print(f"Maximum Matches: {match_counts.max()}")
print(f"Minimum Matches: {match_counts.min()}")

one_match = (match_counts == 1).sum()
two_matches = (match_counts == 2).sum()
three_matches = (match_counts == 3).sum()
four_plus = (match_counts >= 4).sum()

print(f"\n1 Match     : {one_match:,}")
print(f"2 Matches   : {two_matches:,}")
print(f"3 Matches   : {three_matches:,}")
print(f"4+ Matches  : {four_plus:,}")

print("\nPERCENTAGES")
print("=" * 50)

total = len(match_counts)

print(f"1 Match     : {100 * one_match / total:.2f}%")
print(f"2 Matches   : {100 * two_matches / total:.2f}%")
print(f"3 Matches   : {100 * three_matches / total:.2f}%")
print(f"4+ Matches  : {100 * four_plus / total:.2f}%")

print("\nTOP 10 RECORDS WITH MOST MATCHES")
print("=" * 50)

top_matches = match_counts.sort_values(
    ascending=False
).head(10)

for idx, count in top_matches.items():
    source_id = ground_truth.loc[idx, "source1_entity_id"]
    print(f"{source_id}: {count} matches")
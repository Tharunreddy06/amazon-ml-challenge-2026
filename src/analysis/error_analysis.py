import pandas as pd

print("ERROR ANALYSIS MODULE")
print("=" * 50)

# Example prediction data
predictions = pd.DataFrame({
    "actual": [1, 1, 1, 0, 0, 0],
    "predicted": [1, 0, 1, 1, 0, 0]
})

# False Positives
fp = predictions[
    (predictions["actual"] == 0) &
    (predictions["predicted"] == 1)
]

# False Negatives
fn = predictions[
    (predictions["actual"] == 1) &
    (predictions["predicted"] == 0)
]

print(f"False Positives: {len(fp)}")
print(f"False Negatives: {len(fn)}")
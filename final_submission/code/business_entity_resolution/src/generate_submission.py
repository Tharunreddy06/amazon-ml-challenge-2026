import pandas as pd
from pathlib import Path

INPUT_FILE = Path("predictions.csv")
OUTPUT_FILE = Path("submission.csv")

print("Loading predictions...")

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))

df = df.sort_values(
    "probability",
    ascending=False
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Submission saved: {OUTPUT_FILE}"
)
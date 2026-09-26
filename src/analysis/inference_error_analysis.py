from pathlib import Path

file_path = Path("inference_results.txt")

high_confidence = 0
medium_confidence = 0
low_confidence = 0

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

    for line in f:

        if "probability=" not in line:
            continue

        try:
            prob = float(
                line.split("probability=")[1]
                .split("|")[0]
            )

            if prob >= 0.90:
                high_confidence += 1

            elif prob >= 0.50:
                medium_confidence += 1

            else:
                low_confidence += 1

        except:
            pass

print("\nINFERENCE CONFIDENCE ANALYSIS")
print("=" * 50)
print("High Confidence :", high_confidence)
print("Medium Confidence :", medium_confidence)
print("Low Confidence :", low_confidence)
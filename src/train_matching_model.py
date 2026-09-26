"""
Train the first baseline matching model.

Model:
    Random Forest Classifier

Input:
    data/training_pairs_sample.csv

Output:
    models/matching_model.pkl
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_FILE = Path("data/training_pairs_sample.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "matching_model.pkl"

RANDOM_STATE = 42


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading training pairs...")

df = pd.read_csv(DATA_FILE)

print("Total rows:", len(df))
print("Columns:", list(df.columns))


# --------------------------------------------------
# FEATURE COLUMNS
# --------------------------------------------------

FEATURE_COLUMNS = [
    # Existing features
    "exact_name_match",
    "exact_address_match",
    "exact_country_match",
    "name_similarity",
    "address_similarity",
    "name_token_similarity",
    "address_token_similarity",

    # New features
    "name_containment",
    "address_containment",
    "name_token_overlap_count",
    "address_token_overlap_count",
    "address_number_match",
    "name_first_token_match",
    "name_last_token_match",

    # Missing-value features
    "name_a_missing",
    "name_b_missing",
    "address_a_missing",
    "address_b_missing",
    "country_a_missing",
    "country_b_missing",

    # Length features
    "name_length_difference",
    "address_length_difference",
]


# --------------------------------------------------
# CHECK FEATURES
# --------------------------------------------------

missing_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in df.columns
]

if missing_features:

    raise ValueError(
        f"Missing feature columns: "
        f"{missing_features}"
    )


# --------------------------------------------------
# PREPARE X AND y
# --------------------------------------------------

X = df[FEATURE_COLUMNS].copy()
y = df["label"].astype(int)


# --------------------------------------------------
# TRAIN / VALIDATION SPLIT
# --------------------------------------------------

print("\nCreating train/validation split...")

unique_source1_ids = (
    df["source1_entity_id"]
    .drop_duplicates()
    .astype(str)
    .tolist()
)

train_ids, validation_ids = train_test_split(
    unique_source1_ids,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_mask = df[
    "source1_entity_id"
].isin(train_ids)

validation_mask = df[
    "source1_entity_id"
].isin(validation_ids)


X_train = X[train_mask]
y_train = y[train_mask]

X_validation = X[validation_mask]
y_validation = y[validation_mask]


print("Training pairs:", len(X_train))
print("Validation pairs:", len(X_validation))

print(
    "Training positives:",
    int(y_train.sum())
)

print(
    "Validation positives:",
    int(y_validation.sum())
)


# --------------------------------------------------
# TRAIN RANDOM FOREST
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)

print("Training completed.")


# --------------------------------------------------
# PREDICT PROBABILITIES
# --------------------------------------------------

print("\nGenerating validation probabilities...")

probabilities = model.predict_proba(
    X_validation
)[:, 1]


# --------------------------------------------------
# THRESHOLD EVALUATION
# --------------------------------------------------

print("\nThreshold evaluation")
print("=" * 70)

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


best_threshold = None
best_f05 = -1


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f05 = fbeta_score(
        y_validation,
        predictions,
        beta=0.5,
        zero_division=0,
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F0.5={f05:.4f}"
    )

    if f05 > best_f05:

        best_f05 = f05
        best_threshold = threshold


# --------------------------------------------------
# BEST THRESHOLD
# --------------------------------------------------

best_predictions = (
    probabilities >= best_threshold
).astype(int)


best_precision = precision_score(
    y_validation,
    best_predictions,
    zero_division=0,
)

best_recall = recall_score(
    y_validation,
    best_predictions,
    zero_division=0,
)

best_f05 = fbeta_score(
    y_validation,
    best_predictions,
    beta=0.5,
    zero_division=0,
)

matrix = confusion_matrix(
    y_validation,
    best_predictions,
)


print("\n" + "=" * 70)
print("BEST VALIDATION RESULT")
print("=" * 70)

print(
    f"Best threshold: {best_threshold:.2f}"
)

print(
    f"Precision: {best_precision:.4f}"
)

print(
    f"Recall: {best_recall:.4f}"
)

print(
    f"F0.5: {best_f05:.4f}"
)

print("\nConfusion matrix:")
print(matrix)


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

print("\nFeature importance")
print("=" * 50)

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_,
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
)

for _, row in importance_df.iterrows():

    print(
        f"{row['feature']:<30} "
        f"{row['importance']:.4f}"
    )


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model_package = {
    "model": model,
    "feature_columns": FEATURE_COLUMNS,
    "best_threshold": best_threshold,
}

joblib.dump(
    model_package,
    MODEL_FILE,
)


print("\n" + "=" * 70)

print(
    f"Model saved to: {MODEL_FILE}"
)

print("=" * 70)
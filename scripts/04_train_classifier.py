"""
Step 4: Train a Random Forest fall classifier.

Two decisions worth stating explicitly:
1. Split by subject, not by window. Splitting by window would let the
   same person's gait signature appear in both train and test, inflating
   accuracy in a way that wouldn't hold up on a genuinely unseen person.
2. class_weight='balanced' to handle the ~56:1 ADL:fall imbalance that
   resulted from the peak-centered windowing fix (Script 2 revision) —
   the model would otherwise trivially predict "adl" for everything and
   still score ~98% accuracy while catching zero falls.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_PATH = os.path.join("data", "processed", "features.csv")
MODEL_OUTPUT = os.path.join("models", "random_forest.pkl")
METRICS_OUTPUT = os.path.join("results", "04_metrics.txt")

NON_FEATURE_COLUMNS = [
    "subject", "subject_group", "activity_code", "trial",
    "label", "window_index", "source_file",
]

RANDOM_STATE = 8
TEST_SIZE = 0.2


def main():
    df = pd.read_csv(FEATURES_PATH)
    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    excluded_orientation = [c for c in feature_columns
                         if c.startswith(("ax1", "ay1", "az1", "gx", "gy", "gz"))
                         and c.endswith(("_mean", "_max", "_min"))]
    feature_columns = [c for c in feature_columns if c not in excluded_orientation]

    X = df[feature_columns].values
    y = (df["label"] == "fall").astype(int).values
    groups = df["subject"].values

    print(f"Total windows: {len(df)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Unique subjects: {df['subject'].nunique()}")
    print(f"Class balance: {np.bincount(y)} (0=adl, 1=fall)")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_subjects = set(groups[train_idx])
    test_subjects = set(groups[test_idx])
    assert train_subjects.isdisjoint(test_subjects), "Subject leakage between train and test!"

    print(f"\nTrain: {len(X_train)} windows, {len(train_subjects)} subjects")
    print(f"Test:  {len(X_test)} windows, {len(test_subjects)} subjects")
    print(f"Train class balance: {np.bincount(y_train)}")
    print(f"Test class balance:  {np.bincount(y_test)}")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=["adl", "fall"], digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + report)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("           pred_adl  pred_fall")
    print(f"actual_adl   {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"actual_fall  {cm[1][0]:6d}    {cm[1][1]:6d}")

    # Feature importance, top 10
    importances = pd.Series(clf.feature_importances_, index=feature_columns)
    top_features = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 features by importance:")
    print(top_features.to_string())

    joblib.dump(clf, MODEL_OUTPUT)
    with open(METRICS_OUTPUT, "w") as f:
        f.write(report)
        f.write("\n\nConfusion matrix (rows=actual, cols=predicted):\n")
        f.write(f"           pred_adl  pred_fall\n")
        f.write(f"actual_adl   {cm[0][0]:6d}    {cm[0][1]:6d}\n")
        f.write(f"actual_fall  {cm[1][0]:6d}    {cm[1][1]:6d}\n")
        f.write("\nTop 10 features by importance:\n")
        f.write(top_features.to_string())

    print(f"\nModel saved to {MODEL_OUTPUT}")
    print(f"Metrics saved to {METRICS_OUTPUT}")


if __name__ == "__main__":
    main()
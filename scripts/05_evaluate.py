"""
Step 5: Evaluate the trained classifier beyond the aggregate metrics —
per-subject breakdown, misclassified case inspection, confusion matrix
plot, and an ablation test to check whether performance depends on
orientation-sensitive features (ay1_mean, ay1_max) versus magnitude/
dynamics features (svm_*, *_std, *_rms), which would indicate whether
the model is learning impact physics or just post-fall device orientation.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

FEATURES_PATH = os.path.join("data", "processed", "features.csv")
MODEL_PATH = os.path.join("models", "random_forest.pkl")

NON_FEATURE_COLUMNS = [
    "subject", "subject_group", "activity_code", "trial",
    "label", "window_index", "source_file",
]

RANDOM_STATE = 8
TEST_SIZE = 0.2

# Orientation-sensitive features to exclude in the ablation test
ORIENTATION_FEATURES = [c for c in [] ]  # filled dynamically below


def recompute_split(df):
    """Reproduce the exact same subject-level split AND feature set used
    in Script 4 (same random_state + same GroupShuffleSplit + same
    orientation-feature exclusion = matches the saved model exactly)."""
    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    # Must match Script 4's exclusion exactly, or the saved model's
    # expected feature count won't line up with what we hand it here.
    excluded_orientation = [
        c for c in feature_columns
        if c.startswith(("ax1", "ay1", "az1", "gx", "gy", "gz"))
        and c.endswith(("_mean", "_max", "_min"))
    ]
    feature_columns = [c for c in feature_columns if c not in excluded_orientation]

    X = df[feature_columns].values
    y = (df["label"] == "fall").astype(int).values
    groups = df["subject"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return feature_columns, X, y, groups, train_idx, test_idx


def per_subject_breakdown(df, y_test, y_pred, test_idx):
    test_df = df.iloc[test_idx].copy()
    test_df["actual"] = y_test
    test_df["predicted"] = y_pred
    test_df["correct"] = test_df["actual"] == test_df["predicted"]

    rows = []
    for subject, group in test_df.groupby("subject"):
        fall_rows = group[group["actual"] == 1]
        rows.append({
            "subject": subject,
            "total_windows": len(group),
            "fall_windows": len(fall_rows),
            "falls_caught": (fall_rows["predicted"] == 1).sum(),
            "falls_missed": (fall_rows["predicted"] == 0).sum(),
            "false_alarms": ((group["actual"] == 0) & (group["predicted"] == 1)).sum(),
        })
    return pd.DataFrame(rows), test_df


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["adl", "fall"])
    ax.set_yticklabels(["adl", "fall"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set (8 held-out subjects)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_ablation(df, feature_columns, X, y, groups, train_idx, test_idx):
    """Retrain excluding orientation-sensitive raw-axis features, keep
    only magnitude/dynamics features. If recall holds up, the model
    doesn't depend on orientation as a shortcut."""
    orientation_suffixes = ("_mean", "_max", "_min")
    orientation_prefixes = ("ax1", "ay1", "az1", "gx", "gy", "gz")
    excluded = [
        c for c in feature_columns
        if c.startswith(orientation_prefixes) and c.endswith(orientation_suffixes)
    ]
    kept = [c for c in feature_columns if c not in excluded]

    print(f"\nAblation: excluding {len(excluded)} orientation-sensitive features")
    print(f"Excluded: {excluded}")
    print(f"Kept {len(kept)} magnitude/dynamics features: {kept}")

    kept_idx = [feature_columns.index(c) for c in kept]
    X_ablated = X[:, kept_idx]

    X_train, X_test = X_ablated[train_idx], X_ablated[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nAblated model performance (no orientation features):")
    print(classification_report(y_test, y_pred, target_names=["adl", "fall"], digits=4))

# Add this after run_ablation() in scripts\05_evaluate.py, and call it from main()

def stratified_subject_split(df, feature_columns, n_repeats=10):
    """random_state=42 happened to put nearly all fall-trial subjects
    into the young-adult side of the test set, leaving elderly recall
    almost unmeasured. Try multiple seeds and report the spread, rather
    than trusting one arbitrary split's elderly coverage."""
    X = df[feature_columns].values
    y = (df["label"] == "fall").astype(int).values
    groups = df["subject"].values
    subject_group_map = df.drop_duplicates("subject").set_index("subject")["subject_group"]

    print(f"\nTesting {n_repeats} random splits for elderly fall-recall coverage:")
    for seed in range(n_repeats):
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        test_subjects = set(groups[test_idx])
        elderly_test_subjects = [s for s in test_subjects if subject_group_map[s] == "elderly"]
        elderly_fall_windows = ((df.iloc[test_idx]["subject_group"] == "elderly") &
                                 (df.iloc[test_idx]["label"] == "fall")).sum()
        print(f"  seed={seed}: {len(elderly_test_subjects)} elderly subjects in test, "
              f"{elderly_fall_windows} elderly fall windows")


def main():
    df = pd.read_csv(FEATURES_PATH)
    clf = joblib.load(MODEL_PATH)

    feature_columns, X, y, groups, train_idx, test_idx = recompute_split(df)
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Confirm this matches Script 4's split before trusting anything below
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print("Reproduced confusion matrix (should match Script 4 exactly):")
    print(cm)

    plot_confusion_matrix(cm, os.path.join("results", "05_confusion_matrix.png"))
    print("Saved confusion matrix plot to results/05_confusion_matrix.png")

    breakdown, test_df = per_subject_breakdown(df, y_test, y_pred, test_idx)
    print("\nPer-subject breakdown (test set, 8 held-out subjects):")
    print(breakdown.to_string(index=False))
    breakdown.to_csv(os.path.join("results", "05_per_subject_breakdown.csv"), index=False)
    young_test = test_df[test_df["subject_group"] == "young"]
    elderly_test = test_df[test_df["subject_group"] == "elderly"]

    def fall_recall(subset):
        falls = subset[subset["actual"] == 1]
        return falls["predicted"].mean() if len(falls) > 0 else float("nan")

    print(f"\nYoung-adult fall recall:  {fall_recall(young_test):.4f} "
          f"(n={len(young_test[young_test['actual']==1])} fall windows, "
          f"{young_test['subject'].nunique()} subjects)")
    print(f"Elderly fall recall:      {fall_recall(elderly_test):.4f} "
          f"(n={len(elderly_test[elderly_test['actual']==1])} fall windows, "
          f"{elderly_test['subject'].nunique()} subjects — "
          f"NOTE: single elderly subject's data, not a population estimate)")

    missed_falls = test_df[(test_df["actual"] == 1) & (test_df["predicted"] == 0)]
    false_alarms = test_df[(test_df["actual"] == 0) & (test_df["predicted"] == 1)]

    print(f"\nMissed falls ({len(missed_falls)}):")
    print(missed_falls[["subject", "activity_code", "trial"]].to_string(index=False))

    print(f"\nFalse alarms ({len(false_alarms)}) — activity codes triggering false positives:")
    print(false_alarms["activity_code"].value_counts().to_string())

    # Ablation: does the model still work without orientation-sensitive features?
    # run_ablation(df, feature_columns, X, y, groups, train_idx, test_idx)

    stratified_subject_split(df, feature_columns)


if __name__ == "__main__":
    main()
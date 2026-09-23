"""
Step 1: Explore SisFall raw data.
Builds a manifest of every trial file and plots one ADL vs one Fall
signal so we can visually confirm the data before feature extraction.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = os.path.join("data", "raw")
COLUMNS = ["ax1", "ay1", "az1", "gx", "gy", "gz", "ax2", "ay2", "az2"]

FILENAME_PATTERN = re.compile(r"^([DF]\d{2})_(S[AE]\d{2})_(R\d{2})\.txt$", re.IGNORECASE)


def build_manifest(raw_dir):
    """Walk data/raw and build a DataFrame of every trial file found."""
    records = []
    for root, _, files in os.walk(raw_dir):
        for fname in files:
            match = FILENAME_PATTERN.match(fname)
            if not match:
                continue
            activity, subject, trial = match.groups()
            label = "fall" if activity.upper().startswith("F") else "adl"
            records.append({
                "filepath": os.path.join(root, fname),
                "activity_code": activity.upper(),
                "subject": subject.upper(),
                "trial": trial.upper(),
                "label": label,
                "subject_group": "elderly" if subject.upper().startswith("SE") else "young",
            })
    return pd.DataFrame(records)


def load_trial(filepath):
    """Load one raw trial file into a DataFrame with named columns."""
    df = pd.read_csv(filepath, header=None, names=COLUMNS, sep=r"\s*,\s*", engine="python")
    # SisFall's raw format ends every line with a trailing ';' after the
    # last column — strip it before numeric conversion or every row
    # gets silently dropped as NaN.
    df["az2"] = df["az2"].astype(str).str.rstrip(";")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df.reset_index(drop=True)


def main():
    manifest = build_manifest(RAW_DIR)

    if manifest.empty:
        raise SystemExit(
            f"No SisFall trial files found under {RAW_DIR}. "
            "Check that the extracted Kaggle zip landed directly in data/raw/ "
            "(not nested inside an extra subfolder)."
        )

    print(f"Total trials found: {len(manifest)}")
    print(f"Unique subjects: {manifest['subject'].nunique()}")
    print(f"Subject groups:\n{manifest['subject_group'].value_counts()}")
    print(f"Label counts:\n{manifest['label'].value_counts()}")
    print(f"Unique activity codes: {sorted(manifest['activity_code'].unique())}")

    manifest.to_csv(os.path.join("data", "processed", "manifest.csv"), index=False)
    print("\nSaved manifest to data/processed/manifest.csv")

    # Plot one ADL trial and one Fall trial side by side for a sanity check
    sample_adl = manifest[manifest["label"] == "adl"].iloc[0]
    sample_fall = manifest[manifest["label"] == "fall"].iloc[0]

    adl_df = load_trial(sample_adl["filepath"])
    fall_df = load_trial(sample_fall["filepath"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].plot(adl_df["ax1"], label="ax1")
    axes[0].plot(adl_df["ay1"], label="ay1")
    axes[0].plot(adl_df["az1"], label="az1")
    axes[0].set_title(f"ADL sample — {sample_adl['activity_code']} / {sample_adl['subject']}")
    axes[0].legend()

    axes[1].plot(fall_df["ax1"], label="ax1")
    axes[1].plot(fall_df["ay1"], label="ay1")
    axes[1].plot(fall_df["az1"], label="az1")
    axes[1].set_title(f"Fall sample — {sample_fall['activity_code']} / {sample_fall['subject']}")
    axes[1].legend()

    plt.tight_layout()
    output_path = os.path.join("results", "01_sample_signals.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved comparison plot to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
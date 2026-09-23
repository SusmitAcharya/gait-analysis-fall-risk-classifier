"""
Step 2 (revised): Convert raw signals to physical units, then segment.

Revision rationale: naive per-trial labeling (every window in a fall
trial labeled "fall") produced windows where most "fall" samples were
actually the quiet post-impact lying-still segment, diluting the class
to the point of being statistically indistinguishable from ADL on
svm_max/svm_std (verified empirically before this fix). Fall trials
are now windowed around the peak-impact moment only; the pre/post
segments of a fall trial are discarded rather than mislabeled.
"""

import os
import pandas as pd
import numpy as np

MANIFEST_PATH = os.path.join("data", "processed", "manifest.csv")
OUTPUT_WINDOWS = os.path.join("data", "processed", "windows.npz")
OUTPUT_METADATA = os.path.join("data", "processed", "windows_metadata.csv")

COLUMNS = ["ax1", "ay1", "az1", "gx", "gy", "gz", "ax2", "ay2", "az2"]
CHANNELS_USED = ["ax1", "ay1", "az1", "gx", "gy", "gz"]
ACCEL_CHANNELS = ["ax1", "ay1", "az1"]

ACCEL1_RANGE_G = 16
ACCEL1_RESOLUTION_BITS = 13
GYRO_RANGE_DPS = 2000
GYRO_RESOLUTION_BITS = 16

WINDOW_SIZE = 200   # 1 second at 200 Hz
STRIDE = 100        # 50% overlap, used for ADL trials only


def load_trial(filepath):
    df = pd.read_csv(filepath, header=None, names=COLUMNS, sep=r"\s*,\s*", engine="python")
    df["az2"] = df["az2"].astype(str).str.rstrip(";")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df.reset_index(drop=True)


def convert_units(df):
    accel_scale = (2 * ACCEL1_RANGE_G) / (2 ** ACCEL1_RESOLUTION_BITS)
    gyro_scale = (2 * GYRO_RANGE_DPS) / (2 ** GYRO_RESOLUTION_BITS)
    out = df.copy()
    for col in ["ax1", "ay1", "az1"]:
        out[col] = out[col] * accel_scale
    for col in ["gx", "gy", "gz"]:
        out[col] = out[col] * gyro_scale
    return out


def sliding_windows(data, window_size=WINDOW_SIZE, stride=STRIDE):
    windows = []
    for start in range(0, data.shape[0] - window_size + 1, stride):
        windows.append(data[start:start + window_size])
    return windows


def peak_centered_window(data, window_size=WINDOW_SIZE):
    """Locate the sample with peak signal vector magnitude and extract
    one window centered on it, clipped to trial bounds."""
    accel_indices = [CHANNELS_USED.index(c) for c in ACCEL_CHANNELS]
    svm = np.sqrt(np.sum(data[:, accel_indices] ** 2, axis=1))
    peak_idx = np.argmax(svm)

    half = window_size // 2
    start = max(0, peak_idx - half)
    end = start + window_size
    if end > data.shape[0]:
        end = data.shape[0]
        start = max(0, end - window_size)

    window = data[start:end]
    if window.shape[0] < window_size:
        return None  # trial too short to extract a full window, skip it
    return window


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Processing {len(manifest)} trials...")

    all_windows = []
    metadata_rows = []
    skipped_short_falls = 0

    for i, row in manifest.iterrows():
        try:
            raw_df = load_trial(row["filepath"])
            physical_df = convert_units(raw_df)
            data = physical_df[CHANNELS_USED].values
        except Exception as e:
            print(f"  Skipped {row['filepath']}: {e}")
            continue

        if row["label"] == "fall":
            window = peak_centered_window(data)
            if window is None:
                skipped_short_falls += 1
                continue
            all_windows.append(window)
            metadata_rows.append({
                "subject": row["subject"], "subject_group": row["subject_group"],
                "activity_code": row["activity_code"], "trial": row["trial"],
                "label": "fall", "window_index": 0, "source_file": row["filepath"],
            })
        else:
            windows = sliding_windows(data)
            for w_idx, window in enumerate(windows):
                all_windows.append(window)
                metadata_rows.append({
                    "subject": row["subject"], "subject_group": row["subject_group"],
                    "activity_code": row["activity_code"], "trial": row["trial"],
                    "label": "adl", "window_index": w_idx, "source_file": row["filepath"],
                })

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(manifest)} trials processed, {len(all_windows)} windows so far")

    windows_array = np.stack(all_windows).astype(np.float32)
    metadata_df = pd.DataFrame(metadata_rows)

    print(f"\nTotal windows: {windows_array.shape[0]}")
    print(f"Skipped fall trials (too short for a full window): {skipped_short_falls}")
    print(f"Label distribution:\n{metadata_df['label'].value_counts()}")

    np.savez_compressed(OUTPUT_WINDOWS, windows=windows_array)
    metadata_df.to_csv(OUTPUT_METADATA, index=False)
    print(f"\nSaved windows to {OUTPUT_WINDOWS}")
    print(f"Saved metadata to {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
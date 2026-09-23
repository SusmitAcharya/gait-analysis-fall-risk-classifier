"""
Step 3: Extract per-window statistical and frequency-domain features
from the windowed accelerometer/gyro data.

Feature set follows standard practice from SisFall benchmark literature:
time-domain (mean, std, min, max, RMS, SMA) + frequency-domain (dominant
frequency, spectral energy) per channel, plus a cross-channel signal
vector magnitude (SVM) — the single most discriminative fall-detection
feature in the published literature.
"""

import os
import numpy as np
import pandas as pd

WINDOWS_PATH = os.path.join("data", "processed", "windows.npz")
METADATA_PATH = os.path.join("data", "processed", "windows_metadata.csv")
OUTPUT_PATH = os.path.join("data", "processed", "features.csv")

CHANNELS = ["ax1", "ay1", "az1", "gx", "gy", "gz"]
ACCEL_CHANNELS = ["ax1", "ay1", "az1"]
SAMPLE_RATE_HZ = 200


def extract_time_domain_features(window, channel_names):
    """Per-channel statistical features for one window."""
    features = {}
    for i, name in enumerate(channel_names):
        signal = window[:, i]
        features[f"{name}_mean"] = np.mean(signal)
        features[f"{name}_std"] = np.std(signal)
        features[f"{name}_min"] = np.min(signal)
        features[f"{name}_max"] = np.max(signal)
        features[f"{name}_range"] = np.max(signal) - np.min(signal)
        features[f"{name}_rms"] = np.sqrt(np.mean(signal ** 2))
        features[f"{name}_sma"] = np.sum(np.abs(signal)) / len(signal)  # signal magnitude area
    return features


def extract_frequency_domain_features(window, channel_names, sample_rate=SAMPLE_RATE_HZ):
    """Per-channel dominant frequency and spectral energy via FFT."""
    features = {}
    n = window.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    for i, name in enumerate(channel_names):
        signal = window[:, i]
        fft_vals = np.abs(np.fft.rfft(signal))
        # Skip the DC component (index 0) when finding dominant frequency
        dominant_idx = np.argmax(fft_vals[1:]) + 1 if len(fft_vals) > 1 else 0
        features[f"{name}_dominant_freq"] = freqs[dominant_idx]
        features[f"{name}_spectral_energy"] = np.sum(fft_vals ** 2) / n
    return features


def extract_svm_features(window, accel_indices):
    """Signal Vector Magnitude: sqrt(ax^2 + ay^2 + az^2) — the standard
    single most discriminative feature for impact detection."""
    accel = window[:, accel_indices]
    svm = np.sqrt(np.sum(accel ** 2, axis=1))
    return {
        "svm_mean": np.mean(svm),
        "svm_std": np.std(svm),
        "svm_max": np.max(svm),
        "svm_min": np.min(svm),
    }


def extract_all_features(window):
    features = {}
    features.update(extract_time_domain_features(window, CHANNELS))
    features.update(extract_frequency_domain_features(window, CHANNELS))
    accel_indices = [CHANNELS.index(c) for c in ACCEL_CHANNELS]
    features.update(extract_svm_features(window, accel_indices))
    return features


def main():
    print("Loading windows and metadata...")
    windows = np.load(WINDOWS_PATH)["windows"]
    metadata = pd.read_csv(METADATA_PATH)

    assert windows.shape[0] == len(metadata), (
        f"Mismatch: {windows.shape[0]} windows but {len(metadata)} metadata rows. "
        "Re-run Script 2 before continuing."
    )

    print(f"Extracting features from {windows.shape[0]} windows...")
    feature_rows = []
    for i in range(windows.shape[0]):
        feature_rows.append(extract_all_features(windows[i]))
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{windows.shape[0]} windows processed")

    features_df = pd.DataFrame(feature_rows)
    combined = pd.concat([metadata.reset_index(drop=True), features_df], axis=1)

    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nExtracted {features_df.shape[1]} features per window")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"\nSample fall vs ADL SVM means:")
    print(combined.groupby("label")["svm_mean"].describe())


if __name__ == "__main__":
    main()
"""
Step 6 (optional): Run the trained classifier on your own recorded
phone accelerometer + gyroscope data.

Known limitations, stated explicitly rather than hidden:
- Phone sample rate is not fixed at 200Hz like SisFall; it's inferred
  from the recording's own timestamps.
- Units assumed to be SI (m/s^2, rad/s) as exported by Sensor Logger;
  converted to match SisFall's g / deg-per-second scale.
- This is a single self-recorded demo, not a validated test set. It
  demonstrates the pipeline runs end-to-end on real hardware; it does
  not add statistical evidence to the model's reported accuracy.
"""

import os
import numpy as np
import pandas as pd
import joblib

PHONE_DIR = os.path.join("data", "phone")
ACCEL_FILE = os.path.join(PHONE_DIR, "TotalAcceleration.csv")
GYRO_FILE = os.path.join(PHONE_DIR, "Gyroscope.csv")
MODEL_PATH = os.path.join("models", "random_forest.pkl")
OUTPUT_PATH = os.path.join("results", "06_phone_demo_predictions.csv")

WINDOW_DURATION_SEC = 1.0
STRIDE_DURATION_SEC = 0.5

# Must exactly match the 40-feature set Script 4 trained on.
FEATURE_COLUMNS = [
    "ax1_std", "ax1_range", "ax1_rms", "ax1_sma",
    "ay1_std", "ay1_range", "ay1_rms", "ay1_sma",
    "az1_std", "az1_range", "az1_rms", "az1_sma",
    "gx_std", "gx_range", "gx_rms", "gx_sma",
    "gy_std", "gy_range", "gy_rms", "gy_sma",
    "gz_std", "gz_range", "gz_rms", "gz_sma",
    "ax1_dominant_freq", "ax1_spectral_energy",
    "ay1_dominant_freq", "ay1_spectral_energy",
    "az1_dominant_freq", "az1_spectral_energy",
    "gx_dominant_freq", "gx_spectral_energy",
    "gy_dominant_freq", "gy_spectral_energy",
    "gz_dominant_freq", "gz_spectral_energy",
    "svm_mean", "svm_std", "svm_max", "svm_min",
]

MS2_TO_G = 1.0 / 9.80665
RAD_TO_DEG = 180.0 / np.pi


def load_sensor_logger_csv(filepath):
    """Sensor Logger exports 'seconds_elapsed', 'x', 'y', 'z' columns.
    If your app exports differently-named columns, adjust this mapping."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = "seconds_elapsed" if "seconds_elapsed" in df.columns else "time"
    return df[[time_col, "x", "y", "z"]].rename(columns={time_col: "t"})


def infer_sample_rate(t):
    dt = np.diff(t)
    return 1.0 / np.median(dt)


def merge_accel_gyro(accel_df, gyro_df):
    """Align accel and gyro on the accel timestamps via nearest-time merge."""
    accel_df = accel_df.sort_values("t").rename(columns={"x": "ax1", "y": "ay1", "z": "az1"})
    gyro_df = gyro_df.sort_values("t").rename(columns={"x": "gx", "y": "gy", "z": "gz"})
    merged = pd.merge_asof(accel_df, gyro_df, on="t", direction="nearest")
    return merged.dropna()


def extract_time_domain_features(signal, name):
    return {
        f"{name}_std": np.std(signal),
        f"{name}_range": np.max(signal) - np.min(signal),
        f"{name}_rms": np.sqrt(np.mean(signal ** 2)),
        f"{name}_sma": np.sum(np.abs(signal)) / len(signal),
    }


def extract_frequency_domain_features(signal, name, sample_rate):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    fft_vals = np.abs(np.fft.rfft(signal))
    dominant_idx = np.argmax(fft_vals[1:]) + 1 if len(fft_vals) > 1 else 0
    return {
        f"{name}_dominant_freq": freqs[dominant_idx],
        f"{name}_spectral_energy": np.sum(fft_vals ** 2) / n,
    }


def extract_window_features(window, sample_rate):
    channels = {
        "ax1": window[:, 0], "ay1": window[:, 1], "az1": window[:, 2],
        "gx": window[:, 3], "gy": window[:, 4], "gz": window[:, 5],
    }
    features = {}
    for name, signal in channels.items():
        features.update(extract_time_domain_features(signal, name))
        features.update(extract_frequency_domain_features(signal, name, sample_rate))

    svm = np.sqrt(window[:, 0] ** 2 + window[:, 1] ** 2 + window[:, 2] ** 2)
    features["svm_mean"] = np.mean(svm)
    features["svm_std"] = np.std(svm)
    features["svm_max"] = np.max(svm)
    features["svm_min"] = np.min(svm)
    return features


def main():
    if not (os.path.exists(ACCEL_FILE) and os.path.exists(GYRO_FILE)):
        raise SystemExit(
            f"Expected {ACCEL_FILE} and {GYRO_FILE}. "
            "Export both Accelerometer and Gyroscope CSVs from your logging "
            "app into data\\phone\\ before running this script."
        )

    accel_df = load_sensor_logger_csv(ACCEL_FILE)
    gyro_df = load_sensor_logger_csv(GYRO_FILE)
    merged = merge_accel_gyro(accel_df, gyro_df)

    sample_rate = infer_sample_rate(merged["t"].values)
    print(f"Inferred sample rate: {sample_rate:.1f} Hz")
    if sample_rate < 20:
        print("WARNING: sample rate is unusually low — check the logging app's settings.")

    # Convert to SisFall's training units: g for accel, deg/s for gyro
    for col in ["ax1", "ay1", "az1"]:
        merged[col] = merged[col] * MS2_TO_G
    for col in ["gx", "gy", "gz"]:
        merged[col] = merged[col] * RAD_TO_DEG  # skip this line if your app already logs deg/s

    window_size = int(round(WINDOW_DURATION_SEC * sample_rate))
    stride = int(round(STRIDE_DURATION_SEC * sample_rate))
    data = merged[["ax1", "ay1", "az1", "gx", "gy", "gz"]].values

    clf = joblib.load(MODEL_PATH)

    results = []
    for start in range(0, data.shape[0] - window_size + 1, stride):
        window = data[start:start + window_size]
        features = extract_window_features(window, sample_rate)
        feature_vector = np.array([[features[c] for c in FEATURE_COLUMNS]])
        prediction = clf.predict(feature_vector)[0]
        probability = clf.predict_proba(feature_vector)[0][1]
        results.append({
            "window_start_sec": merged["t"].values[start],
            "predicted_label": "fall" if prediction == 1 else "adl",
            "fall_probability": probability,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nProcessed {len(results_df)} windows")
    print(results_df["predicted_label"].value_counts())
    print(f"\nSaved predictions to {OUTPUT_PATH}")

    fall_windows = results_df[results_df["predicted_label"] == "fall"]
    if len(fall_windows) > 0:
        print(f"\nFall detected at these timestamps (seconds into recording):")
        print(fall_windows["window_start_sec"].to_string(index=False))
    else:
        print("\nNo fall windows detected — check whether your 'fall' motion "
              "produced a clear enough impact signature, or was too gentle "
              "to register at the trained model's sensitivity.")


if __name__ == "__main__":
    main()
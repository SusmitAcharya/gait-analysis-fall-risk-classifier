# Gait Analysis / Fall-Risk Classifier

## 1. Objective

Train a machine learning model to tell the difference between a fall and everyday movement using wearable accelerometer and gyroscope data, and then test it against a signal I record myself, on my own phone.

## 2. Dataset

**SisFall** (Sucerquia et al., 2017): 4,505 trials, 38 subjects (23 young adults, 15 elderly), 19 ADL activity types, 15 fall types, 200 Hz sampling, triaxial accelerometer + gyroscope, waist-mounted.

The original host (sistemic.udea.edu.co) was unreachable during this build (connection timeout, confirmed via browser). Sourced instead via a third-party Kaggle mirror (`nvnikhil0001/sis-fall-original-dataset`, chosen over the same uploader's "enhanced" version specifically to avoid an unknown preprocessing step invalidating comparison against published benchmarks). Verified before use: file count (4,505 — exact match), file format (raw `.txt`, comma-separated, matching the original per-trial structure), and per-subject folder layout — all consistent with the dataset as described in the original paper, giving reasonable confidence this is a faithful, unmodified mirror.

## 3. Preprocessing

**Unit conversion**: SisFall stores raw ADC counts, not physical units. Converted to g (accelerometer, ±16g range, 13-bit resolution) and deg/s (gyroscope, ±2000 deg/s range, 16-bit resolution) per the dataset's documented sensor specifications, using the raw file format's trailing `;` delimiter quirk (undocumented in the file itself, discovered when initial parsing silently dropped every row to NaN and produced empty plots — fixed by stripping the trailing character before numeric conversion).

**Windowing — the central methodological finding of this project:**

The first-pass approach used fixed 1-second sliding windows (200 samples, 50% overlap) across every trial, with fall trials' windows inheriting the trial-level "fall" label. This is the simpler, more common approach in the literature. It produced a serious, only-later-obvious problem: a SisFall fall trial runs ~15 seconds, but the actual impact lasts a fraction of a second. Most windows drawn from a "fall" trial therefore capture pre-fall standing or post-fall lying still — not the fall itself.

This was caught empirically, not assumed. Before training anything, `svm_max` and `svm_std` (signal vector magnitude features, expected to be the primary fall/ADL discriminators based on the wider fall-detection literature) were checked across the two classes:

| Feature | ADL median | Fall median (naive labeling) |
|---|---|---|
| `svm_max` | 1.216 | 1.064 |
| `svm_std` | 0.076 | 0.011 |

Both features ran **backwards** — ADL windows appeared more dynamic than fall windows, because ADL captures real movement (walking, sitting) while most "fall" windows captured stillness. Training on this would have taught the model an inverted signal.

**Fix**: fall trials switched to a single peak-centered window — locate the sample with maximum signal vector magnitude within the trial, extract one 1-second window centered on it. ADL trials kept the full sliding-window treatment (they don't have this problem; the activity is genuinely happening throughout the trial). Re-checking the same features after the fix:

| Feature | ADL median | Fall median (peak-centered) |
|---|---|---|
| `svm_max` | 1.216 | 6.197 (≈5x) |
| `svm_std` | 0.076 | 1.002 (≈13x) |

Clean, physically sensible separation. This fix cost the fall class its window count (52,019 → 1,798 — one window per fall trial instead of many), producing a much starker ~56:1 class imbalance against ADL. The tradeoff: a smaller, correctly-labeled positive class over a larger, wrong one.

## 4. Feature Extraction

40–58 features per window (see ablation, section 6), computed per channel (ax1, ay1, az1, gx, gy, gz):

- **Time-domain**: mean, std, min, max, range, RMS, signal magnitude area (SMA)
- **Frequency-domain**: dominant frequency and spectral energy via FFT
- **Cross-channel**: signal vector magnitude (SVM) — mean, std, max, min, computed from the three accelerometer axes

## 5. Model & Evaluation Methodology

**Random Forest** (300 trees, `class_weight='balanced'` to handle the 56:1 imbalance from the windowing fix), no deep learning — the class separation achieved through feature engineering made this unnecessary and kept the whole project CPU-only, consistent with this week's scope in the roadmap.

**Split methodology**: subject-level `GroupShuffleSplit` (80/20), not window-level. Window-level splitting would let the same person's gait/impact signature appear in both train and test, inflating accuracy on a way that wouldn't hold for a genuinely unseen person. Verified disjoint subject sets via assertion on every run.

## 6. Final Results (Verified Split)

#### Subject-level 80/20 split, 30 train subjects / 8 test subjects, `class_weight='balanced'`, 40-feature ablated set.


| | precision | recall | f1-score | support |
|---|---|---|---|---|
adl | 0.9989 | 0.9999 | 0.9994 | 20320 |
fall | 0.9928 | 0.9233 | 0.9568 | 300 |

#### Confusion matrix (rows=actual, cols=predicted):

| | pred_adl | pred_fall |
|---|---|---|
actual_adl | 20318 | 2 |
actual_fall | 23 | 277 |


**Top features by importance**: `svm_max`, `ax1_range`, `az1_std`, `svm_std`, `ax1_std` — magnitude and variability features dominate, consistent with the ablation finding above.

**Young-adult fall recall**: 93.3% (210/225 windows, 3 subjects: SA01, SA02, SA12)
**Elderly fall recall**: 89.3% (67/75 windows, 1 subject: SE06)

### The elderly-recall limitation

A stratified re-split test (10 different random seeds) was run specifically to check whether elderly fall coverage could be improved by re-splitting. However, in SisFall's dataset, 14 elderly subjects performed ADLs only and exactly one elderly participant (age 60) performed the complete ADL + fall protocol. This means the reported 89.3% elderly recall is an n=1 result and should not be read as a population estimate. No amount of re-splitting fixes this; the data for a broader elderly fall-recall estimate simply doesn't exist in SisFall. This also directly explains why the original SisFall paper itself reported reduced fall-detection performance in elderly validation testing.

### Error analysis

**False alarms (2 total)**: SisFall's D11 is "sitting a moment, trying to get up, and collapsing into a chair" — a deliberate, fast, high-magnitude motion that shares genuine kinematic similarity with a fall.

**Missed falls (23 total)**: cluster meaningfully around fall code **F11** across multiple subjects. F11 is a slow-onset fall — a gradual collapse rather than a sudden trip or slip — and the peak-centered windowing approach, tuned to find the sharpest impact spike in a trial, appears less reliable at isolating a gradual event.

## 7. Phone Hardware Demo

Recorded ~22 seconds of accelerometer + gyroscope data via a phone sensor logging app (Sensor Logger, Android), including normal movement and one deliberate controlled fall.

**Result**: model correctly isolated a single contiguous cluster of fall predictions (4 overlapping 1-second windows, ~9.6s–11.6s into the recording, fall probability 0.53–0.84) against a clean baseline of near-zero fall probability across the rest of the ~22-second recording. Zero false positives.

## 8. Limitations

1. Elderly fall recall is an n=1 result (structural limitation of SisFall itself, not this project's methodology).
2. Slow-onset falls (F11-type, gradual collapse) are missed more often than sudden falls — a specific, named weakness of peak-centered windowing.
3. Fall-trial windowing (one peak-centered window) and ADL-trial windowing (full sliding window) are asymmetric by necessity, given the difference between a brief event and a sustained activity — a design choice, documented rather than hidden.
4. The phone hardware validation is a single demo recording, not a statistically meaningful test set.

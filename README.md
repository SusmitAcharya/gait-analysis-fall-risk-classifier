# Gait Analysis / Fall-Risk Classifier

A subject-independent fall detector trained on the SisFall dataset, validated end-to-end on self-recorded phone accelerometer data.

## Results

| Metric | Value |
|---|---|
| Young-adult fall recall | 93.3% (210/225 windows, 3 held-out subjects) |
| Elderly fall recall | 89.3% (67/75 windows, 1 held-out subject — see limitations) |
| False positive rate | 2 windows out of 20,320 ADL windows (0.01%) |
| Published SisFall baseline (threshold method) | 96% overall |

Full methodology and result breakdown available at: [`./results/report.md`](./results/report.md).

## What this is

I built:
- A data pipeline that parses SisFall's raw sensor files and converts them into real physical units
- A feature extraction step that computes 40 statistical and frequency-based measurements per one-second window
- A Random Forest classifier trained on those features, evaluated on people it had never seen during training
- A full evaluation suite that breaks results down by individual subject, not just an overall score
- An inference script that runs the trained model on accelerometer and gyroscope data I recorded on my own phone

See [`results/06_phone_demo_predictions.csv`](./results/06_phone_demo_predictions.csv) for the phone demonstration results.

## Repo structure

```
gait-fall-risk-classifier/
├── data/
│   ├── phone/                      # recordings from own phone's accelerometer, using Sensor Logger App, gitignored
│   ├── raw/                        # untouched SisFall files, gitignored
│   └── processed/                  # windowed, feature-extracted CSVs, gitignored
├── scripts/
│   ├── 01_data_exploration.py
│   ├── 02_preprocessing.py
│   ├── 03_feature_extraction.py
│   ├── 04_train_classifier.py
│   ├── 05_evaluate.py
│   └── 06_phone_demo.py
├── models/                         # saved .pkl classifier, gitignored
├── results/           
├── README.md
├── requirements.txt
└── .gitignore
```

## Reproducing this

``` cmd
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```

Download SisFall (original, raw format — see `report.md` for source and verification notes) into `data/raw/`, then:

``` cmd
python scripts\01_data_exploration.py
python scripts\02_preprocessing.py
python scripts\03_feature_extraction.py
python scripts\04_train_classifier.py
python scripts\05_evaluate.py
```

`scripts\06_phone_demo.py` is optional and requires your own phone-recorded accelerometer + gyroscope CSVs in `data/phone/`.

## Known limitations

- **Elderly fall recall (89.3%) is a single-subject result, not a population estimate.** SisFall's elderly cohort (14 subjects) performed ADLs only; exactly one elderly participant performed the full ADL + fall protocol. This is a structural limit of the dataset itself.
- Slow-onset falls (collapsing while already seated, SisFall code F11) are missed more often than sudden falls — the peak-centered windowing approach that fixed the labeling problem is tuned to sharp impact spikes, and a gradual collapse doesn't always produce one.
- Trial-level window labeling for ADL (unlike the peak-centered fall windows) means some ADL activities that closely resemble fall kinematics — deliberately collapsing into a chair, changing lying position — produce the model's few false positives.

## Dataset

[SisFall: A Fall and Movement Dataset](https://doi.org/10.3390/s17010198) (Sucerquia, López, Vargas-Bonilla, 2017). Sourced via a Kaggle mirror during this build due to the original host (sistemic.udea.edu.co) being intermittently unreachable. The file counts and structure were verified against the published paper's specification (4,505 trials, 38 subjects) before use.

---

#### Presented By: Susmit Acharya

# Cloud-Based ML DDoS/Intrusion Detection Service

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/RohithDharshan/Cloud-DDoS-Detector)

Team project for Cloud Computing (23N014) — implementation based on:

> S. Abiramasundari and V. Ramaswamy, "Distributed denial-of-service (DDOS) attack
> detection using supervised machine learning algorithms," *Scientific Reports*,
> vol. 15, art. 13098, 2025. DOI: [10.1038/s41598-024-84879-y](https://doi.org/10.1038/s41598-024-84879-y)
> (open access, CC BY 4.0 — full PDF in [`docs/paper/`](docs/paper/)).

The paper compares SVM, Logistic Regression, Random Forest, KNN, and Decision Tree
on CICIDS2017 / CICIDS2018 / CICDDoS2019, using StandardScaler + PCA preprocessing
and SMOTE-style oversampling for class imbalance. Random Forest was their best
performer on CICIDS2017 at **98.9% accuracy**. This project reproduces that
pipeline and wraps the trained model as a real-time monitoring microservice —
the "innovate on top of it" part of the assignment, since the paper itself
stops at offline model evaluation and doesn't build a deployable service.

**See [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) for a step-by-step mapping
of exactly which paper method each file in this repo implements, and how our real
results compare to the paper's.**

## Live deployment

**Live now: [cloud-ddos-detector.onrender.com](https://cloud-ddos-detector.onrender.com/)**
— the real dashboard, running on Render's cloud infrastructure, not localhost.
(Free tier: it spins down after 15 minutes idle and takes ~30-50s to wake back up
on the first request — normal, not a bug. Worth a warm-up click a minute before
presenting.)

Click the "Deploy to Render" button above to spin up your own instance (free tier,
no credit card required — Render auto-detects `render.yaml` and the `Dockerfile`).
Render connects to your GitHub account once, then builds and deploys this repo
directly; the live URL it gives you serves the same dashboard as running locally.

## What's here

```
ddos-cloud-detector/
├── requirements.txt
├── Dockerfile / docker-compose.yml / .dockerignore / render.yaml
├── docs/
│   ├── IMPLEMENTATION.md        # paper-to-code mapping, the "we implemented it" evidence
│   └── paper/                   # the reference paper's open-access PDF
├── data/raw/                    # dataset CSVs live here (gitignored size-wise)
├── model/artifacts/             # trained model + fitted scaler/PCA + metrics
└── src/
    ├── download_dataset.py      # instructions to fetch the REAL CICIDS2017 file
    ├── generate_sample_data.py  # synthetic fallback (same 78-column schema)
    ├── preprocess.py            # clean -> label encode -> split -> oversample -> scale -> PCA
    ├── train.py                 # trains RandomForest (+ LR/GB for comparison), saves metrics.json
    ├── service.py                # FastAPI monitoring service (/predict, /predict/batch)
    ├── demo_traffic.py           # streams flows at the running service for a live demo
    ├── export_sample_flows.py    # regenerates frontend/sample_flows.json from a real CSV
    └── export_dataset_summary.py # regenerates frontend/dataset_summary.json (dashboard "Dataset" section)
```

## Dataset

`data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv` is the **real**
CIC-IDS2017 Friday-afternoon capture (Canadian Institute for Cybersecurity,
UNB) — 225,745 real network flows, 97,686 BENIGN / 128,025 DDoS, official
`MachineLearningCSV` release, CICFlowMeter-extracted features. This is the
single file the reference paper's DDoS-specific Random Forest benchmark is
built on. `data/raw/sample_cicids2017.csv` (synthetic, same schema) is kept
only as an offline fallback if you ever need to re-run without the real file.

To fetch it yourself on a new machine, pick ONE:
- **Kaggle mirror** (needs a Kaggle account + `~/.kaggle/kaggle.json` API token):
  ```bash
  pip install kaggle
  kaggle datasets download -d shadman1028/cicids2017-official-flow-feature-csv-files \
    -f "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" -p data/raw
  # then rename to match the expected filename (note the CSV's "DDos" vs "DDoS"):
  mv "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" \
     "data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"
  ```
- **Official UNB source**: https://www.unb.ca/cic/datasets/ids-2017.html → "Download this dataset"
  → fill the CIC request form (name/email/org, for their stats only) → grab `MachineLearningCSV.zip`
  → place `Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv` at `data/raw/`.

Either way, after placing the file, regenerate the artifacts the dashboard depends on:
```bash
python3 src/preprocess.py --input data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv --output-dir model/artifacts
python3 src/train.py --artifacts-dir model/artifacts --compare
python3 src/export_dataset_summary.py --input data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
python3 src/export_sample_flows.py --input data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the full pipeline

The real dataset is already in `data/raw/` and `model/artifacts/` already
holds the model trained on it (see Results below) — you only need to re-run
this if you re-fetch the data or want to retrain:

```bash
# 1. Get data — see "Dataset" section above (already done; real CSV is in data/raw/)

# 2. Preprocess
python3 src/preprocess.py --input data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv --output-dir model/artifacts

# 3. Train + evaluate (writes model/artifacts/model.joblib and metrics.json)
python3 src/train.py --artifacts-dir model/artifacts --compare

# 4. Refresh the dashboard's dataset preview + demo sample flows
python3 src/export_dataset_summary.py --input data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
python3 src/export_sample_flows.py --input data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
```

## Run the monitoring service

Locally:
```bash
uvicorn src.service:app --host 0.0.0.0 --port 8000
```

Or with Docker (`docker` + `docker compose` on your machine):
```bash
docker compose up --build
```
This builds a container with the trained artifacts baked in and exposes the
API on `http://localhost:8000`.

Endpoints:
- `GET /health` — model status
- `GET /features` — the 78 raw feature names the service expects, in order
- `POST /predict` — `{"features": {"Flow Duration": 123, "Flow Bytes/s": 4.5, ...}}` → `{"prediction": "ATTACK"/"BENIGN", "attack_probability": 0.0-1.0, "latency_ms": ...}`
- `POST /predict/batch` — `{"records": [{"features": {...}}, ...]}` for scoring several flows in one call

## Live demo (for the presentation)

**Option A — browser dashboard (recommended for the presentation):**
With the service running, open **http://localhost:8000/** in a browser (it
redirects to `/dashboard/`). It shows the detection pipeline, the model
metrics table, and a "Start monitoring" button that streams real sample
flows through the live model with a running feed and accuracy counter —
plus buttons to fire a single benign or DDoS flow on demand. Everything
on the page talks directly to this same running service, so it's safe to
project on screen and click through live.

**Option B — terminal script:**
```bash
python3 src/demo_traffic.py --n 40 --delay 0.3
```
This streams 40 real flow records from the dataset to the service one at a
time, printing each prediction next to the ground-truth label, and finishes
with an accuracy/confusion-matrix summary. `--delay 0` for a fast run, higher
for a slower, more presentable pace.

## Results (real CIC-IDS2017 data — Friday-afternoon DDoS capture)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
|---|---|---|---|---|---|---|
| Random Forest | 99.98% | 99.99% | 99.98% | 99.98% | 1.000 | 17.12s |
| Logistic Regression | 99.88% | 99.89% | 99.90% | 99.89% | 1.000 | 0.32s |
| Gradient Boosting | 99.93% | 99.97% | 99.91% | 99.94% | 1.000 | 178.03s |

(Full numbers in `model/artifacts/metrics.json`.) These are close to but a
bit above the reference paper's 98.9% — expected, since the paper's number is
averaged across the full 5-day, multi-attack-type CICIDS2017 set, while this
is the single-day, two-class (BENIGN vs DDoS) slice, an easier separation
task. Worth calling out explicitly in the report as a scope difference, not
a claim of beating the paper.

## What's left for the team

- [x] Swap in the real CICIDS2017 file (Friday-afternoon DDoS capture, 225,745
      real flows) and retrain — `model/artifacts/metrics.json` above is the
      real result.
- [x] Document the paper-to-code mapping (`docs/IMPLEMENTATION.md`) with the
      paper's PDF in the repo as evidence.
- [ ] **Click the "Deploy to Render" button above** (one team member, 2 minutes)
      to get a public live URL — Render needs a GitHub login the first time,
      which only a human can do; everything else is already configured
      (`render.yaml` + `Dockerfile`).
- [ ] Write the report itself (methodology, comparison to the paper, your
      "service" contribution as the innovation on top of it, screenshots of
      the demo).
- [ ] Practice the live demo (`demo_traffic.py`) for the Oct 1 presentation.
- [ ] Fill in the team gsheet (team members, topic, paper link) if not done already.

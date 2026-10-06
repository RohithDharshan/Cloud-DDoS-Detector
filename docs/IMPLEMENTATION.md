# Paper-to-code mapping

This project reproduces, then extends, the method in:

> S. Abiramasundari and V. Ramaswamy, "Distributed denial-of-service (DDOS) attack
> detection using supervised machine learning algorithms," *Scientific Reports*,
> vol. 15, art. 13098, 2025. DOI: [10.1038/s41598-024-84879-y](https://doi.org/10.1038/s41598-024-84879-y)
> (open access, CC BY 4.0 — full PDF in [`docs/paper/`](paper/)).

This document is the evidence trail: for every step the paper describes, which file
in this repo implements it, and where our results and theirs line up or diverge.

## What the paper does

The paper benchmarks five supervised classifiers — SVM, Logistic Regression, Random
Forest, KNN, and Decision Tree — on CICIDS2017 (plus CICIDS2018 and CICDDoS2019 as
extra benchmarks), using a pipeline of: label encoding, StandardScaler normalization,
SMOTE-style oversampling for class imbalance, and PCA dimensionality reduction
(79 raw features down to 61 components). Random Forest is their best performer on
CICIDS2017 at **98.9% accuracy**. The paper stops at this offline evaluation — it
does not build or deploy anything.

## Step-by-step mapping

| # | Paper step | Our file | What we did |
|---|---|---|---|
| 1 | Dataset: CICIDS2017 | [`data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv`](../data/raw/) | The real CIC-IDS2017 Friday-afternoon DDoS capture (UNB/CIC official release), 225,745 flows — not synthetic data |
| 2 | Label encoding | [`src/preprocess.py`](../src/preprocess.py) `encode_labels()` | BENIGN → 0, any attack label → 1 |
| 3 | 70/30 train/test split | [`src/preprocess.py`](../src/preprocess.py) `main()` | Stratified 70/30 split, exactly as the paper specifies |
| 4 | Oversampling for class imbalance | [`src/preprocess.py`](../src/preprocess.py) (`RandomOverSampler`) | Fit on the TRAIN split only, avoiding test-set leakage |
| 5 | StandardScaler normalization | [`src/preprocess.py`](../src/preprocess.py) | Fit on train only, applied to both splits |
| 6 | PCA, 79 → 61 features | [`src/preprocess.py`](../src/preprocess.py) `--pca-variance-ratio` | Same 61/79 ratio applied to our 78 raw CICIDS2017 columns → 60 components, retaining 100% of variance |
| 7 | Compare SVM / LR / RF / KNN / Decision Tree | [`src/train.py`](../src/train.py) | We reproduce **Random Forest** (the paper's best performer) and add **Logistic Regression** and **Gradient Boosting** as our own comparison set. SVM, KNN and plain Decision Tree are out of scope here — our effort went into the deployment half (steps 8–9) rather than re-running all five algorithms |
| 8 | Report accuracy (98.9% RF on CICIDS2017) | [`model/artifacts/metrics.json`](../model/artifacts/metrics.json) | Our Random Forest: **99.98%** accuracy on the real Friday-afternoon capture. Slightly higher because we train on one day's **binary** BENIGN-vs-DDoS task, while the paper's 98.9% is averaged across 5 days and multiple attack types — a harder, broader problem. Same method, an easier slice of data. |
| 9 | *(not in the paper — our innovation)* | [`src/service.py`](../src/service.py), [`Dockerfile`](../Dockerfile), [`frontend/index.html`](../frontend/index.html) | The paper stops at an offline notebook result. We package the trained model as a live FastAPI microservice, containerize it with Docker, and ship a real-time animated monitoring dashboard — turning the benchmark into a deployed, running cloud service |

## Honest scope notes

- We target the paper's **DDoS-specific** benchmark (the Friday-afternoon capture), not its full multi-day, multi-attack-type CICIDS2017 run — that's why our accuracy is a little higher, and we say so in the presentation rather than let it look like we beat the paper outright.
- We did not reproduce SVM, KNN, or plain Decision Tree from the paper's five-model comparison; Random Forest was their best result and the one we carry through to production.
- Everything in the table above is checkable: run `python src/preprocess.py` and `python src/train.py` yourself against [`model/artifacts/metrics.json`](../model/artifacts/metrics.json) for the real numbers.

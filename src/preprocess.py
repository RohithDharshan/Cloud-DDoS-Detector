"""
Preprocessing pipeline for CICIDS2017-style flow data, following the method
described in Abiramasundari & Ramaswamy (2025, Scientific Reports 15:13098):

    1. Clean (strip column names, drop inf/NaN rows — CICIDS2017 is known to
       contain literal 'Infinity' values in Flow Bytes/s and Flow Packets/s
       for zero-duration flows)
    2. Binary label encoding: BENIGN -> 0, any attack label -> 1
    3. Stratified 70/30 train/test split
    4. Class-imbalance handling via oversampling the minority class
       (fit on TRAIN ONLY, to avoid leaking test-set information)
    5. StandardScaler normalization (fit on TRAIN ONLY)
    6. PCA dimensionality reduction (fit on TRAIN ONLY) — the paper reduces
       79 raw columns to 61 principal components; this script mirrors that
       ratio automatically for whatever feature count the input file has.

Outputs (into --output-dir):
    X_train.npy, X_test.npy, y_train.npy, y_test.npy
    scaler.joblib, pca.joblib          <- reused unchanged by the serving API
    feature_columns.json               <- the RAW column order the API must
                                           receive requests in, before scaling/PCA

Usage:
    python src/preprocess.py --input data/raw/sample_cicids2017.csv --output-dir model/artifacts
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "Label" not in df.columns:
        raise ValueError(f"Expected a 'Label' column, got columns: {list(df.columns)[:5]}...")

    # CICIDS2017 stores +/-Infinity as strings or floats in a couple of
    # rate columns when Flow Duration is 0. Replace with NaN, then drop.
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows containing NaN/Infinity ({dropped/before:.2%} of data)")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Label"] = df["Label"].astype(str).str.strip()
    df["y"] = (df["Label"].str.upper() != "BENIGN").astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the CICIDS2017 CSV (real or synthetic)")
    ap.add_argument("--output-dir", default="model/artifacts")
    ap.add_argument("--test-size", type=float, default=0.30, help="Matches the paper's 70/30 split")
    ap.add_argument("--pca-variance-ratio", type=float, default=61 / 79,
                     help="Fraction of raw feature count to keep as principal "
                          "components, mirroring the paper's 79->61 reduction")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input} ...")
    df = load_and_clean(args.input)
    df = encode_labels(df)

    feature_columns = [c for c in df.columns if c not in ("Label", "y")]
    X = df[feature_columns].astype(float).values
    y = df["y"].values

    print(f"Loaded {len(df)} rows, {len(feature_columns)} raw features. "
          f"Class balance: {np.bincount(y)} (0=BENIGN, 1=ATTACK)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Oversample the minority class in TRAIN only
    ros = RandomOverSampler(random_state=args.seed)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print(f"After oversampling train set: {np.bincount(y_train)}")

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA (fit on train only)
    n_components = max(1, round(len(feature_columns) * args.pca_variance_ratio))
    n_components = min(n_components, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {len(feature_columns)} -> {n_components} components "
          f"({explained:.1%} variance retained)")

    np.save(os.path.join(args.output_dir, "X_train.npy"), X_train_pca)
    np.save(os.path.join(args.output_dir, "X_test.npy"), X_test_pca)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test)
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))
    joblib.dump(pca, os.path.join(args.output_dir, "pca.joblib"))
    with open(os.path.join(args.output_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    print(f"\nSaved preprocessed arrays + fitted scaler/PCA to {args.output_dir}/")


if __name__ == "__main__":
    main()

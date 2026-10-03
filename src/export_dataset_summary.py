"""
Exports a small JSON summary of the raw CICIDS2017 CSV (row/column counts,
class balance, a handful of sample rows) into frontend/dataset_summary.json,
which the dashboard's "Dataset" section reads to showcase the real data
before walking through the pipeline.

Usage:
    python src/export_dataset_summary.py --input data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

PREVIEW_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count", "Label",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="frontend/dataset_summary.json")
    ap.add_argument("--n-preview-rows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    raw_bytes = os.path.getsize(args.input)
    df = pd.read_csv(args.input, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    n_raw_rows = len(df)

    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    df_clean["Label"] = df_clean["Label"].astype(str).str.strip()

    label_counts = df_clean["Label"].value_counts()
    n_benign = int((df_clean["Label"].str.upper() == "BENIGN").sum())
    n_attack = int((df_clean["Label"].str.upper() != "BENIGN").sum())

    preview_cols = [c for c in PREVIEW_COLUMNS if c in df_clean.columns]
    preview = df_clean[preview_cols].sample(
        n=min(args.n_preview_rows, len(df_clean)), random_state=args.seed
    ).round(3).to_dict(orient="records")

    summary = {
        "source_file": os.path.basename(args.input),
        "source": "CIC-IDS2017 (Canadian Institute for Cybersecurity, UNB) — "
                   "Friday afternoon DDoS capture, official MachineLearningCSV release",
        "size_mb": round(raw_bytes / (1024 * 1024), 1),
        "n_rows_raw": n_raw_rows,
        "n_rows_clean": len(df_clean),
        "n_columns": len(df.columns),
        "n_features": len(df.columns) - 1,
        "n_benign": n_benign,
        "n_attack": n_attack,
        "label_breakdown": {str(k): int(v) for k, v in label_counts.items()},
        "preview_columns": preview_cols,
        "preview_rows": preview,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote dataset summary ({n_raw_rows} rows, {n_benign} benign / {n_attack} attack) to {args.output}")


if __name__ == "__main__":
    main()

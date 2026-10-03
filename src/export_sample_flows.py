"""
Exports a small shuffled sample of real flow records (BENIGN + DDoS, balanced)
from a CICIDS2017 CSV into frontend/sample_flows.json, which the dashboard's
"Live traffic monitor" and "Try it yourself" panels stream against.

Run this once whenever the dataset backing the demo changes (e.g. after
swapping in the real CICIDS2017 file), so the presentation streams genuine
flow records instead of stale/synthetic ones.

Usage:
    python src/export_sample_flows.py --input data/raw/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
"""

import argparse
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="frontend/sample_flows.json")
    ap.add_argument("--n-per-class", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["Label"] = df["Label"].astype(str).str.strip()

    benign = df[df["Label"].str.upper() == "BENIGN"].sample(
        n=min(args.n_per_class, (df["Label"].str.upper() == "BENIGN").sum()), random_state=args.seed
    )
    attack = df[df["Label"].str.upper() != "BENIGN"].sample(
        n=min(args.n_per_class, (df["Label"].str.upper() != "BENIGN").sum()), random_state=args.seed
    )
    sample = pd.concat([benign, attack]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    records = []
    for _, row in sample.iterrows():
        true_label = row["Label"]
        features = row.drop("Label").astype(float).to_dict()
        records.append({"true_label": true_label, "features": features})

    with open(args.output, "w") as f:
        json.dump(records, f)

    print(f"Wrote {len(records)} real flow records ({args.n_per_class} benign / {args.n_per_class} attack, "
          f"before dedup) to {args.output}")


if __name__ == "__main__":
    main()

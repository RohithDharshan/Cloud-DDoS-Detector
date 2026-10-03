"""
Generates a small SYNTHETIC dataset that matches the exact 79-column schema
of the real CICIDS2017 "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"
file (78 CICFlowMeter features + Label).

Why this exists: the sandbox this pipeline was built in can't reach
Kaggle/Drive/UNB to pull the real ~90MB dataset. This script lets the whole
preprocessing -> training -> serving pipeline be built, run, and verified
end-to-end right now, using data that is shaped exactly like the real thing
(same columns, roughly realistic value ranges and a genuine BENIGN vs DDoS
signal). Swap this file for the real one later — same column names, so
nothing else in the pipeline needs to change.

Usage:
    python src/generate_sample_data.py --n-benign 4000 --n-ddos 4000 \
        --out data/raw/sample_cicids2017.csv
"""

import argparse
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def _benign_rows(n, rng):
    """Roughly realistic everyday traffic: longer flows, moderate packet
    rate, few SYN-only flags, balanced fwd/bwd traffic."""
    df = pd.DataFrame(index=range(n))
    df["Destination Port"] = rng.choice([80, 443, 22, 53, 3306, 8080], size=n)
    df["Flow Duration"] = rng.lognormal(mean=11, sigma=1.5, size=n).astype(int)  # microseconds, longer flows
    df["Total Fwd Packets"] = rng.poisson(12, size=n) + 1
    df["Total Backward Packets"] = rng.poisson(10, size=n) + 1
    df["Total Length of Fwd Packets"] = df["Total Fwd Packets"] * rng.normal(500, 150, size=n).clip(40)
    df["Total Length of Bwd Packets"] = df["Total Backward Packets"] * rng.normal(600, 200, size=n).clip(40)
    df["Fwd Packet Length Max"] = rng.normal(800, 200, size=n).clip(40)
    df["Fwd Packet Length Min"] = rng.normal(60, 20, size=n).clip(0)
    df["Fwd Packet Length Mean"] = rng.normal(400, 100, size=n).clip(0)
    df["Fwd Packet Length Std"] = rng.normal(150, 50, size=n).clip(0)
    df["Bwd Packet Length Max"] = rng.normal(900, 250, size=n).clip(40)
    df["Bwd Packet Length Min"] = rng.normal(60, 20, size=n).clip(0)
    df["Bwd Packet Length Mean"] = rng.normal(450, 120, size=n).clip(0)
    df["Bwd Packet Length Std"] = rng.normal(160, 60, size=n).clip(0)
    df["Flow Bytes/s"] = rng.normal(5000, 2000, size=n).clip(0)
    df["Flow Packets/s"] = rng.normal(30, 15, size=n).clip(0.1)
    for col in ["Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
                "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max"]:
        df[col] = rng.lognormal(mean=8, sigma=1.2, size=n)
    for col in ["Flow IAT Min", "Fwd IAT Min", "Bwd IAT Min"]:
        df[col] = rng.exponential(500, size=n)
    df["Fwd IAT Total"] = df["Fwd IAT Mean"] * df["Total Fwd Packets"]
    df["Bwd IAT Total"] = df["Bwd IAT Mean"] * df["Total Backward Packets"]
    df["Fwd PSH Flags"] = rng.binomial(1, 0.3, size=n)
    df["Bwd PSH Flags"] = rng.binomial(1, 0.3, size=n)
    df["Fwd URG Flags"] = 0
    df["Bwd URG Flags"] = 0
    df["Fwd Header Length"] = df["Total Fwd Packets"] * 32
    df["Bwd Header Length"] = df["Total Backward Packets"] * 32
    df["Fwd Packets/s"] = df["Total Fwd Packets"] / (df["Flow Duration"] / 1e6).clip(1e-3)
    df["Bwd Packets/s"] = df["Total Backward Packets"] / (df["Flow Duration"] / 1e6).clip(1e-3)
    df["Min Packet Length"] = rng.normal(40, 10, size=n).clip(0)
    df["Max Packet Length"] = rng.normal(1000, 200, size=n).clip(40)
    df["Packet Length Mean"] = rng.normal(450, 100, size=n).clip(0)
    df["Packet Length Std"] = rng.normal(200, 60, size=n).clip(0)
    df["Packet Length Variance"] = df["Packet Length Std"] ** 2
    df["FIN Flag Count"] = rng.binomial(1, 0.4, size=n)
    df["SYN Flag Count"] = rng.binomial(1, 0.5, size=n)
    df["RST Flag Count"] = rng.binomial(1, 0.05, size=n)
    df["PSH Flag Count"] = rng.binomial(1, 0.3, size=n)
    df["ACK Flag Count"] = rng.binomial(1, 0.9, size=n)
    df["URG Flag Count"] = 0
    df["CWE Flag Count"] = 0
    df["ECE Flag Count"] = 0
    df["Down/Up Ratio"] = rng.normal(1, 0.3, size=n).clip(0)
    df["Average Packet Size"] = df["Packet Length Mean"]
    df["Avg Fwd Segment Size"] = df["Fwd Packet Length Mean"]
    df["Avg Bwd Segment Size"] = df["Bwd Packet Length Mean"]
    df["Fwd Header Length.1"] = df["Fwd Header Length"]
    for col in ["Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
                "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"]:
        df[col] = 0
    df["Subflow Fwd Packets"] = df["Total Fwd Packets"]
    df["Subflow Fwd Bytes"] = df["Total Length of Fwd Packets"]
    df["Subflow Bwd Packets"] = df["Total Backward Packets"]
    df["Subflow Bwd Bytes"] = df["Total Length of Bwd Packets"]
    df["Init_Win_bytes_forward"] = rng.choice([8192, 16384, 29200, 65535], size=n)
    df["Init_Win_bytes_backward"] = rng.choice([8192, 16384, 29200, 65535], size=n)
    df["act_data_pkt_fwd"] = df["Total Fwd Packets"] - 1
    df["min_seg_size_forward"] = rng.choice([20, 32], size=n)
    for col in ["Active Mean", "Active Std", "Active Max", "Active Min",
                "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]:
        df[col] = rng.exponential(2000, size=n)
    df["Label"] = "BENIGN"
    return df


def _ddos_rows(n, rng):
    """DDoS flood signature: very short flow duration, extreme packet rate,
    tiny/uniform packet sizes, lopsided fwd-heavy traffic, elevated SYN/ACK
    flag counts — the pattern a volumetric flood actually produces."""
    df = pd.DataFrame(index=range(n))
    df["Destination Port"] = rng.choice([80, 443, 53], size=n)
    df["Flow Duration"] = rng.exponential(500, size=n).astype(int)  # very short flows
    df["Total Fwd Packets"] = rng.poisson(200, size=n) + 50
    df["Total Backward Packets"] = rng.poisson(2, size=n)
    df["Total Length of Fwd Packets"] = df["Total Fwd Packets"] * rng.normal(60, 15, size=n).clip(20)
    df["Total Length of Bwd Packets"] = df["Total Backward Packets"] * rng.normal(60, 20, size=n).clip(0)
    df["Fwd Packet Length Max"] = rng.normal(70, 15, size=n).clip(20)
    df["Fwd Packet Length Min"] = rng.normal(40, 10, size=n).clip(0)
    df["Fwd Packet Length Mean"] = rng.normal(55, 10, size=n).clip(0)
    df["Fwd Packet Length Std"] = rng.normal(10, 5, size=n).clip(0)
    df["Bwd Packet Length Max"] = rng.normal(60, 20, size=n).clip(0)
    df["Bwd Packet Length Min"] = 0
    df["Bwd Packet Length Mean"] = rng.normal(20, 10, size=n).clip(0)
    df["Bwd Packet Length Std"] = rng.normal(10, 5, size=n).clip(0)
    df["Flow Bytes/s"] = rng.normal(500000, 200000, size=n).clip(0)  # huge byte rate
    df["Flow Packets/s"] = rng.normal(4000, 1500, size=n).clip(1)   # huge packet rate
    for col in ["Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
                "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max"]:
        df[col] = rng.exponential(30, size=n)  # tiny inter-arrival times = flood
    for col in ["Flow IAT Min", "Fwd IAT Min", "Bwd IAT Min"]:
        df[col] = rng.exponential(5, size=n)
    df["Fwd IAT Total"] = df["Fwd IAT Mean"] * df["Total Fwd Packets"]
    df["Bwd IAT Total"] = df["Bwd IAT Mean"] * df["Total Backward Packets"]
    df["Fwd PSH Flags"] = rng.binomial(1, 0.05, size=n)
    df["Bwd PSH Flags"] = 0
    df["Fwd URG Flags"] = 0
    df["Bwd URG Flags"] = 0
    df["Fwd Header Length"] = df["Total Fwd Packets"] * 32
    df["Bwd Header Length"] = df["Total Backward Packets"] * 32
    df["Fwd Packets/s"] = df["Total Fwd Packets"] / (df["Flow Duration"] / 1e6).clip(1e-3)
    df["Bwd Packets/s"] = df["Total Backward Packets"] / (df["Flow Duration"] / 1e6).clip(1e-3)
    df["Min Packet Length"] = rng.normal(20, 5, size=n).clip(0)
    df["Max Packet Length"] = rng.normal(80, 15, size=n).clip(20)
    df["Packet Length Mean"] = rng.normal(50, 10, size=n).clip(0)
    df["Packet Length Std"] = rng.normal(8, 3, size=n).clip(0)
    df["Packet Length Variance"] = df["Packet Length Std"] ** 2
    df["FIN Flag Count"] = 0
    df["SYN Flag Count"] = rng.binomial(1, 0.9, size=n)  # SYN-flood signature
    df["RST Flag Count"] = rng.binomial(1, 0.1, size=n)
    df["PSH Flag Count"] = rng.binomial(1, 0.05, size=n)
    df["ACK Flag Count"] = rng.binomial(1, 0.2, size=n)
    df["URG Flag Count"] = 0
    df["CWE Flag Count"] = 0
    df["ECE Flag Count"] = 0
    df["Down/Up Ratio"] = rng.normal(0.05, 0.02, size=n).clip(0)  # almost no return traffic
    df["Average Packet Size"] = df["Packet Length Mean"]
    df["Avg Fwd Segment Size"] = df["Fwd Packet Length Mean"]
    df["Avg Bwd Segment Size"] = df["Bwd Packet Length Mean"]
    df["Fwd Header Length.1"] = df["Fwd Header Length"]
    for col in ["Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
                "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"]:
        df[col] = 0
    df["Subflow Fwd Packets"] = df["Total Fwd Packets"]
    df["Subflow Fwd Bytes"] = df["Total Length of Fwd Packets"]
    df["Subflow Bwd Packets"] = df["Total Backward Packets"]
    df["Subflow Bwd Bytes"] = df["Total Length of Bwd Packets"]
    df["Init_Win_bytes_forward"] = rng.choice([256, 512, 1024], size=n)
    df["Init_Win_bytes_backward"] = -1  # CICIDS2017 uses -1 when no backward window observed
    df["act_data_pkt_fwd"] = df["Total Fwd Packets"] - 1
    df["min_seg_size_forward"] = 20
    for col in ["Active Mean", "Active Std", "Active Max", "Active Min",
                "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]:
        df[col] = rng.exponential(20, size=n)
    df["Label"] = "DDoS"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-benign", type=int, default=4000)
    ap.add_argument("--n-ddos", type=int, default=4000)
    ap.add_argument("--out", type=str, default="data/raw/sample_cicids2017.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    benign = _benign_rows(args.n_benign, rng)
    ddos = _ddos_rows(args.n_ddos, rng)
    df = pd.concat([benign, ddos], ignore_index=True)
    df = df[FEATURE_COLUMNS + ["Label"]]
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # shuffle

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows ({args.n_benign} BENIGN, {args.n_ddos} DDoS) to {args.out}")


if __name__ == "__main__":
    main()

"""
Demo / traffic-simulation script for the presentation.

Streams a shuffled mix of BENIGN and DDoS flow records from the sample (or
real, once swapped in) CICIDS2017 CSV to the running monitoring service one
at a time, printing a live "traffic monitor" style line for each — this is
what you show on screen during the presentation to prove the service detects
attacks in real time.

At the end it prints a small accuracy summary comparing the service's
predictions against the ground-truth Label column, which is a nice number to
drop straight into the report / final slide.

Usage (service must already be running, e.g. `uvicorn src.service:app`
or the Docker container):
    python src/demo_traffic.py --csv data/raw/sample_cicids2017.csv \
        --n 40 --delay 0.3 --url http://localhost:8000
"""

import argparse
import time

import pandas as pd
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/sample_cicids2017.csv")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=40, help="number of flows to stream")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between requests")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    try:
        health = requests.get(f"{args.url}/health", timeout=5).json()
    except requests.exceptions.ConnectionError:
        print(f"Could not reach {args.url} — start the service first:\n"
              f"  uvicorn src.service:app --host 0.0.0.0 --port 8000\n"
              f"or:  docker compose up")
        return
    print(f"Connected to monitoring service: {health}\n")
    print(f"{'#':>3}  {'TRUE LABEL':<10} {'PREDICTED':<10} {'P(attack)':<10} {'LATENCY':<10} RESULT")
    print("-" * 65)

    correct = 0
    tp = fp = tn = fn = 0

    for i, row in sample.iterrows():
        true_label = str(row["Label"]).strip()
        true_attack = true_label.upper() != "BENIGN"
        features = row.drop("Label").astype(float).to_dict()

        t0 = time.time()
        resp = requests.post(f"{args.url}/predict", json={"features": features}, timeout=5)
        wall_ms = (time.time() - t0) * 1000
        result = resp.json()

        pred_attack = result["is_attack"]
        is_correct = pred_attack == true_attack
        correct += is_correct
        if pred_attack and true_attack:
            tp += 1
        elif pred_attack and not true_attack:
            fp += 1
        elif not pred_attack and not true_attack:
            tn += 1
        else:
            fn += 1

        flag = "OK" if is_correct else "MISMATCH"
        marker = "🚨 ATTACK" if pred_attack else "  benign "
        print(f"{i+1:>3}  {true_label:<10} {result['prediction']:<10} "
              f"{result['attack_probability']:<10.3f} {wall_ms:>6.1f}ms   {marker}  [{flag}]")

        time.sleep(args.delay)

    n = len(sample)
    print("-" * 65)
    print(f"Streamed {n} flows | accuracy vs ground truth: {correct}/{n} ({correct/n:.1%})")
    print(f"Confusion matrix  TP={tp}  FP={fp}  TN={tn}  FN={fn}")


if __name__ == "__main__":
    main()

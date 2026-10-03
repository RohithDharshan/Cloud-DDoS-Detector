"""
Trains the DDoS classifier on the preprocessed CICIDS2017 arrays and reports
the metrics your report needs (the reference paper's headline number is
98.9% accuracy for Random Forest on CICIDS2017).

Usage:
    python src/train.py --artifacts-dir model/artifacts --compare
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)


def evaluate(name, model, X_test, y_test, timings):
    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - t0
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_time_sec": timings.get("train", None),
        "inference_time_sec_per_1000": infer_time / max(1, len(y_test)) * 1000,
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"], zero_division=0))
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{np.array(metrics['confusion_matrix'])}")
    print(f"Accuracy={metrics['accuracy']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
          f"Train time={timings.get('train', 0):.2f}s  "
          f"Inference={metrics['inference_time_sec_per_1000']*1000:.3f}ms/1000 rows")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", default="model/artifacts")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--compare", action="store_true",
                     help="Also train Logistic Regression and Gradient Boosting "
                          "for a comparison table in the report")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = args.artifacts_dir
    X_train = np.load(os.path.join(d, "X_train.npy"))
    X_test = np.load(os.path.join(d, "X_test.npy"))
    y_train = np.load(os.path.join(d, "y_train.npy"))
    y_test = np.load(os.path.join(d, "y_test.npy"))

    all_metrics = []

    # --- Primary model: Random Forest (matches the reference paper) ---
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=None, n_jobs=-1,
        random_state=args.seed, class_weight="balanced",
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0
    all_metrics.append(evaluate("RandomForest", rf, X_test, y_test, {"train": rf_train_time}))
    joblib.dump(rf, os.path.join(d, "model.joblib"))

    if args.compare:
        lr = LogisticRegression(max_iter=1000, random_state=args.seed)
        t0 = time.time()
        lr.fit(X_train, y_train)
        all_metrics.append(evaluate("LogisticRegression", lr, X_test, y_test, {"train": time.time() - t0}))

        gb = GradientBoostingClassifier(random_state=args.seed)
        t0 = time.time()
        gb.fit(X_train, y_train)
        all_metrics.append(evaluate("GradientBoosting", gb, X_test, y_test, {"train": time.time() - t0}))

    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved primary model to {d}/model.joblib and metrics to {d}/metrics.json")
    best = max(all_metrics, key=lambda m: m["f1"])
    print(f"Best by F1: {best['model']} (F1={best['f1']:.4f}, accuracy={best['accuracy']:.4f})")


if __name__ == "__main__":
    main()

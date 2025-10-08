#!/usr/bin/env python3
"""
score_all.py

Usage:
  python score_all.py features.csv model.joblib ranked_nodes.json

Produces a JSON array of nodes with:
- id, label, utility_prob, core_importance, classification, features
"""
import argparse
import pandas as pd
import numpy as np
import joblib
import json

def main(features_csv, model_joblib, out_json, threshold=0.6):
    data = pd.read_csv(features_csv)
    model_bundle = joblib.load(model_joblib)
    model = model_bundle["model"]
    feature_cols = model_bundle["features"]
    # Ensure cols present
    X = data[feature_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]  # probability of core (label 1)
    # If model was trained with label_manual 1=core, fine. If reversed, adjust.
    # We'll interpret higher proba as "core_importance".
    results = []
    for i, row in data.iterrows():
        prob_core = float(probs[i])
        core_importance = round(prob_core, 3)
        utility_prob = round(1.0 - core_importance, 3)
        classification = "core" if core_importance >= threshold else "utility"
        # include features subset for transparency
        features = {c: (row[c] if c in row else None) for c in feature_cols}
        results.append({
            "id": row.get("id"),
            "label": row.get("label"),
            "utility_prob": utility_prob,
            "core_importance": core_importance,
            "classification": classification,
            "features": features
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Wrote ranked results to {out_json} (rows: {len(results)})")
    print(f"Note: classification threshold used = {threshold}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score all nodes using a trained model")
    parser.add_argument("features_csv", help="features CSV")
    parser.add_argument("model_joblib", help="model.joblib saved from train_model.py")
    parser.add_argument("out_json", help="output JSON path (ranked_nodes.json)")
    parser.add_argument("--threshold", type=float, default=0.6, help="core probability threshold (default 0.6)")
    args = parser.parse_args()
    main(args.features_csv, args.model_joblib, args.out_json, threshold=args.threshold)

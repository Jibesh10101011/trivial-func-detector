#!/usr/bin/env python3
"""
predict.py

Usage:
  python predict.py model.joblib new_analysis.json predictions.csv

Takes a trained model + new code analysis JSON,
runs the same feature extraction pipeline, and outputs predictions.
"""

import argparse
import joblib
import pandas as pd
from feature_extraction import extract_features_from_node, build_call_graph, compute_graph_features
import json

def main(model_path, input_json, output_csv):
    # Load model
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["features"]

    # Load new code data
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("analysisData", {}).get("graphNodes", [])

    # Extract features
    rows = []
    for node in nodes:
        feats = extract_features_from_node(node)
        feats["id"] = node.get("id")
        feats["label"] = node.get("label")
        feats["type"] = node.get("type")
        rows.append(feats)

    # Build call graph (for graph-based features)
    G = build_call_graph(rows)
    feature_map = {r["identifier"]: r for r in rows}
    graph_feats = compute_graph_features(G, feature_map)

    final_rows = []
    for r in rows:
        ident = r["identifier"]
        gf = graph_feats.get(ident, {"pagerank": 0.0, "betweenness": 0.0, "fan_in": 0, "fan_out": 0})
        combined = dict(r)
        combined.update(gf)
        combined["num_called_unique"] = len(set(r.get("calls", [])))
        combined.pop("calls", None)
        final_rows.append(combined)

    df = pd.DataFrame(final_rows)
    # Align columns with model training
    X = df.reindex(columns=feature_cols, fill_value=0)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df["prediction"] = preds
    df["prob_core"] = probs
    df["classification"] = df["prediction"].map({1: "core", 0: "utility"})

    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions written to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using trained model")
    parser.add_argument("model_path", help="Path to trained model.joblib")
    parser.add_argument("input_json", help="New analysis-with-code.json")
    parser.add_argument("output_csv", help="Output CSV with predictions")
    args = parser.parse_args()
    main(args.model_path, args.input_json, args.output_csv)

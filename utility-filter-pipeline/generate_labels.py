#!/usr/bin/env python3
"""
generate_labels.py
Automatically generate simple heuristic labels for training.
Usage:
  python generate_labels.py features.csv labeling.csv
"""
import pandas as pd

def main(features_csv, labeling_csv):
    df = pd.read_csv(features_csv)

    # Simple rule-based labeling:
    # Utility = has util/helper keywords in label or path
    df["label_manual"] = df.apply(
        lambda r: 0 if (
            isinstance(r["label"], str) and
            any(k in r["label"].lower() for k in ["util", "helper", "convert", "parse"])
        ) or (
            isinstance(r["id"], str) and
            any(k in r["id"].lower() for k in ["utils", "helpers"])
        ) else 1,
        axis=1
    )

    df[["id", "label_manual"]].to_csv(labeling_csv, index=False)
    print(f"Generated heuristic labeling.csv with {len(df)} labeled rows.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("features_csv")
    parser.add_argument("labeling_csv")
    args = parser.parse_args()
    main(args.features_csv, args.labeling_csv)

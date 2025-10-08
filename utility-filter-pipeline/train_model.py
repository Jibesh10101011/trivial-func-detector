#!/usr/bin/env python3
"""
train_model.py

Usage:
  python train_model.py features.csv labeling.csv model.joblib

labeling.csv should contain two columns:
- id : matching the 'id' field from features.csv
- label_manual : 1 for core, 0 for utility

This script joins the feature table with labels, trains a RandomForest,
and writes a model (.joblib). It also prints simple evaluation metrics.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
import joblib

def load_data(features_csv, labeling_csv):
    feats = pd.read_csv(features_csv)
    labels = pd.read_csv(labeling_csv)
    # Merge on id
    if "id" not in labels.columns:
        raise ValueError("labeling CSV must contain 'id' column matching features.csv")
    df = feats.merge(labels[["id", "label_manual"]], on="id", how="left")
    # Drop rows with missing label_manual
    labeled = df[df["label_manual"].notna()].copy()
    # Use unlabeled separately if needed
    unlabeled = df[df["label_manual"].isna()].copy()
    return labeled, unlabeled

def prepare_Xy(df):
    # Choose features to use (drop non-numeric columns)
    drop_cols = ["id", "label", "type", "identifier"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["label_manual"], errors='ignore')
    # Keep numeric only
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df["label_manual"].astype(int)
    return X, y

def main(features_csv, labeling_csv, model_out):
    labeled, unlabeled = load_data(features_csv, labeling_csv)
    if labeled.shape[0] < 10:
        raise ValueError("Not enough labeled samples. Please label at least ~30-50 nodes for reasonable training.")
    X, y = prepare_Xy(labeled)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("Classification report (test set):")
    print(classification_report(y_test, y_pred))
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.3f}")
    except Exception:
        pass

    # Save model and feature columns
    joblib.dump({"model": model, "features": X.columns.tolist()}, model_out)
    print(f"✅ Saved model to {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForest on labeled features")
    parser.add_argument("features_csv", help="features.csv output from feature_extraction.py")
    parser.add_argument("labeling_csv", help="CSV with columns: id,label_manual")
    parser.add_argument("model_out", help="model output path (joblib)")
    args = parser.parse_args()
    main(args.features_csv, args.labeling_csv, args.model_out)

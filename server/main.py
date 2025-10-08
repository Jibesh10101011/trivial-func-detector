from fastapi import FastAPI, UploadFile
import joblib
import pandas as pd
import json
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utility-filter-pipeline")))

from feature_extraction import extract_features_from_node, build_call_graph, compute_graph_features

app = FastAPI(title="Utility Classifier API")

bundle = joblib.load("model.joblib")
model = bundle["model"]
feature_cols = bundle["features"]

@app.post("/predict")
async def predict(file: UploadFile):
    data = json.load(file.file)
    nodes = data.get("analysisData", {}).get("graphNodes", [])
    rows = []
    for node in nodes:
        feats = extract_features_from_node(node)
        feats["id"] = node.get("id")
        feats["label"] = node.get("label")
        feats["type"] = node.get("type")
        rows.append(feats)

    G = build_call_graph(rows)
    feature_map = {r["identifier"]: r for r in rows}
    graph_feats = compute_graph_features(G, feature_map)

    for r in rows:
        ident = r["identifier"]
        gf = graph_feats.get(ident, {"pagerank": 0.0, "betweenness": 0.0, "fan_in": 0, "fan_out": 0})
        r.update(gf)

    df = pd.DataFrame(rows)
    X = df.reindex(columns=feature_cols, fill_value=0)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df["prediction"] = preds
    df["prob_core"] = probs
    df["classification"] = df["prediction"].map({1: "core", 0: "utility"})

    return df[["id", "label", "classification", "prob_core"]].to_dict(orient="records")

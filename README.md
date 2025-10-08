## Description
This project develops a system that automatically detects and filters **trivial utility functions** (like formatters, helpers, loggers, and simple wrappers) from large Python codebases — focusing the analysis on meaningful, business-critical logic.  

The solution uses a hybrid approach combining **static code analysis** and **machine learning** to improve classification accuracy and reduce false positives.

---

## Setup Instructions

1. **Install required libraries**
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. **Navigate to the main working directory**
    ```bash
    cd utility-filter-pipeline
    ```

---

## Model Development Workflow

### 1️⃣ Extract Features

Run static feature extraction from the input dataset:

```bash
python3 feature_extraction.py analysis-with-code.json output/features.csv
```

This step computes various static metrics like:  
`loc`, `cyclomatic_complexity`, `num_statements`, `num_returns`, `num_calls`, `fan_in`, `fan_out`, `pagerank`, `betweenness`, `constant_ratio`, `comment_ratio`, etc.

✅ **Output:** `output/features.csv`

---

### 2️⃣ Generate Labels

Label functions as `core` or `utility` manually/semi-automatically for training:

```bash
python3 generate_labels.py output/features.csv output/labeling.csv
```

✅ **Output:** `output/labeling.csv`

---

### 3️⃣ Train the ML Model

Train a **RandomForestClassifier** model using the generated labels:

```bash
python3 train_model.py output/features.csv output/labeling.csv model/model.joblib
```

The model automatically learns feature weights instead of relying on manual weight tuning.  
Achieved **~98% accuracy (ROC-AUC: 0.995)**.

<img width="662" height="264" alt="image" src="https://github.com/user-attachments/assets/bed5042c-fbd1-48a1-bfc1-33baaf5765ac" />


✅ **Output:** `model/model.joblib`

---

### 4️⃣ Analyze Dataset

Run analysis on the provided dataset and classify each function as *core* or *utility*:

```bash
python3 analyze.py analysis-with-code.json output/results.json
```

✅ **Output:** `output/results.json`  
Contains predicted labels, confidence scores, and importance metrics.

---

### 5️⃣ Rank Nodes by Importance

Rank all nodes by their business logic importance:

```bash
python3 score_all.py output/features.csv model/model.joblib output/ranked_nodes.json
```

✅ **Output:** `output/ranked_nodes.json`  
Ranks nodes by their **core relevance** while filtering trivial ones.

---

### 6️⃣ Test Model on New Dataset

Test the trained model with unseen data to evaluate generalization:

```bash
python3 predict.py model/model.joblib testing/new_analysis.json testing/output/predictions.csv
```

✅ **Output:** `testing/output/predictions.csv`

---

### 🧭 End-to-End Workflow Diagram

```
 ┌────────────────────────┐
 │ analysis-with-code.json│
 └────────────┬───────────┘
              │
              ▼
     ┌───────────────────────────┐
     │ feature_extraction.py     │
     │ → Extract static features │
     └────────────┬──────────────┘
                  │
                  ▼
     ┌───────────────────────────┐
     │ generate_labels.py        │
     │ → Create labeled dataset  │
     └────────────┬──────────────┘
                  │
                  ▼
     ┌───────────────────────────┐
     │ train_model.py            │
     │ → Train Random Forest     │
     └────────────┬──────────────┘
                  │
                  ▼
     ┌───────────────────────────┐
     │ analyze.py                │
     │ → Predict core/utility    │
     └────────────┬──────────────┘
                  │
                  ▼
     ┌───────────────────────────┐
     │ score_all.py              │
     │ → Rank function importance│
     └────────────┬──────────────┘
                  │
                  ▼
     ┌───────────────────────────┐
     │ predict.py                │
     │ → Test on new dataset     │
     └───────────────────────────┘
```

---

## Novelty of this Approach

Unlike purely heuristic or rule-based detection (which depends on fixed weights and thresholds), this approach:
- **Learns from data**: ML model infers which features indicate utility behavior.  
- **Reduces false positives**: Uses contextual graph and code structure signals.  
- **Combines static + semantic analysis**: Extracts both structural and linguistic cues from code.  
- **Scalable for new repositories**: Can easily be retrained with new labeled data.  

---

## 🏁 Summary

| Step | Script | Purpose | Output |
|------|---------|----------|---------|
| 1 | `feature_extraction.py` | Extracts static features | `features.csv` |
| 2 | `generate_labels.py` | Creates labeled dataset | `labeling.csv` |
| 3 | `train_model.py` | Trains RandomForest model | `model.joblib` |
| 4 | `analyze.py` | Predicts and classifies nodes | `results.json` |
| 5 | `score_all.py` | Ranks node importance | `ranked_nodes.json` |
| 6 | `predict.py` | Tests on new data | `predictions.csv` |

---

**Final Deliverable:**  
A fully functional service (via FastAPI) that accepts a JSON input of code nodes and outputs their classification (`core` vs `utility`) along with a confidence score and ranking.


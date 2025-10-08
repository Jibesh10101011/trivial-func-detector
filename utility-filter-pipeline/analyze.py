import ast
import re
import json
from radon.complexity import cc_visit

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

UTILITY_NAME_PATTERNS = re.compile(
    r"(util|helper|format|parse|convert|check|get|set|log|validate|to_|from_)",
    re.IGNORECASE
)

# Weight configuration for scoring
WEIGHTS = {
    "loc": 0.25,
    "complexity": 0.25,
    "name": 0.20,
    "string_ratio": 0.10,
    "comment_ratio": 0.10,
    "constant_ratio": 0.10
}

# ---------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------

def extract_features(code: str, label: str):
    """
    Extract static features from the given code snippet using AST and Radon.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return None  # skip unparsable code

    loc = len([line for line in code.splitlines() if line.strip()])
    complexity_data = cc_visit(code)
    complexity = sum(c.complexity for c in complexity_data) if complexity_data else 1

    # Count number of string constants and numeric constants
    all_nodes = list(ast.walk(tree))
    string_nodes = [n for n in all_nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    const_nodes = [n for n in all_nodes if isinstance(n, ast.Constant)]
    string_ratio = len(string_nodes) / max(1, len(all_nodes))
    constant_ratio = len(const_nodes) / max(1, len(all_nodes))

    # Comment ratio (approximation)
    comment_ratio = len([l for l in code.splitlines() if l.strip().startswith("#")]) / max(1, loc)

    # Name-based flag
    name_flag = 1 if UTILITY_NAME_PATTERNS.search(label) else 0

    return {
        "loc": loc,
        "complexity": complexity,
        "string_ratio": string_ratio,
        "constant_ratio": constant_ratio,
        "comment_ratio": comment_ratio,
        "name_flag": name_flag,
    }

# ---------------------------------------------------------------------
# SCORING LOGIC
# ---------------------------------------------------------------------

def compute_utility_score(features):
    """
    Combine normalized features into a single utility likelihood score (0–1).
    """
    loc = min(features["loc"] / 50, 1)
    complexity = min(features["complexity"] / 10, 1)

    score = (
        WEIGHTS["loc"] * (1 - loc)
        + WEIGHTS["complexity"] * (1 - complexity)
        + WEIGHTS["name"] * features["name_flag"]
        + WEIGHTS["string_ratio"] * features["string_ratio"]
        + WEIGHTS["comment_ratio"] * features["comment_ratio"]
        + WEIGHTS["constant_ratio"] * features["constant_ratio"]
    )
    return round(score, 3)

# ---------------------------------------------------------------------
# MAIN ANALYSIS PIPELINE
# ---------------------------------------------------------------------

def analyze_codebase(data):
    """
    Given the JSON input with graphNodes, compute utility scores and filter.
    """
    output = []
    nodes = data["analysisData"]["graphNodes"]

    for node in nodes:
        code = node.get("code", "")
        label = node.get("label", "")
        feats = extract_features(code, label)
        if not feats:
            continue

        utility_score = compute_utility_score(feats)
        core_importance = round(1 - utility_score, 3)
        classification = "utility" if utility_score >= 0.6 else "core"

        output.append({
            "id": node["id"],
            "label": label,
            "utility_score": utility_score,
            "core_importance": core_importance,
            "classification": classification,
            "features": feats
        })

    # Sort by importance descending
    output.sort(key=lambda x: x["core_importance"], reverse=True)
    return output

# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect and filter trivial utility functions.")
    parser.add_argument("input", help="Path to JSON file with analysisData.")
    parser.add_argument("output", help="Path to output JSON file with ranked results.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = analyze_codebase(data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Analysis complete. Results written to {args.output}")

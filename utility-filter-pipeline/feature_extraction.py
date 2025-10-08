#!/usr/bin/env python3
"""
feature_extraction.py

Usage:
  python feature_extraction.py analysis-with-code.json features.csv
"""
import json
import argparse
import ast
import re
from collections import Counter, defaultdict
from radon.complexity import cc_visit
import pandas as pd
import networkx as nx

from utils import safe_parse, get_called_name, normalize_label_to_identifier

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

UTILITY_NAME_PATTERNS = re.compile(
    r"(util|helper|format|parse|convert|check|get_|set_|to_|from_|to$|from$|validate|log)",
    re.IGNORECASE
)

PATH_UTIL_KEYWORDS = ["utils", "helpers", "schemas", "models", "contrib"]
PATH_CORE_KEYWORDS = ["routes", "endpoints", "handlers", "app", "router", "views"]

# ---------------------------------------------------------------------
# AST FEATURE EXTRACTION HELPERS
# ---------------------------------------------------------------------

def ast_node_counts(tree):
    counts = Counter()
    for n in ast.walk(tree):
        counts[type(n).__name__] += 1
    return counts


def max_nesting_depth(tree):
    max_depth = 0
    def visit(node, depth=0):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            visit(child, depth + 1)
    visit(tree, 0)
    return max_depth


def extract_features_from_node(node):
    """
    node: dict with keys id, label, code, type
    returns: dict of features
    """
    code = node.get("code", "") or ""
    label = node.get("label", "") or ""
    nid = node.get("id", "") or ""

    feats = {}
    lines = code.splitlines()
    loc = len([l for l in lines if l.strip()])
    feats["loc"] = loc

    tree = safe_parse(code)
    if tree is None:
        feats.update({
            "cyclomatic_complexity": 1.0,
            "num_statements": 0,
            "num_returns": 0,
            "num_assigns": 0,
            "num_calls": 0,
            "num_if": 0,
            "num_for": 0,
            "num_while": 0,
            "max_nesting_depth": 0,
            "string_ratio": 0.0,
            "constant_ratio": 0.0,
            "comment_ratio": 0.0,
            "name_flag": 1 if UTILITY_NAME_PATTERNS.search(label) else 0,
            "decorator_count": 0,
            "path_util_signal": 0,
            "path_core_signal": 0,
            "has_fastapi_decorator": 0,
            "calls": [],
        })
        return feats

    # Cyclomatic complexity
    try:
        cc = cc_visit(code)
        cc_sum = sum(c.complexity for c in cc) if cc else 1.0
    except Exception:
        cc_sum = 1.0
    feats["cyclomatic_complexity"] = float(cc_sum)

    # AST-level stats
    statements = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)]
    feats["num_statements"] = len(statements)
    feats["num_returns"] = len([n for n in ast.walk(tree) if isinstance(n, ast.Return)])
    feats["num_assigns"] = len([n for n in ast.walk(tree) if isinstance(n, ast.Assign)])
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    feats["num_calls"] = len(calls)
    feats["num_if"] = len([n for n in ast.walk(tree) if isinstance(n, ast.If)])
    feats["num_for"] = len([n for n in ast.walk(tree) if isinstance(n, ast.For)])
    feats["num_while"] = len([n for n in ast.walk(tree) if isinstance(n, ast.While)])
    feats["max_nesting_depth"] = max_nesting_depth(tree)

    # constants / string ratio
    all_nodes = list(ast.walk(tree))
    const_nodes = [n for n in all_nodes if isinstance(n, ast.Constant)]
    string_nodes = [n for n in const_nodes if isinstance(n.value, str)]
    feats["constant_ratio"] = len(const_nodes) / max(1, len(all_nodes))
    feats["string_ratio"] = len(string_nodes) / max(1, len(all_nodes))

    # comment ratio (approx)
    comment_lines = len([l for l in lines if l.strip().startswith("#")])
    feats["comment_ratio"] = comment_lines / max(1, loc)

    # decorators
    decorator_count = 0
    has_fastapi_decorator = 0
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if getattr(n, "decorator_list", None):
                decorator_count += len(n.decorator_list)
                for d in n.decorator_list:
                    try:
                        dstr = ast.unparse(d) if hasattr(ast, "unparse") else ""
                        if re.search(r"app\.|router\.|get|post|put|delete|patch", dstr):
                            has_fastapi_decorator = 1
                    except Exception:
                        pass
    feats["decorator_count"] = decorator_count
    feats["has_fastapi_decorator"] = has_fastapi_decorator

    # name & path heuristics
    feats["name_flag"] = 1 if UTILITY_NAME_PATTERNS.search(label) else 0
    lowered = nid.lower()
    feats["path_util_signal"] = 1 if any(k in lowered for k in PATH_UTIL_KEYWORDS) else 0
    feats["path_core_signal"] = 1 if any(k in lowered for k in PATH_CORE_KEYWORDS) else 0

    # request/response/dependency usage
    feats["uses_request"] = 1 if re.search(r"\bRequest\b", code) else 0
    feats["uses_response"] = 1 if re.search(r"\bResponse\b", code) else 0
    feats["uses_depends"] = 1 if re.search(r"\bDepends\b", code) else 0

    # collect call names
    call_names = []
    for c in calls:
        name = get_called_name(c)
        if name:
            call_names.append(name)
    feats["calls"] = call_names

    # ✅ Unique identifier
    feats["identifier"] = f"{nid}::{label}"

    return feats


# ---------------------------------------------------------------------
# GRAPH CONSTRUCTION + GRAPH METRICS
# ---------------------------------------------------------------------

def build_call_graph(feature_rows):
    """
    Build a best-effort directed call graph using call names + identifiers.
    """
    G = nx.DiGraph()
    for r in feature_rows:
        G.add_node(r.get("identifier", r.get("id", r.get("label"))), id_field=r.get("id"))

    # Map names to identifiers for matching
    name_index = defaultdict(list)
    for r in feature_rows:
        ident = r["identifier"]
        label = ident.split("::")[-1]
        name_index[label].append(ident)

    # Edges: caller -> callee
    for r in feature_rows:
        caller = r["identifier"]
        for called in r.get("calls", []):
            for target_ident in name_index.get(called, []):
                if target_ident != caller:
                    G.add_edge(caller, target_ident)
    return G


def compute_graph_features(G, feature_map):
    feats = {}
    if len(G) == 0:
        for ident in feature_map:
            feats[ident] = {"pagerank": 0.0, "betweenness": 0.0, "fan_in": 0, "fan_out": 0}
        return feats

    try:
        pagerank = nx.pagerank(G)
    except Exception:
        pagerank = {n: 0.0 for n in G.nodes()}

    try:
        betw = nx.betweenness_centrality(G)
    except Exception:
        betw = {n: 0.0 for n in G.nodes()}

    for n in G.nodes():
        feats[n] = {
            "pagerank": float(pagerank.get(n, 0.0)),
            "betweenness": float(betw.get(n, 0.0)),
            "fan_in": int(G.in_degree(n)),
            "fan_out": int(G.out_degree(n)),
        }
    return feats


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def main(input_json, output_csv):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("analysisData", {}).get("graphNodes", [])
    rows = []
    for node in nodes:
        feats = extract_features_from_node(node)
        feats["id"] = node.get("id")
        feats["label"] = node.get("label")
        feats["type"] = node.get("type")
        rows.append(feats)

    # ✅ Ensure identifier field exists for all rows
    for r in rows:
        if "identifier" not in r:
            r["identifier"] = f"{r['id']}::{r['label']}"

    # Build call graph and compute graph-based features
    G = build_call_graph(rows)
    feature_map = {r["identifier"]: r for r in rows}
    graph_feats = compute_graph_features(G, feature_map)

    # Merge
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
    df.to_csv(output_csv, index=False)
    print(f"✅ Wrote features to {output_csv}. Rows: {len(df)}")
    print(f"📊 Graph nodes: {len(G.nodes())}, edges: {len(G.edges())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from nodes JSON.")
    parser.add_argument("input", help="Path to analysis-with-code.json")
    parser.add_argument("output", help="Path to output features CSV")
    args = parser.parse_args()
    main(args.input, args.output)

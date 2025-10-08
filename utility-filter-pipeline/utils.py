# utils.py
import ast
import re
from typing import Optional

def safe_parse(code: str):
    try:
        return ast.parse(code)
    except Exception:
        return None

def get_called_name(node: ast.Call) -> Optional[str]:
    """
    Best-effort: extract a string name for a Call node: function name or attribute name.
    """
    func = node.func
    # direct function name: foo()
    if isinstance(func, ast.Name):
        return func.id
    # attribute: obj.method()
    if isinstance(func, ast.Attribute):
        # try full dotted name obj.method
        parts = []
        cur = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return ".".join(parts)
    return None

def normalize_label_to_identifier(id_str: str) -> str:
    """
    Convert the id field into module-like string for matching.
    Example id: "code:fastapi/applications.py:__call__:1129"
    -> returns "fastapi.applications.__call__"
    """
    try:
        # id format: code:<path>:<label>:<lineno>
        parts = id_str.split(":")
        if len(parts) >= 3:
            path = parts[1]
            label = parts[2]
            # replace path separators and .py
            mod = path.replace("/", ".").replace("\\", ".")
            if mod.endswith(".py"):
                mod = mod[:-3]
            return f"{mod}.{label}"
    except Exception:
        pass
    return id_str

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


class FormulaError(ValueError):
    pass


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)
_ALLOWED_FUNCS = {"abs": abs, "min": min, "max": max, "round": round}


def normalize_metric_name(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%]+", " ", name.lower())).strip()


def split_formula(expression: str) -> Tuple[str, str]:
    if "=" not in expression:
        raise FormulaError("Formula must contain '='")
    lhs, rhs = expression.split("=", 1)
    lhs, rhs = lhs.strip(), rhs.strip()
    if not lhs or not rhs:
        raise FormulaError("Invalid empty formula side")
    return lhs, rhs


def extract_dependency_phrases(expression: str) -> List[str]:
    """Best-effort extraction for transparent finance formulas."""
    _, rhs = split_formula(expression)
    rhs = rhs.replace("×", "*").replace("÷", "/").replace("−", "-")
    rhs = re.sub(r"\b(abs|min|max|round)\s*\(", "(", rhs, flags=re.I)
    chunks = re.split(r"[+\-*/^(),]", rhs)
    out: List[str] = []
    seen = set()
    for chunk in chunks:
        s = chunk.strip()
        s = re.sub(r"^[\d\s.,%]+|[\d\s.,%]+$", "", s).strip()
        if not s or re.fullmatch(r"[\d.]+", s):
            continue
        if s.lower() in {"and", "or", "the"}:
            continue
        key = normalize_metric_name(s)
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


@dataclass
class CompiledFormula:
    lhs: str
    rhs: str
    placeholder_by_metric: Dict[str, str]
    tree: ast.Expression


class SafeFormulaExecutor:
    def compile(self, expression: str, dependencies: Iterable[str]) -> CompiledFormula:
        lhs, rhs = split_formula(expression)
        normalized_rhs = rhs.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
        deps = [d for d in dependencies if d]
        placeholder_by_metric: Dict[str, str] = {}
        for idx, dep in enumerate(sorted(deps, key=len, reverse=True)):
            ph = f"v_{idx}"
            pattern = re.compile(re.escape(dep), flags=re.I)
            normalized_rhs, count = pattern.subn(ph, normalized_rhs)
            if count == 0:
                raise FormulaError(f"Dependency '{dep}' not found in expression '{expression}'")
            placeholder_by_metric[dep] = ph
        normalized_rhs = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%", r"(\1/100)", normalized_rhs)
        try:
            tree = ast.parse(normalized_rhs, mode="eval")
        except SyntaxError as e:
            raise FormulaError(f"Formula cannot be parsed: {normalized_rhs}") from e
        self._validate_ast(tree)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        allowed_names = set(placeholder_by_metric.values()) | set(_ALLOWED_FUNCS)
        unknown = names - allowed_names
        if unknown:
            raise FormulaError(f"Unknown symbols in formula: {sorted(unknown)}")
        return CompiledFormula(lhs=lhs, rhs=normalized_rhs, placeholder_by_metric=placeholder_by_metric, tree=tree)

    def can_compile(self, expression: str, dependencies: Iterable[str]) -> bool:
        try:
            self.compile(expression, dependencies)
            return True
        except FormulaError:
            return False

    def execute(self, expression: str, values: Dict[str, float], dependencies: Iterable[str]) -> float:
        compiled = self.compile(expression, dependencies)
        env = {compiled.placeholder_by_metric[k]: float(values[k]) for k in compiled.placeholder_by_metric}
        env.update(_ALLOWED_FUNCS)
        value = self._eval(compiled.tree.body, env)
        if not math.isfinite(value):
            raise FormulaError("Non-finite result")
        return float(value)

    def _validate_ast(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Expression, ast.Load, ast.Constant, ast.Name)):
                continue
            if isinstance(node, ast.BinOp):
                if not isinstance(node.op, _ALLOWED_BINOPS):
                    raise FormulaError(f"Operator not allowed: {type(node.op).__name__}")
                continue
            if isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, _ALLOWED_UNARY):
                    raise FormulaError("Unary operator not allowed")
                continue
            if isinstance(node, _ALLOWED_BINOPS + _ALLOWED_UNARY):
                continue
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                    raise FormulaError("Function call not allowed")
                continue
            raise FormulaError(f"AST node not allowed: {type(node).__name__}")

    def _eval(self, node: ast.AST, env: Dict[str, float]):
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise FormulaError("Only numeric constants are allowed")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise FormulaError(f"Unknown variable {node.id}")
            return env[node.id]
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand, env)
            return +v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            a = self._eval(node.left, env)
            b = self._eval(node.right, env)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div):
                if b == 0: raise FormulaError("Division by zero")
                return a / b
            if isinstance(node.op, ast.Pow): return a ** b
            if isinstance(node.op, ast.Mod): return a % b
        if isinstance(node, ast.Call):
            func = _ALLOWED_FUNCS[node.func.id]
            args = [self._eval(x, env) for x in node.args]
            return func(*args)
        raise FormulaError(f"Unsupported node: {type(node).__name__}")

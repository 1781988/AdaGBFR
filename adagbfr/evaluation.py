from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, Optional

from .schemas import Status

_UNANSWERABLE = {"unanswerable", "none", "null", "nan", "⊥", "cannot answer", "insufficient information"}


def is_unanswerable_ground_truth(value: Any) -> bool:
    if value is None: return True
    s = str(value).strip().lower()
    return s in _UNANSWERABLE or any(x in s for x in ("unanswerable", "cannot be answered", "insufficient evidence"))


def numeric_ground_truth(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)): return float(value)
    if value is None: return None
    s = str(value).strip(); percent = "%" in s
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", s)
    if not nums: return None
    out = float(nums[-1].replace(",", ""))
    return out / 100.0 if percent else out


def score_prediction(result: Dict[str, Any], ground_truth: Any,
                     rtol: float = 1e-3, atol: float = 1e-6) -> Dict[str, Any]:
    gt_unanswerable = is_unanswerable_ground_truth(ground_truth)
    pred_abstain = result.get("status") != Status.ANSWER.value
    if gt_unanswerable:
        return {"correct": pred_abstain, "abstention_correct": pred_abstain, "gt_unanswerable": True}
    gt = numeric_ground_truth(ground_truth); pred = result.get("value")
    if gt is None or pred is None:
        return {"correct": False, "abstention_correct": None, "gt_unanswerable": False}
    return {"correct": math.isclose(float(pred), float(gt), rel_tol=rtol, abs_tol=atol),
            "abstention_correct": None, "gt_unanswerable": False}


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records); n = len(records)
    correct = sum(bool(r.get("score", {}).get("correct")) for r in records)
    unanswerable = [r for r in records if r.get("score", {}).get("gt_unanswerable")]
    abst_correct = sum(bool(r.get("score", {}).get("abstention_correct")) for r in unanswerable)
    budgets = [r.get("result", {}).get("budget", {}) for r in records]
    return {"n":n,"accuracy":correct/n if n else 0.0,"unanswerable_n":len(unanswerable),
            "abstention_accuracy":abst_correct/len(unanswerable) if unanswerable else None,
            "avg_llm_calls":_avg(budgets,"llm_calls"),"avg_total_tokens":_avg(budgets,"total_tokens"),
            "avg_cost_usd":_avg(budgets,"estimated_cost_usd"),"avg_paths_considered":_avg(budgets,"paths_considered"),
            "avg_paths_executed":_avg(budgets,"paths_executed"),"dynamic_request_rate":_avg(budgets,"dynamic_requests"),
            "dynamic_accept_rate":_avg(budgets,"dynamic_accepts")}


def _avg(items, key):
    return sum(float(x.get(key, 0) or 0) for x in items) / len(items) if items else 0.0

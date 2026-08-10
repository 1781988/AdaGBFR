import json
from pathlib import Path

from adagbfr.config import load_config
from adagbfr.factory import build_pipeline


def _case(index: int):
    lines = Path("data/demo_questions.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[index])


def test_demo_dynamic_margin():
    cfg = load_config("configs/demo.yaml")
    case = _case(0)
    result = build_pipeline(cfg).run(case["query"], case["context"], plan_override=case["plan"])
    assert result.status.value == "answer"
    assert abs(result.value - 0.4) < 1e-12
    assert result.overlay_metrics


def test_demo_safe_abstention():
    cfg = load_config("configs/demo.yaml")
    case = _case(1)
    result = build_pipeline(cfg).run(case["query"], case["context"], plan_override=case["plan"])
    assert result.status.value == "evidence_absent"
    assert result.value is None

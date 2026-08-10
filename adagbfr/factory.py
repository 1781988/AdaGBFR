from __future__ import annotations

from typing import Any, Dict, Optional

from .budget import BudgetLimits, BudgetTracker
from .config import deep_get, env_value, resolve_path
from .dynamic_expansion import DynamicMetricExpander
from .evidence import HybridEvidenceExtractor
from .formula import SafeFormulaExecutor
from .knowledge import MetricStore
from .llm import OpenAICompatibleLLM
from .path_ranker import BudgetAwarePathRanker
from .pipeline import AdaGBFRPipeline
from .question import QuestionAnalyzer
from .resolver import AdaptiveResolver
from .source_retriever import LocalSourceIndex
from .verification import AdaptiveVerifier


def build_pipeline(cfg: Dict[str, Any], base_store: Optional[MetricStore] = None) -> AdaGBFRPipeline:
    kb_path = resolve_path(cfg, deep_get(cfg, "knowledge_base.path"))
    if base_store is None:
        if not kb_path: raise ValueError("knowledge_base.path is required")
        base_store = MetricStore.from_json(kb_path)
    store = base_store.new_session()

    bcfg = deep_get(cfg, "reasoning.budget", {}) or {}
    limits = BudgetLimits(
        max_llm_calls=int(bcfg.get("max_llm_calls", 24)), max_total_tokens=int(bcfg.get("max_total_tokens", 16000)),
        max_cost_usd=float(bcfg.get("max_cost_usd", 0.25)), max_paths_total=int(bcfg.get("max_paths_total", 12)),
        input_price_per_million=float(bcfg.get("input_price_per_million", 0.0)),
        output_price_per_million=float(bcfg.get("output_price_per_million", 0.0)))
    budget = BudgetTracker(limits)

    llm = None; lcfg = cfg.get("llm", {}) or {}
    if bool(lcfg.get("enabled", True)):
        key = env_value(lcfg.get("api_key_env", "OPENAI_API_KEY"), str(lcfg.get("api_key", "")))
        llm = OpenAICompatibleLLM(model=str(lcfg.get("model", "gpt-4.1-mini")), api_key=key,
            base_url=lcfg.get("base_url"), temperature=float(lcfg.get("temperature", 0.0)),
            timeout=float(lcfg.get("timeout", 120)), budget=budget)

    dcfg = deep_get(cfg, "reasoning.dynamic", {}) or {}; source_index = None
    source_path = resolve_path(cfg, dcfg.get("source_path"))
    if source_path: source_index = LocalSourceIndex.from_jsonl(source_path)

    executor = SafeFormulaExecutor()
    extractor = HybridEvidenceExtractor(llm=llm,
        use_llm_fallback=bool(deep_get(cfg, "reasoning.evidence.use_llm_fallback", True)))
    ranker = BudgetAwarePathRanker(store, deep_get(cfg, "reasoning.path", {}) or {})
    verifier = AdaptiveVerifier(llm=llm, budget=budget, config=deep_get(cfg, "reasoning.verification", {}) or {})
    expander = None
    if bool(dcfg.get("enabled", True)):
        expander = DynamicMetricExpander(store=store, source_index=source_index, executor=executor,
                                         llm=llm, budget=budget, config=dcfg)
    resolver = AdaptiveResolver(store=store, evidence_extractor=extractor, ranker=ranker,
        executor=executor, verifier=verifier, expander=expander, budget=budget,
        config=deep_get(cfg, "reasoning", {}) or {})
    return AdaGBFRPipeline(analyzer=QuestionAnalyzer(llm=llm), resolver=resolver, budget=budget, store=store)

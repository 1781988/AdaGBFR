from adagbfr.budget import BudgetLimits, BudgetTracker
from adagbfr.dynamic_expansion import DynamicMetricExpander
from adagbfr.formula import SafeFormulaExecutor
from adagbfr.knowledge import MetricStore
from adagbfr.schemas import MetricNode
from adagbfr.source_retriever import LocalSourceIndex, SourceDoc


def test_rule_dynamic_expansion():
    store = MetricStore([MetricNode(name="Gross Profit"), MetricNode(name="Revenue")]).new_session()
    index = LocalSourceIndex([
        SourceDoc("s1", "Gross Margin", "Formula: Gross Margin = Gross Profit / Revenue", authority=1.0)
    ])
    budget = BudgetTracker(BudgetLimits(max_llm_calls=0, max_total_tokens=1, max_cost_usd=0, max_paths_total=10))
    expander = DynamicMetricExpander(
        store=store,
        source_index=index,
        executor=SafeFormulaExecutor(),
        llm=None,
        budget=budget,
        config={"min_confidence": 0.8, "min_sources": 1, "use_llm_fallback": False},
    )
    metric = expander.expand("Gross Margin")
    assert metric is not None
    assert metric.origin == "dynamic_rule"
    assert metric.formulas[0].dependencies == ["Gross Profit", "Revenue"]
    assert store.get("Gross Margin") is not None

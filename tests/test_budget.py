import pytest

from adagbfr.budget import BudgetExceeded, BudgetLimits, BudgetTracker


def test_path_budget():
    b = BudgetTracker(BudgetLimits(max_paths_total=1))
    b.record_path_considered()
    with pytest.raises(BudgetExceeded):
        b.record_path_considered()

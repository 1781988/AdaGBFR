from adagbfr.formula import SafeFormulaExecutor, extract_dependency_phrases

def test_formula_execution():
    ex=SafeFormulaExecutor(); value=ex.execute("Gross Margin = Gross Profit / Revenue",{"Gross Profit":400.0,"Revenue":1000.0},["Gross Profit","Revenue"])
    assert abs(value-0.4)<1e-12

def test_dependency_extraction():
    assert extract_dependency_phrases("Average Equity = (Beginning Equity + Ending Equity) / 2")==["Beginning Equity","Ending Equity"]

"""
Test suite for chart_tool — runs 4 LLM-driven scenarios + 1 direct unit test.
"""
import sys
import io
import os
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from graph import graph
from tools import chart_tool

# ── Helpers ──────────────────────────────────────────────────────────────────

def run(query: str, df: pd.DataFrame, label: str):
    print(f"\n{'='*60}")
    print(f"  TEST: {label}")
    print(f"  QUERY: {query}")
    print(f"{'='*60}")
    state = {
        "user_query": query,
        "dataframe": df.copy(),
        "plan": [],
        "current_step_index": 0,
        "result": None,
        "error": "",
    }
    final = graph.invoke(state)
    if final.get("error"):
        print(f"  ❌ Error: {final['error']}")
    else:
        print("  📋 Plan:")
        for i, step in enumerate(final.get("plan", []), 1):
            print(f"     Step {i}: {step['tool_name']} → {step['tool_args']}")
        print(f"  ✅ Result: {final.get('result')}")


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading sales.xlsx …")
df = pd.read_excel("sales.xlsx")
print(f"Loaded {len(df):,} rows × {len(df.columns)} columns\n")
print("Columns:", list(df.columns))

# ── Test 1: Bar chart — total sales per city (uses built-in agg) ──────────────
run(
    query="draw a bar chart of total sales per city",
    df=df,
    label="Bar chart — Total by City (agg built-in)",
)

# ── Test 2: Pie chart — sales share by category ───────────────────────────────
run(
    query="show a pie chart of sales distribution by category",
    df=df,
    label="Pie chart — Sales by Category",
)

# ── Test 3: Multi-step — filter then line chart ───────────────────────────────
run(
    query="filter only Electronics category, then plot a bar chart of total sales per city",
    df=df,
    label="Filter → Bar chart (Electronics per City)",
)

# ── Test 4: Grouped bar — sales per category coloured by city ─────────────────
run(
    query="create a grouped bar chart showing total sales for each category, grouped by city",
    df=df,
    label="Grouped Bar — Category × City",
)

# ── Test 5: Direct unit test (no LLM) ────────────────────────────────────────
print(f"\n{'='*60}")
print("  TEST: Direct unit test — chart_tool(scatter)")
print(f"{'='*60}")
sample = df.sample(500, random_state=42)
result = chart_tool(
    df=sample,
    chart_type="scatter",
    x="Quantity",
    y="Total",
    color_by="Category",
    title="Quantity vs Total (sample 500 rows)",
    output_path="charts/test_scatter.html",
)
exists = os.path.exists("charts/test_scatter.html")
print(f"  ✅ {result}")
print(f"  File exists on disk: {exists}")

print("\n\nAll tests done. Charts saved to ./charts/")

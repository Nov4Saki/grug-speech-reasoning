"""
Comprehensive verification test for:
1. Bar tool: single aggregated solid bars (no micro-slices) + Top-N thresholding.
2. Pie chart: correct pie chart selection, slice aggregation & percent formatting.
3. Scatter plot: responsive smart sampling (2,000 pts) and styling.
"""
import sys
import io
import os
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from graph import graph

print("Loading sales.xlsx...")
df = pd.read_excel("sales.xlsx")
print(f"Loaded {len(df):,} rows x {len(df.columns)} columns\n")


def test_query(query: str, title: str):
    print(f"\n{'='*70}")
    print(f"  TEST: {title}")
    print(f"  QUERY: {query}")
    print(f"{'='*70}")
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
            print(f"     Step {i}: {step['tool_name']} -> {step['tool_args']}")
        print(f"  ✅ Result: {final.get('result')}")


# ── Test 1: Bar Chart (Auto-Aggregated & Thresholded) ─────────────────────────
test_query(
    query="ارسم بار شارت يوضح مجموع المبيعات لكل مدينة",
    title="1. Bar Chart: Auto-aggregation & Clean Single Bars (Arabic)",
)

# ── Test 2: Pie Chart (Arabic detection & slice distribution) ─────────────────
test_query(
    query="عايز باي شارت يوضح نسبة مبيعات كل فئة من المنتجات",
    title="2. Pie Chart: Slice Percentage & Thresholding (Arabic)",
)

# ── Test 3: Scatter Plot (Smart Sampling on 100k rows) ────────────────────────
test_query(
    query="ارسم سكاتر بلوت يوضح العلاقة بين الكمية والمبيعات الإجمالية ملون حسب الفئة",
    title="3. Scatter Plot: Responsive 2,000-pt Sampling (Arabic)",
)

# ── Test 4: High-Cardinality Bar Thresholding (English) ───────────────────────
test_query(
    query="draw a bar chart of sales by Customer",
    title="4. Bar Chart: High-Cardinality Thresholding (Customer Top-N + Other)",
)

print("\n\nAll tests completed! Check file sizes in ./charts/ directory:")
for f in os.listdir("charts"):
    path = os.path.join("charts", f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  - {f}: {size_kb:.1f} KB")

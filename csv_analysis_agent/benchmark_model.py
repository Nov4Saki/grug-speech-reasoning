import sys
import io
import time
import json
import pandas as pd
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from graph import graph

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

MODEL_NAME = "gemma-4-e2b-grugspeech-native"

print(f"================================================================")
print(f"  BENCHMARKING MODEL: {MODEL_NAME}")
print(f"================================================================\n")

# 1. Check available models in LM Studio
try:
    models_response = client.models.list()
    available_ids = [m.id for m in models_response.data]
    print(f"Connected to LM Studio successfully.")
    print(f"Available loaded models in LM Studio: {available_ids}")
    if MODEL_NAME not in available_ids:
        print(f"⚠️ Warning: '{MODEL_NAME}' is not explicitly listed in available models, but LM Studio often routes to the currently loaded model.")
except Exception as e:
    print(f"⚠️ Could not query models list: {e}")

# 2. Load dataset
print("\nLoading sales.xlsx...")
t0 = time.perf_counter()
df = pd.read_excel("sales.xlsx")
load_time = time.perf_counter() - t0
print(f"Loaded {len(df):,} rows x {len(df.columns)} columns in {load_time:.2f}s\n")

# 3. Test Cases
test_cases = [
    {
        "id": "T1_Simple_EN",
        "category": "Basic Aggregation (EN)",
        "query": "What is the total sales amount?",
        "expected_tool": "aggregate_tool"
    },
    {
        "id": "T2_MultiStep_EN",
        "category": "Multi-step Filter + Grouping (EN)",
        "query": "Filter rows where Total is greater than 500, then group by City and show the sum of Total for each city, sorted from highest to lowest",
        "expected_tool": "filter_tool"
    },
    {
        "id": "T3_Simple_AR",
        "category": "Basic Aggregation (AR)",
        "query": "احسب متوسط السعر لكل فئة من المنتجات",
        "expected_tool": "aggregate_tool"
    },
    {
        "id": "T4_MultiStep_AR",
        "category": "Multi-step Filter + Grouping (AR)",
        "query": "فلتر الصفوف اللي فيها الكمية أكبر من 5، وبعدين احسب مجموع المبيعات لكل فئة، وترتيب النتيجة من الأعلى للأقل",
        "expected_tool": "filter_tool"
    },
    {
        "id": "T5_Chart_Pie_AR",
        "category": "Pie Chart (AR)",
        "query": "عايز باي شارت يوضح نسبة مبيعات كل فئة من المنتجات",
        "expected_tool": "chart_tool"
    },
    {
        "id": "T6_Chart_GroupedBar_EN",
        "category": "Grouped Bar Chart (EN)",
        "query": "create a grouped bar chart showing total sales for each category, grouped by city",
        "expected_tool": "chart_tool"
    }
]

results = []

for idx, tc in enumerate(test_cases, 1):
    print(f"\n----------------------------------------------------------------")
    print(f"[{idx}/{len(test_cases)}] Running: {tc['category']}")
    print(f"Query: {tc['query']}")
    print(f"----------------------------------------------------------------")

    state = {
        "user_query": tc["query"],
        "dataframe": df.copy(),
        "plan": [],
        "current_step_index": 0,
        "result": None,
        "error": "",
    }

    start_time = time.perf_counter()
    try:
        final_state = graph.invoke(state)
        elapsed = time.perf_counter() - start_time
        
        has_error = bool(final_state.get("error"))
        plan = final_state.get("plan", [])
        result = final_state.get("result")
        
        first_tool = plan[0].get("tool_name") if plan else None
        
        # Validation checks
        json_valid = len(plan) > 0 and not has_error
        plan_sensible = json_valid and (first_tool == tc["expected_tool"] or any(s.get("tool_name") == tc["expected_tool"] for s in plan))
        
        status = "PASSED" if (json_valid and plan_sensible and not has_error) else "FAILED"
        
        print(f"  ⏱️ Time Taken: {elapsed:.2f}s")
        print(f"  📋 Steps ({len(plan)}):")
        for step_num, step in enumerate(plan, 1):
            print(f"     Step {step_num}: {step.get('tool_name')} -> {step.get('tool_args')}")
        
        if has_error:
            print(f"  ❌ Error: {final_state['error']}")
        else:
            res_str = str(result)
            preview = res_str[:120] + "..." if len(res_str) > 120 else res_str
            print(f"  ✅ Result preview: {preview}")
            
        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "elapsed_s": elapsed,
            "status": status,
            "steps_count": len(plan),
            "first_tool": first_tool,
            "error": final_state.get("error", "")
        })

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        print(f"  💥 Exception occurred: {exc}")
        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "elapsed_s": elapsed,
            "status": "EXCEPTION",
            "steps_count": 0,
            "first_tool": None,
            "error": str(exc)
        })

# 4. Summary Report
print(f"\n================================================================")
print(f"  BENCHMARK SUMMARY FOR: {MODEL_NAME}")
print(f"================================================================")

total_tests = len(results)
passed_tests = sum(1 for r in results if r["status"] == "PASSED")
avg_time = sum(r["elapsed_s"] for r in results) / total_tests if total_tests else 0
min_time = min(r["elapsed_s"] for r in results) if results else 0
max_time = max(r["elapsed_s"] for r in results) if results else 0

print(f"Total Tests Run   : {total_tests}")
print(f"Passed Tests      : {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
print(f"Avg Response Time : {avg_time:.2f}s")
print(f"Min Response Time : {min_time:.2f}s")
print(f"Max Response Time : {max_time:.2f}s\n")

print(f"{'Test ID':<22} | {'Category':<32} | {'Time (s)':<9} | {'Status'}")
print(f"{'-'*22}-|-{'-'*32}-|-{'-'*9}-|-{'-'*10}")
for r in results:
    print(f"{r['id']:<22} | {r['category']:<32} | {r['elapsed_s']:<9.2f} | {r['status']}")

print(f"\nBenchmark completed.")

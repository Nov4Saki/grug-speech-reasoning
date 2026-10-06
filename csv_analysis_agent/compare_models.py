import sys
import io
import time
import json
import pandas as pd
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import nodes
from graph import graph

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Candidate models of similar 2B size available in LM Studio
MODELS_TO_TEST = [
    "qwen3.5-2b-grugspeech",
    "qwen3.5-2b-claude-4.6-opus-reasoning-distilled",
    "minicpm5-2b"
]

print("Loading sales.xlsx...")
t0 = time.perf_counter()
df = pd.read_excel("sales.xlsx")
print(f"Loaded {len(df):,} rows in {time.perf_counter() - t0:.2f}s\n")

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

# We will dynamically override the planner_node model call
original_planner = nodes.planner_node

all_model_results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n================================================================")
    print(f"  TESTING MODEL: {model_name}")
    print(f"================================================================")

    # Custom planner wrapper for specific model
    def make_planner(target_model):
        def custom_planner(state):
            user_query = nodes._sanitize(state["user_query"])
            current_df = state["dataframe"]
            columns = list(current_df.columns)
            raw_sample = json.dumps(current_df.head(2).to_dict(orient="records"), default=str)
            sample_data = nodes._sanitize(raw_sample)

            prompt = f"""You are an expert data analysis planner. Break down the user's request into a sequential execution plan of tools.

Dataframe Columns: {columns}
Sample Rows: {sample_data}

Available Tools:
{nodes.tool_descriptions}

Rules:
- Match user intent (Arabic/English) to the appropriate tools.
- If request requires multiple actions (e.g. filter then calculate, or calculate per group), output ordered steps in the "steps" list.
- Use EXACT column names from the dataframe.
- NEVER translate or invent dataframe values. Use the exact values that exist in the dataframe.
- If the user asks to replace one specific existing value with another value, use transform_tool with action "replace".
- For "replace", replacements must be a dictionary in the form {{"old_value": "new_value"}}.
- If the user asks to filter rows and then change ALL values of a column in those filtered rows to one new value, use TWO steps:
  1. filter_tool
  2. transform_tool with action "set_value"
- For "set_value", use:
  {{"action": "set_value", "column": "exact column name", "value": "new value"}}
- If the user says "products with price greater than X", ALWAYS use the "Price" column.
- If the user says "change the product/product name", use the "Product" column.
- If the user says "change the city", use the "City" column.
- If the user says "change the category", use the "Category" column.
- Do NOT use "*" as an old value in replacements.
- Do NOT use the new value as the old replacement key.
- For charts/visualizations, ALWAYS use "chart_tool":
  * If user asks for pie chart / باي شارت / مخطط دائري / نسبة مئوية / distribution / حصة: use chart_type "pie", x="dimension column" (e.g. Category/City), y="metric column" (e.g. Total/Price).
  * If user asks for bar chart / بار شارت / أعمدة / رسم بياني: use chart_type "bar", x="dimension column", y="metric column", and optional color_by for grouped bars.
  * If user asks for scatter plot / سكاتر / انتشار / علاقة بين متغيرين: use chart_type "scatter", x="metric1", y="metric2", and optional color_by.
  * If user asks for line chart / خطي / تريند: use chart_type "line", x="date or dimension", y="metric".
- Output ONLY a valid JSON object matching this schema:
{{
    "steps": [
        {{
            "tool_name": "exact_tool_name",
            "tool_args": {{ "arg_name": "arg_value" }}
        }}
    ]
}}

User Query: {user_query}"""

            response = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return {"error": f"Failed to extract valid JSON plan from {target_model}."}

            try:
                plan_data = json.loads(match.group(0))
                steps = plan_data.get("steps", [])
                if not steps:
                    return {"error": "Model returned an empty execution plan."}
                return {"plan": steps, "current_step_index": 0, "error": ""}
            except Exception as e:
                return {"error": f"JSON parsing failed: {e}"}
        return custom_planner

    # Rebuild graph dynamically
    from langgraph.graph import StateGraph, START, END
    from state import GraphState
    from graph import should_continue

    builder = StateGraph(GraphState)
    builder.add_node("planner", make_planner(model_name))
    builder.add_node("executor", nodes.executor_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges("executor", should_continue, ["executor", END])
    model_graph = builder.compile()

    model_results = []

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}/6] {tc['category']} -> '{tc['query'][:45]}...'")
        state = {
            "user_query": tc["query"],
            "dataframe": df.copy(),
            "plan": [],
            "current_step_index": 0,
            "result": None,
            "error": "",
        }
        t_start = time.perf_counter()
        try:
            final_state = model_graph.invoke(state)
            t_elapsed = time.perf_counter() - t_start
            plan = final_state.get("plan", [])
            has_error = bool(final_state.get("error"))
            first_tool = plan[0].get("tool_name") if plan else None

            json_valid = len(plan) > 0 and not has_error
            status = "PASSED" if (json_valid and not has_error) else "FAILED"

            print(f"  ⏱️ Time: {t_elapsed:.2f}s | Status: {status}")
            print(f"  📋 Steps ({len(plan)}):")
            for s_idx, s in enumerate(plan, 1):
                print(f"     Step {s_idx}: {s.get('tool_name')} -> {s.get('tool_args')}")
            if has_error:
                print(f"  ❌ Error: {final_state['error']}")
            else:
                res_str = str(final_state.get("result"))
                print(f"  ✅ Result: {res_str[:70]}...")

            model_results.append({
                "id": tc["id"],
                "elapsed_s": t_elapsed,
                "status": status,
                "error": final_state.get("error", "")
            })
        except Exception as exc:
            t_elapsed = time.perf_counter() - t_start
            print(f"  💥 Exception: {exc}")
            model_results.append({
                "id": tc["id"],
                "elapsed_s": t_elapsed,
                "status": "EXCEPTION",
                "error": str(exc)
            })

    all_model_results[model_name] = model_results

# Comparison Summary
print(f"\n================================================================")
print(f"  CROSS-MODEL COMPARISON (SIMILAR SIZE ~2B)")
print(f"================================================================")

# Add reference from gemma benchmark
print(f"{'Model Name':<45} | {'Pass Rate':<10} | {'Avg Time':<10} | {'Multi-step OK?'}")
print(f"{'-'*45}-|-{'-'*10}-|-{'-'*10}-|-{'-'*15}")
print(f"{'gemma-4-e2b-grugspeech-native (reference)':<45} | 4/6 (66.7%) | 5.11s      | Failed (col naming)")

for m_name, res in all_model_results.items():
    passed = sum(1 for r in res if r["status"] == "PASSED")
    total = len(res)
    avg_t = sum(r["elapsed_s"] for r in res) / total if total else 0
    t2_pass = next((r["status"] for r in res if r["id"] == "T2_MultiStep_EN"), "FAILED")
    t4_pass = next((r["status"] for r in res if r["id"] == "T4_MultiStep_AR"), "FAILED")
    multi_ok = "YES" if (t2_pass == "PASSED" and t4_pass == "PASSED") else "NO"
    print(f"{m_name:<45} | {passed}/{total} ({passed/total*100:.1f}%) | {avg_t:.2f}s      | {multi_ok}")

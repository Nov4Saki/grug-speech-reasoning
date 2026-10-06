import os
import sys
import io
import time
import re
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM, BitsAndBytesConfig

# Ensure utf-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add tools directory to path
PROJECT_DIR = "/content/final_project/final project"
sys.path.append(PROJECT_DIR)
from tools import aggregate_tool, filter_tool, view_tool, transform_tool, chart_tool

TOKEN = "HF_TOKEN_REDACTED"
MODEL_ID = "google/gemma-4-12B-it"
RESULTS_FILE_LOCAL = os.path.join(PROJECT_DIR, "results_gemma-4-12b.json")
RESULTS_FILE_EVAL = "/content/grug-speech-reasoning/evaluation/results_gemma-4-12b_csv_agent.json"
os.makedirs(os.path.dirname(RESULTS_FILE_EVAL), exist_ok=True)

TOOLS = {
    "aggregate_tool": aggregate_tool,
    "filter_tool": filter_tool,
    "view_tool": view_tool,
    "transform_tool": transform_tool,
    "chart_tool": chart_tool
}

BENCHMARK_PROMPTS = [
    {
        "id": "Q1_AR_MultiStep",
        "name": "Arabic Multi-step (Filter + Group + Sort)",
        "query": "فلتر الصفوف اللي فيها الكمية أكبر من 5، وبعدين احسب مجموع المبيعات لكل فئة، وترتيب النتيجة من الأعلى للأقل"
    },
    {
        "id": "Q2_EN_Chart_Grouped",
        "name": "English Visualization (Grouped Bar Chart)",
        "query": "create a grouped bar chart showing total sales for each category, grouped by city"
    },
    {
        "id": "Q3_EN_MultiStep",
        "name": "English Multi-step (Filter + Group + Sort)",
        "query": "Filter rows where Total is greater than 500, then group by City and show the sum of Total for each city, sorted from highest to lowest"
    }
]

BASE_SCHEMA_CONTEXT = """
Dataframe Columns: ['Order_ID', 'Customer', 'Product', 'Category', 'Price', 'Quantity', 'Discount_Percent', 'Total', 'City', 'Payment_Method', 'Status', 'Order_Date']
Sample Rows: [{"Order_ID": 1001, "Customer": "Ahmed", "Product": "Laptop", "Category": "Electronics", "Price": 1200, "Quantity": 2, "Discount_Percent": 0.1, "Total": 2160, "City": "Cairo", "Payment_Method": "Credit Card", "Status": "Completed", "Order_Date": "2024-01-15"}]

Available Tools:
- aggregate_tool: Performs stats (sum, mean, min, max, count, median). Supports by_column for group by. Note: output metric column keeps its original name (e.g. Total).
  Arguments: column (string), operation (string: sum, mean, min, max, count, median), by_column (optional string), n (optional int)
- filter_tool: Filters rows using pandas query string (e.g. `Quantity` > 5).
  Arguments: query_string (string)
- view_tool: Sorts data (sort_by, ascending) or views top N rows (n).
  Arguments: sort_by (optional string), ascending (optional boolean, default True), n (optional int)
- chart_tool: Generates interactive Plotly charts (bar, line, pie, scatter).
  Arguments: chart_type (string: bar, line, pie, scatter), x (string), y (string), title (optional string), color_by (optional string), agg (optional string: sum, mean, count)
"""

STANDARD_PROMPT_TEMPLATE = """You are an expert data analysis planner. Break down the user's request into a sequential execution plan of tools.
""" + BASE_SCHEMA_CONTEXT + """
Output ONLY a valid JSON object matching this schema:
{
    "steps": [
        {
            "tool_name": "exact_tool_name",
            "tool_args": { "arg_name": "arg_value" }
        }
    ]
}

User Query: """

GRUG_PROMPT_TEMPLATE = """You are a data analysis planner. Reason in Grug Speech inside <think> tags.
""" + BASE_SCHEMA_CONTEXT + """
Output format:
<think>
Domain: tool_use
Goal: [State target]
Invariants: [Conserved properties: columns, types, filters]
Action: [Step 1 -> Step 2 -> Step 3]
Done.
</think>
{
    "steps": [
        {
            "tool_name": "exact_tool_name",
            "tool_args": { "arg_name": "arg_value" }
        }
    ]
}

User Query: """

def execute_plan(df, plan_json):
    """Executes the tool plan steps on the actual DataFrame."""
    current_df = df.copy()
    step_results = []
    
    steps = plan_json.get("steps", [])
    if not steps:
        return False, "No steps in plan"
        
    for i, step in enumerate(steps):
        tool_name = step.get("tool_name")
        tool_args = step.get("tool_args", {})
        
        if tool_name not in TOOLS:
            return False, f"Unknown tool: {tool_name}"
            
        tool_fn = TOOLS[tool_name]
        try:
            # Call tool with current df
            result = tool_fn(current_df, **tool_args)
            if isinstance(result, pd.DataFrame):
                current_df = result
            step_results.append({
                "step": i + 1,
                "tool": tool_name,
                "args": tool_args,
                "status": "success"
            })
        except Exception as e:
            return False, f"Step {i+1} ({tool_name}) error: {str(e)}"
            
    return True, step_results

def main():
    print("="*80)
    print("  GEMMA 4 12B BENCHMARK ON CSV DATA ANALYSIS AGENT APP")
    print("  Testing Normal (Standard Plan) vs Grugified (Grug Speech Scaffold)")
    print("="*80)

    # 1. Load Data
    excel_path = os.path.join(PROJECT_DIR, "sales.xlsx")
    print(f"Loading sales data from {excel_path}...")
    df = pd.read_excel(excel_path)
    print(f"Sales data loaded! Shape: {df.shape}")

    # 2. Load Model in 4-bit NF4
    print("\nLoading google/gemma-4-12B-it in 4-bit NF4...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=TOKEN
        )
        print("Loaded with AutoModelForImageTextToText!")
    except Exception as e:
        print(f"AutoModelForImageTextToText fallback: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=TOKEN
        )
        print("Loaded with AutoModelForCausalLM!")

    print(f"Model loaded! Allocated VRAM: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")

    evaluation_report = {
        "model": "google/gemma-4-12B-it",
        "benchmark_date": "2026-10-06",
        "modes": {}
    }

    # Test both modes: Normal vs Grugified
    for mode_name, is_grug, template in [
        ("Normal (Direct JSON)", False, STANDARD_PROMPT_TEMPLATE),
        ("Grugified (Grug Speech)", True, GRUG_PROMPT_TEMPLATE)
    ]:
        print(f"\n{'#'*80}")
        print(f"  TESTING MODE: {mode_name}")
        print(f"{'#'*80}")

        mode_results = []

        for q_item in BENCHMARK_PROMPTS:
            q_id = q_item["id"]
            q_name = q_item["name"]
            query = q_item["query"]
            print(f"\n--- [{q_id}] {q_name} ---")
            print(f"Query: \"{query}\"")

            full_prompt = template + query
            inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
            input_len = inputs["input_ids"].shape[1]

            t0 = time.time()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            t1 = time.time()
            total_duration = t1 - t0

            gen_ids = out_ids[0][input_len:]
            raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            num_tokens = len(gen_ids)
            tps = num_tokens / total_duration if total_duration > 0 else 0.0

            # Extract thinking trace
            think_trace = ""
            if "<think>" in raw_output and "</think>" in raw_output:
                think_trace = raw_output.split("<think>")[1].split("</think>")[0].strip()

            # Extract JSON plan
            plan_json = None
            valid_json = False
            exec_success = False
            exec_details = ""

            json_match = re.search(r'\{[\s\S]*"steps"[\s\S]*\}', raw_output)
            if json_match:
                try:
                    plan_json = json.loads(json_match.group(0))
                    valid_json = True
                except Exception as e:
                    exec_details = f"JSON parse error: {e}"
            else:
                exec_details = "No JSON steps found in generation"

            # Execute plan if valid
            if valid_json and plan_json:
                exec_success, exec_details = execute_plan(df, plan_json)

            print(f"Total Time: {total_duration:.2f}s | Tokens: {num_tokens} | TPS: {tps:.1f} tok/s")
            print(f"Valid Plan: {valid_json} | Executed on Data: {exec_success}")
            if think_trace:
                print(f"Thinking Trace ({len(think_trace.split())} words):\n{think_trace}")
            if plan_json:
                print(f"Generated Steps: {len(plan_json.get('steps', []))}")
                for s in plan_json.get("steps", []):
                    print(f"  -> {s.get('tool_name')}: {s.get('tool_args')}")

            mode_results.append({
                "id": q_id,
                "name": q_name,
                "query": query,
                "total_time_sec": round(total_duration, 2),
                "tokens": num_tokens,
                "tps": round(tps, 1),
                "valid_plan": valid_json,
                "executed_on_data": exec_success,
                "think_trace": think_trace,
                "plan": plan_json,
                "execution_details": str(exec_details),
                "raw_output": raw_output
            })

        evaluation_report["modes"][mode_name] = {
            "results": mode_results,
            "pass_rate_pct": round(sum(1 for r in mode_results if r["executed_on_data"]) / len(mode_results) * 100, 1),
            "avg_latency_sec": round(sum(r["total_time_sec"] for r in mode_results) / len(mode_results), 2),
            "avg_tokens": round(sum(r["tokens"] for r in mode_results) / len(mode_results), 1)
        }

    # Save to disk
    with open(RESULTS_FILE_LOCAL, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

    with open(RESULTS_FILE_EVAL, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

    print(f"\n================================================================================")
    print(f"EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"Saved local report to: {RESULTS_FILE_LOCAL}")
    print(f"Saved evaluation report to: {RESULTS_FILE_EVAL}")
    print(f"================================================================================")

    # Master Comparative Summary
    print("\n" + "="*80)
    print(f"{'Condition':<30} | {'Pass Rate':<12} | {'Avg Tokens':<12} | {'Avg Latency':<12}")
    print("="*80)
    for mode, data in evaluation_report["modes"].items():
        print(f"{mode:<30} | {data['pass_rate_pct']:>10.1f}% | {data['avg_tokens']:>10.1f} | {data['avg_latency_sec']:>10.2f}s")
    print("="*80)

if __name__ == "__main__":
    main()

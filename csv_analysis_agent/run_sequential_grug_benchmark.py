import sys
import io
import time
import re
import json
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Models to test one at a time (Gemma grugspeech vs non-Qwen peers)
MODELS_TO_TEST = [
    "gemma-4-e2b-grugspeech-native",
    "minicpm5-2b",
    "nanbeige4.1-3b@q8_0",
    "ternary-bonsai-1.7b"
]

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
- filter_tool: Filters rows using query string.
- view_tool: Sorts data (sort_by, ascending) or views top N rows.
- chart_tool: Generates interactive Plotly charts (bar, line, pie, scatter).
"""

GRUG_PROMPT_TEMPLATE = """You are a data analysis planner. Reason in Grug Speech inside <think> tags.
""" + BASE_SCHEMA_CONTEXT + """
Output format:
<think>
(Reason in Grug Speech here)
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

all_metrics = {}

for model_idx, model_name in enumerate(MODELS_TO_TEST, 1):
    is_grug = "grug" in model_name.lower()
    prompt_type = "GRUG SPEECH (<think>)" if is_grug else "STANDARD (Direct JSON)"
    
    print(f"\n{'#'*80}")
    print(f"  [{model_idx}/{len(MODELS_TO_TEST)}] TESTING ISOLATED MODEL: {model_name}")
    print(f"  MODE: {prompt_type}")
    print(f"{'#'*80}\n")
    
    model_records = []
    
    for q_idx, q_item in enumerate(BENCHMARK_PROMPTS, 1):
        template = GRUG_PROMPT_TEMPLATE if is_grug else STANDARD_PROMPT_TEMPLATE
        full_prompt = template + q_item["query"]
        
        print(f"  --------------------------------------------------------------")
        print(f"  [Prompt {q_idx}/3] {q_item['name']}")
        print(f"  Query: \"{q_item['query']}\"")
        print(f"  --------------------------------------------------------------")
        
        t0 = time.perf_counter()
        first_token_time = None
        full_text = ""
        token_count = 0
        
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    content_piece = chunk.choices[0].delta.content
                    full_text += content_piece
                    token_count += 1
                    
            t_end = time.perf_counter()
            ttft = first_token_time - t0 if first_token_time else 0
            decode_time = t_end - first_token_time if first_token_time else 0
            total_time = t_end - t0
            throughput = token_count / decode_time if decode_time > 0 else 0
            
            # Extract think tag and json
            think_match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
            think_content = think_match.group(1).strip() if think_match else ("(None - standard mode)" if not is_grug else "(Omitted by model)")
            
            json_match = re.search(r"\{.*\}", full_text, re.DOTALL)
            json_valid = False
            steps_count = 0
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    steps = data.get("steps", [])
                    steps_count = len(steps)
                    json_valid = steps_count > 0
                except Exception:
                    json_valid = False
                    
            print(f"  ⏱️  TTFT (Prompt Prefill) : {ttft:.2f}s")
            print(f"  ⚡ Generation Speed     : {throughput:.1f} tokens/s ({token_count} tokens in {decode_time:.2f}s)")
            print(f"  ⌛ Total Latency        : {total_time:.2f}s")
            if is_grug:
                preview_think = think_content[:120] + "..." if len(think_content) > 120 else think_content
                print(f"  💭 Grug <think> Trace   : {preview_think}")
            print(f"  📦 JSON Valid Plan      : {json_valid} ({steps_count} steps)")
            print(f"  📜 Full Raw Output:\n{full_text.strip()}\n")
            
            model_records.append({
                "prompt_id": q_item["id"],
                "ttft_s": ttft,
                "decode_s": decode_time,
                "total_s": total_time,
                "tokens": token_count,
                "throughput_tps": throughput,
                "json_valid": json_valid,
                "think_trace": think_content,
                "full_output": full_text
            })
            
        except Exception as e:
            t_end = time.perf_counter()
            print(f"  ❌ Error executing {model_name}: {e}\n")
            model_records.append({
                "prompt_id": q_item["id"],
                "ttft_s": 0,
                "decode_s": 0,
                "total_s": t_end - t0,
                "tokens": 0,
                "throughput_tps": 0,
                "json_valid": False,
                "think_trace": f"Error: {e}",
                "full_output": ""
            })
            
    all_metrics[model_name] = model_records
    print(f"  [Model {model_name} complete. Pausing 2s before next model...]\n")
    time.sleep(2)

# ==============================================================================
# FINAL COMPARATIVE REPORT
# ==============================================================================
print(f"\n{'='*95}")
print(f"  FINAL SEQUENTIAL BENCHMARK: GEMMA (GRUG PROMPT) vs OTHERS (STANDARD)")
print(f"{'='*95}\n")

print(f"{'Model Name':<32} | {'Prompt Style':<15} | {'Avg TTFT':<9} | {'Avg TPS':<10} | {'Avg Tokens':<11} | {'Avg Total':<10} | {'Valid'}")
print(f"{'-'*32}-|-{'-'*15}-|-{'-'*9}-|-{'-'*10}-|-{'-'*11}-|-{'-'*10}-|-{'-'*5}")

for m_name, recs in all_metrics.items():
    is_g = "Grug <think>" if "grug" in m_name.lower() else "Standard"
    valid_recs = [r for r in recs if r["tokens"] > 0]
    if not valid_recs:
        print(f"{m_name:<32} | {is_g:<15} | {'FAILED':<9} | {'N/A':<10} | {'0':<11} | {'N/A':<10} | 0/3")
        continue
    avg_ttft = sum(r["ttft_s"] for r in valid_recs) / len(valid_recs)
    avg_tps = sum(r["throughput_tps"] for r in valid_recs) / len(valid_recs)
    avg_tok = sum(r["tokens"] for r in valid_recs) / len(valid_recs)
    avg_tot = sum(r["total_s"] for r in valid_recs) / len(valid_recs)
    valid_json_count = sum(1 for r in recs if r["json_valid"])
    
    print(f"{m_name:<32} | {is_g:<15} | {avg_ttft:<7.2f}s | {avg_tps:<8.1f} t/s| {avg_tok:<11.1f} | {avg_tot:<8.2f}s  | {valid_json_count}/3")

print("\nBenchmark completed successfully.")

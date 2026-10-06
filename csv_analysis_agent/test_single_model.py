import sys
import io
import time
import re
import json
import argparse
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--is_grug", action="store_true")
args = parser.parse_args()

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

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

template = GRUG_PROMPT_TEMPLATE if args.is_grug else STANDARD_PROMPT_TEMPLATE
prompt_mode = "GRUG SPEECH (<think>)" if args.is_grug else "STANDARD (Direct JSON)"

print(f"\n{'='*75}")
print(f"  TESTING MODEL: {args.model}")
print(f"  PROMPT MODE  : {prompt_mode}")
print(f"{'='*75}\n")

results = []

for q_idx, q_item in enumerate(BENCHMARK_PROMPTS, 1):
    full_prompt = template + q_item["query"]
    print(f"[{q_idx}/3] {q_item['name']}")
    print(f"Query: \"{q_item['query']}\"")
    
    t0 = time.perf_counter()
    first_token_time = None
    full_text = ""
    token_count = 0
    
    try:
        stream = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0,
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text += chunk.choices[0].delta.content
                token_count += 1
                
        t_end = time.perf_counter()
        ttft = first_token_time - t0 if first_token_time else 0
        decode_time = t_end - first_token_time if first_token_time else 0
        total_time = t_end - t0
        tps = token_count / decode_time if decode_time > 0 else 0
        
        think_match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
        think_text = think_match.group(1).strip() if think_match else ("(N/A Standard Mode)" if not args.is_grug else "(No think tag)")
        
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
                
        print(f"  ⏱️ TTFT: {ttft:.2f}s | Decode: {decode_time:.2f}s ({token_count} tokens @ {tps:.1f} t/s) | Total: {total_time:.2f}s")
        if args.is_grug:
            print(f"  💭 <think>: {think_text}")
        print(f"  📦 JSON Steps: {json_valid} ({steps_count} steps)")
        print(f"  📜 Output:\n{full_text.strip()}\n")
        print(f"{'-'*75}")
        
        results.append({
            "id": q_item["id"],
            "ttft": ttft,
            "decode": decode_time,
            "total": total_time,
            "tokens": token_count,
            "tps": tps,
            "valid": json_valid,
            "think": think_text,
            "output": full_text
        })
    except Exception as e:
        print(f"  ❌ Error: {e}\n")

# Save summary to a json file for aggregation
with open(f"results_{args.model.replace('/', '_').replace(':', '_')}.json", "w", encoding="utf-8") as f:
    json.dump({"model": args.model, "is_grug": args.is_grug, "results": results}, f, indent=2)

avg_ttft = sum(r["ttft"] for r in results) / len(results) if results else 0
avg_tps = sum(r["tps"] for r in results) / len(results) if results else 0
avg_tot = sum(r["total"] for r in results) / len(results) if results else 0
avg_tok = sum(r["tokens"] for r in results) / len(results) if results else 0

print(f"\n>>> SUMMARY FOR {args.model}: Avg TTFT={avg_ttft:.2f}s, Avg TPS={avg_tps:.1f} t/s, Avg Tokens={avg_tok:.1f}, Avg Latency={avg_tot:.2f}s <<<\n")

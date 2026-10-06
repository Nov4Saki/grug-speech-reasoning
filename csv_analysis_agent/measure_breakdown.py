import sys
import io
import time
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

prompt = """You are an expert data analysis planner. Break down the user's request into a sequential execution plan of tools.

Dataframe Columns: ['Order_ID', 'Customer', 'Product', 'Category', 'Price', 'Quantity', 'Discount_Percent', 'Total', 'City', 'Payment_Method', 'Status', 'Order_Date']
Sample Rows: [{"Order_ID": 1001, "Customer": "Ahmed", "Product": "Laptop", "Category": "Electronics", "Price": 1200, "Quantity": 2, "Discount_Percent": 0.1, "Total": 2160, "City": "Cairo", "Payment_Method": "Credit Card", "Status": "Completed", "Order_Date": "2024-01-15"}]

Available Tools:
- aggregate_tool: Performs stats (sum, mean, min, max, count, median). Supports by_column for group by.
- filter_tool: Filters rows using query string.
- view_tool: Sorts data (sort_by, ascending) or views top N rows.
- chart_tool: Generates interactive Plotly charts (bar, line, pie, scatter).

Output a valid JSON object matching:
{
    "steps": [
        {
            "tool_name": "exact_tool_name",
            "tool_args": { "arg_name": "arg_value" }
        }
    ]
}

User Query: فلتر الصفوف اللي فيها الكمية أكبر من 5، وبعدين احسب مجموع المبيعات لكل فئة، وترتيب النتيجة من الأعلى للأقل"""

models = [
    "gemma-4-e2b-grugspeech-native",
    "qwen3.5-2b-claude-4.6-opus-reasoning-distilled"
]

print(f"{'Model':<45} | {'TTFT (Prefill)':<15} | {'Decode (Gen)':<15} | {'Tokens':<8} | {'Gen Speed':<12} | {'Total'}")
print(f"{'-'*45}-|-{'-'*15}-|-{'-'*15}-|-{'-'*8}-|-{'-'*12}-|-{'-'*8}")

for m in models:
    t0 = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    stream = client.chat.completions.create(
        model=m,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True
    )
    
    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        if chunk.choices and chunk.choices[0].delta.content:
            token_count += 1
            
    t_end = time.perf_counter()
    
    ttft = first_token_time - t0 if first_token_time else 0
    decode_time = t_end - first_token_time if first_token_time else 0
    total_time = t_end - t0
    gen_speed = token_count / decode_time if decode_time > 0 else 0
    
    print(f"{m:<45} | {ttft:<15.2f}s | {decode_time:<15.2f}s | {token_count:<8} | {gen_speed:<10.1f} t/s | {total_time:.2f}s")

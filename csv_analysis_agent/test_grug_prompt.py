import sys
import io
import time
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

prompt = """You are a data analysis planner. Reason in Grug Speech inside <think> tags.

Dataframe Columns: ['Order_ID', 'Customer', 'Product', 'Category', 'Price', 'Quantity', 'Discount_Percent', 'Total', 'City', 'Payment_Method', 'Status', 'Order_Date']
Sample Rows: [{"Order_ID": 1001, "Customer": "Ahmed", "Product": "Laptop", "Category": "Electronics", "Price": 1200, "Quantity": 2, "Discount_Percent": 0.1, "Total": 2160, "City": "Cairo", "Payment_Method": "Credit Card", "Status": "Completed", "Order_Date": "2024-01-15"}]

Available Tools:
- aggregate_tool: Performs stats (sum, mean, min, max, count, median). Supports by_column for group by. Note: output metric column keeps its original name (e.g. Total, not sum_Total).
- filter_tool: Filters rows using query string.
- view_tool: Sorts data (sort_by, ascending) or views top N rows.
- chart_tool: Generates interactive Plotly charts (bar, line, pie, scatter).

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

User Query: فلتر الصفوف اللي فيها الكمية أكبر من 5، وبعدين احسب مجموع المبيعات لكل فئة، وترتيب النتيجة من الأعلى للأقل"""

grug_models = [
    "gemma-4-e2b-grugspeech-native",
    "qwen3.5-2b-grugspeech",
    "qwen3.5-4b-grugspeech-native"
]

for m in grug_models:
    print(f"\n{'='*75}")
    print(f"  MODEL: {m}")
    print(f"{'='*75}")
    t0 = time.perf_counter()
    try:
        res = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        t_elapsed = time.perf_counter() - t0
        content = res.choices[0].message.content or ""
        print(f"⏱️ Time taken: {t_elapsed:.2f}s\n")
        print(content)
    except Exception as e:
        print(f"❌ Error: {e}")

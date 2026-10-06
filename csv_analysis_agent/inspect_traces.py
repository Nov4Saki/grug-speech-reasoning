import sys
import io
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
    "qwen3.5-2b-grugspeech",
    "qwen3.5-2b-claude-4.6-opus-reasoning-distilled",
    "minicpm5-2b"
]

for m in models:
    print(f"\n{'='*70}")
    print(f"RAW TRACE FROM: {m}")
    print(f"{'='*70}")
    try:
        res = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = res.choices[0].message.content or ""
        print(content)
    except Exception as e:
        print(f"Error: {e}")

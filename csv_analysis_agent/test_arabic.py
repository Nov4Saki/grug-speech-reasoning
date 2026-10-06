import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from graph import graph

df = pd.read_excel("sales.xlsx")
print("Data loaded:", df.shape)

query = "فلتر الصفوف اللي فيها الكمية أكبر من 5، وبعدين احسب مجموع المبيعات لكل فئة، وترتيب النتيجة من الأعلى للأقل"

print(f"\nQuery: {query}\n")

state = {
    "user_query": query,
    "dataframe": df.copy(),
    "plan": [],
    "current_step_index": 0,
    "result": None,
    "error": "",
}

final_state = graph.invoke(state)

if final_state.get("error"):
    print(f"Error: {final_state['error']}")
else:
    print("Execution Plan:")
    for i, step in enumerate(final_state.get("plan", []), 1):
        print(f"  Step {i}: {step['tool_name']} → {step['tool_args']}")
    print("\nFinal Result:")
    print(final_state.get("result"))

import sys
import io
import pandas as pd
from graph import graph

# Ensure stdin/stdout handle Arabic (UTF-8) correctly on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "buffer"):
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")


def main():
    try:
        df = pd.read_excel("sales.xlsx")
        print(" Sales data loaded successfully!")
    except Exception as e:
        print(f" Error loading Excel file: {e}")
        return

    print("\n=== AI Data Assistant (Type 'exit' or 'quit' to end) ===")

    while True:
        user_query = input("\nUser Query: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_query:
            continue

      
        initial_state = {
            "user_query": user_query,
            "dataframe": df.copy(),
            "plan": [],
            "current_step_index": 0,
            "result": None,
            "error": "",
        }

        final_state = graph.invoke(initial_state)

        if final_state.get("error"):
            print(f"\n Error: {final_state['error']}")
        else:
            print("\n Execution Plan:")
            for i, step in enumerate(final_state.get("plan", []), 1):
                print(
                    f"  Step {i}: {step['tool_name']} with args {step['tool_args']}"
                )

            print("\n Final Result:")
            print(final_state.get("result"))


if __name__ == "__main__":
    main()
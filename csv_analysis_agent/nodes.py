import json
import re
import pandas as pd
from openai import OpenAI
from tools import aggregate_tool, filter_tool, view_tool, transform_tool, chart_tool

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

TOOLS = {
    "aggregate_tool": {
        "function": aggregate_tool,
        "description": "Performs math stats (sum, mean, min, max, count, median, std). Supports group by using 'by_column', and top/bottom N using 'n'.",
        "arguments": {
            "column": "string",
            "operation": "string (sum, mean, min, max, median, count, std)",
            "by_column": "optional string (column to group by)",
            "n": "optional integer (for top or bottom N rows)",
        },
    },
    "filter_tool": {
        "function": filter_tool,
        "description": "Filters rows using a logical query string. Supports 'and', 'or', 'in', '>', '<', '=='. Wrap column names in backticks if they contain spaces.",
        "arguments": {
            "query_string": "string expression, e.g., (`Sales` > 1000 and `Branch` == 'Cairo') or `Branch` in ['Alex', 'Giza']"
        },
    },
    "view_tool": {
        "function": view_tool,
        "description": "Sorts data, views top N rows, or gets unique values of a column.",
        "arguments": {
            "sort_by": "optional string (column to sort)",
            "ascending": "optional boolean (default True)",
            "n": "optional integer (rows to view)",
            "unique_column": "optional string (column to get unique values from)",
        },
    },
    "transform_tool": {
        "function": transform_tool,
        "description": "Modifies table schema: add calculated column, drop empty rows, drop columns, replace specific values, or set all values in a column to one value.",
        "arguments": {
            "action": "string: 'add_column', 'drop_na', 'drop_columns', 'replace', or 'set_value'",
            "new_column": "string (required for add_column)",
            "formula": "string formula e.g. 'ColA * ColB' (required for add_column)",
            "columns": "optional list of column strings (for drop_na or drop_columns)",
            "column": "string (column to modify, required for replace or set_value)",
            "replacements": "optional dictionary mapping old values to new values",
            "value": "value to assign to all rows in the selected column (required for set_value)"
        },
    },
    "chart_tool": {
        "function": chart_tool,
        "description": (
            "Creates an interactive Plotly chart and saves it as an HTML file. "
            "Use this whenever the user asks to visualize, plot, draw, or create a chart/graph/dashboard. "
            "Supports: bar, line, area, scatter, histogram, pie, heatmap. "
            "Has a built-in 'agg' param (sum/mean/count/min/max) so you can skip a separate aggregate_tool step for simple cases. "
            "Use 'color_by' for grouped/coloured charts (e.g. sales per city per category)."
        ),
        "arguments": {
            "chart_type": "string: 'bar', 'line', 'area', 'scatter', 'histogram', 'pie', or 'heatmap'",
            "x": "string — column for X-axis (or label column for pie)",
            "y": "string — column for Y-axis (or value column for pie/histogram)",
            "title": "optional string — chart title",
            "color_by": "optional string — column to group/colour by",
            "agg": "optional string — auto-aggregate y by x before plotting: 'sum', 'mean', 'count', 'min', 'max'",
            "output_path": "optional string — file path to save (e.g. 'charts/sales.html'). Defaults to charts/chart_<type>.html",
        },
    },
}


tool_descriptions = json.dumps(
    {
        k: {"desc": v["description"], "args": v["arguments"]}
        for k, v in TOOLS.items()
    },
    indent=2,
)


def _sanitize(text: str) -> str:
    """Strip lone surrogate characters that break UTF-8 encoding."""
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def planner_node(state):
    user_query = _sanitize(state["user_query"])
    df = state["dataframe"]

    columns = list(df.columns)
    # Sanitize sample data: convert to string then strip surrogates
    raw_sample = json.dumps(df.head(2).to_dict(orient="records"), default=str)
    sample_data = _sanitize(raw_sample)

    prompt = f"""You are an expert data analysis planner. Break down the user's request into a sequential execution plan of tools.

Dataframe Columns: {columns}
Sample Rows: {sample_data}

Available Tools:
{tool_descriptions}

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
        model="gemma-4-e2b-grugspeech-native",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    text = response.choices[0].message.content or ""

    # Safe JSON extraction using regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"error": "Failed to extract valid JSON plan from the model."}

    try:
        plan_data = json.loads(match.group(0))
        steps = plan_data.get("steps", [])
        if not steps:
            return {"error": "Model returned an empty execution plan."}
        return {"plan": steps, "current_step_index": 0, "error": ""}
    except Exception as e:
        return {"error": f"JSON parsing failed: {e}"}


def executor_node(state):
    plan = state.get("plan", [])
    index = state.get("current_step_index", 0)
    df = state["dataframe"]

    step = plan[index]
    tool_name = step.get("tool_name")
    tool_args = step.get("tool_args", {})

    if tool_name not in TOOLS:
        return {
            "error": f"Tool '{tool_name}' is not recognized.",
            "current_step_index": index + 1,
        }

    tool_fn = TOOLS[tool_name]["function"]

    try:
        output = tool_fn(df, **tool_args)

        # Update dataframe if transformation or filter produced a new DataFrame
        if isinstance(output, pd.DataFrame):
            return {
                "dataframe": output,
                "result": output,
                "current_step_index": index + 1,
            }
        else:
            return {
                "result": output,
                "current_step_index": index + 1,
            }
    except Exception as e:
        return {
            "error": f"Error executing '{tool_name}': {e}",
            "current_step_index": index + 1,
        }
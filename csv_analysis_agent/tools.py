import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



def aggregate_tool(df, column, operation, by_column=None, n=None):
   
    if by_column:
        return df.groupby(by_column)[column].agg(operation).reset_index()

    if n is not None:
        if operation == "max":
            return df.nlargest(int(n), column)
        elif operation == "min":
            return df.nsmallest(int(n), column)

    col = pd.to_numeric(df[column], errors="coerce")

    ops = {
        "sum": col.sum,
        "mean": col.mean,
        "average": col.mean,
        "min": col.min,
        "max": col.max,
        "median": col.median,
        "count": col.count,
        "std": col.std,
    }

    if operation in ops:
        return ops[operation]()
    else:
        raise ValueError(f"Operation '{operation}' not supported")


def filter_tool(df, query_string):
   
    return df.query(query_string, engine="python")


def view_tool(df, sort_by=None, ascending=True, n=None, unique_column=None):
  
    if unique_column:
        return df[unique_column].dropna().unique().tolist()

    result = df

    if sort_by:
        result = result.sort_values(
            by=sort_by,
            ascending=bool(ascending)
        )

    if n is not None:
        result = result.head(int(n))

    return result


def transform_tool(df, action, **kwargs):

    df = df.copy()

    if action == "drop_na":
        subset = kwargs.get("columns")
        return df.dropna(subset=subset) if subset else df.dropna()

    elif action == "drop_columns":
        cols = kwargs.get("columns", [])
        return df.drop(columns=cols, errors="ignore")

    elif action == "add_column":
        new_col = kwargs.get("new_column")
        formula = kwargs.get("formula")

        df[new_col] = df.eval(formula)

        return df
    elif action == "replace":
        column = kwargs.get("column")
        replacements = kwargs.get("replacements", {})
     
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        if not isinstance(replacements, dict):
            raise ValueError("'replacements' must be a dictionary")

        df[column] = df[column].replace(replacements)

        return df

    elif action == "set_value":
        column = kwargs.get("column")
        value = kwargs.get("value")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        df[column] = value

        return df

    return df


# ── Chart types supported ────────────────────────────────────────────────────
_CHART_TYPES = ("bar", "line", "area", "scatter", "histogram", "pie", "heatmap")


def chart_tool(
    df: pd.DataFrame,
    chart_type: str,
    x: str = None,
    y: str = None,
    title: str = None,
    color_by: str = None,
    agg: str = None,
    top_n: int = 15,
    output_path: str = None,
) -> str:
    """
    Create an interactive Plotly chart and save it to an HTML file.
    Optimized for dashboards with auto-aggregation, Top-N thresholding, and smart sampling.

    Args:
        df         : The pandas DataFrame.
        chart_type : One of 'bar', 'line', 'area', 'scatter', 'histogram', 'pie', 'heatmap'.
        x          : Column for X-axis (or label column for pie).
        y          : Column for Y-axis (or value column for pie/histogram).
        title      : Chart title (auto-generated if omitted).
        color_by   : Column to group/colour by (grouped bars, coloured lines/scatter …).
        agg        : Aggregation operation — 'sum', 'mean', 'count', 'min', 'max'.
                     Defaults to 'sum' for numeric y in bar/pie/line/area charts.
        top_n      : Max distinct categories on X / pie slices before grouping into 'Other' (default 15).
        output_path: File path for saved HTML (defaults to 'charts/chart_<type>.html').

    Returns:
        The absolute path of the saved file (str).
    """
    df = df.copy()
    chart_type = chart_type.lower().strip()
    if chart_type not in _CHART_TYPES:
        raise ValueError(
            f"chart_type '{chart_type}' not supported. "
            f"Choose from: {_CHART_TYPES}"
        )

    # ── Resolve default column fallbacks ──────────────────────────────────────
    # For pie charts, users/models often put the label in color_by or x
    if chart_type == "pie":
        label_col = x or color_by or "Category"
        value_col = y or "Total"
        if label_col not in df.columns:
            # Fallback to first non-numeric or Category
            non_nums = df.select_dtypes(exclude=["number"]).columns.tolist()
            label_col = non_nums[0] if non_nums else df.columns[0]
        if value_col not in df.columns or not pd.api.types.is_numeric_dtype(df[value_col]):
            nums = df.select_dtypes(include=["number"]).columns.tolist()
            value_col = nums[0] if nums else df.columns[-1]

        # Aggregate slices
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
        grouped = df.groupby(label_col, as_index=False)[value_col].sum()
        grouped = grouped.sort_values(by=value_col, ascending=False)

        # Thresholding: keep top 7 slices and group the rest into 'Other'
        max_slices = 7
        if len(grouped) > max_slices:
            top_slices = grouped.iloc[:max_slices]
            other_val = grouped.iloc[max_slices:][value_col].sum()
            other_row = pd.DataFrame([{label_col: "Other", value_col: other_val}])
            grouped = pd.concat([top_slices, other_row], ignore_index=True)

        chart_title = title or f"Distribution of {value_col} by {label_col}"
        fig = px.pie(
            grouped,
            names=label_col,
            values=value_col,
            title=chart_title,
            hole=0.35,
            template="plotly_white",
        )
        fig.update_traces(textinfo="label+percent", hovertemplate="<b>%{label}</b><br>Value: %{value:,.2f}<br>Share: %{percent}")

    # ── Bar Chart with Smart Aggregation & Top-N Thresholding ────────────────
    elif chart_type == "bar":
        # Determine numeric y and dimension x
        if not y and x:
            # Count occurrences of x
            df_plot = df[x].value_counts().reset_index()
            df_plot.columns = [x, "Count"]
            y = "Count"
        else:
            y_col = y or "Total"
            x_col = x or "City"
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0)
            agg_op = agg or "sum"

            group_cols = [x_col] if not color_by else [x_col, color_by]
            df_plot = df.groupby(group_cols, as_index=False)[y_col].agg(agg_op)

            # Thresholding on X dimension: if too many distinct categories, keep Top N
            unique_x = df_plot[x_col].nunique()
            if unique_x > top_n:
                # Rank X categories by total Y
                top_categories = (
                    df_plot.groupby(x_col)[y_col]
                    .sum()
                    .nlargest(top_n)
                    .index.tolist()
                )
                if not color_by:
                    other_sum = df_plot[~df_plot[x_col].isin(top_categories)][y_col].sum()
                    df_plot = df_plot[df_plot[x_col].isin(top_categories)]
                    if other_sum > 0:
                        df_plot = pd.concat(
                            [df_plot, pd.DataFrame([{x_col: "Other", y_col: other_sum}])],
                            ignore_index=True,
                        )
                else:
                    # Filter to top categories for grouped readability
                    df_plot = df_plot[df_plot[x_col].isin(top_categories)]

            # Sort descending for single-series bar
            if not color_by:
                df_plot = df_plot.sort_values(by=y_col, ascending=False)

        chart_title = title or f"{y} by {x}"
        if color_by:
            chart_title += f" (grouped by {color_by})"

        fig = px.bar(
            df_plot,
            x=x,
            y=y,
            color=color_by,
            barmode="group",
            title=chart_title,
            template="plotly_white",
        )
        fig.update_layout(xaxis_tickangle=-30 if df_plot[x].nunique() > 6 else 0)

    # ── Scatter Plot with Smart Sampling & Styling ────────────────────────────
    elif chart_type == "scatter":
        # If dataset is large (>2000 rows), sample to keep dashboard light and interactive
        sample_size = 2000
        is_sampled = len(df) > sample_size
        if is_sampled:
            df_plot = df.sample(n=sample_size, random_state=42)
            subtitle = f" (Sampled {sample_size:,} of {len(df):,} rows)"
        else:
            df_plot = df
            subtitle = ""

        chart_title = (title or f"{y} vs {x}") + subtitle
        fig = px.scatter(
            df_plot,
            x=x,
            y=y,
            color=color_by,
            title=chart_title,
            opacity=0.7,
            template="plotly_white",
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))

    # ── Line / Area Chart ─────────────────────────────────────────────────────
    elif chart_type in ("line", "area"):
        agg_op = agg or "sum"
        group_cols = [x] if not color_by else [x, color_by]
        if y and pd.api.types.is_numeric_dtype(df[y]):
            df_plot = df.groupby(group_cols, as_index=False)[y].agg(agg_op)
        else:
            df_plot = df

        chart_title = title or f"{chart_type.capitalize()} of {y} over {x}"
        if chart_type == "line":
            fig = px.line(df_plot, x=x, y=y, color=color_by, markers=True, title=chart_title, template="plotly_white")
        else:
            fig = px.area(df_plot, x=x, y=y, color=color_by, title=chart_title, template="plotly_white")

    # ── Histogram ─────────────────────────────────────────────────────────────
    elif chart_type == "histogram":
        chart_title = title or f"Distribution of {x or y}"
        fig = px.histogram(df, x=x or y, color=color_by, title=chart_title, template="plotly_white")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    elif chart_type == "heatmap":
        if not (x and y and color_by):
            raise ValueError("heatmap requires x (columns), color_by (rows), and y (metric values).")
        pivot = df.pivot_table(index=color_by, columns=x, values=y, aggfunc=agg or "sum", fill_value=0)
        chart_title = title or f"Heatmap of {y} ({color_by} vs {x})"
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="Blues",
            )
        )
        fig.update_layout(title=chart_title, template="plotly_white")

    # ── Save Output ───────────────────────────────────────────────────────────
    if not output_path:
        os.makedirs("charts", exist_ok=True)
        output_path = os.path.join("charts", f"chart_{chart_type}.html")

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.endswith(".png"):
        fig.write_image(output_path)
    else:
        fig.write_html(output_path, include_plotlyjs="cdn")

    return f"Chart saved → {output_path}"

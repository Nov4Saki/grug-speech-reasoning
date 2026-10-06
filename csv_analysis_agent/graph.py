from langgraph.graph import StateGraph, START, END
from state import GraphState
from nodes import planner_node, executor_node


def should_continue(state: GraphState):
    # إيقاف التنفيذ لو حصل خطأ
    if state.get("error"):
        return END

    plan = state.get("plan", [])
    current_index = state.get("current_step_index", 0)

    # الاستمرار لو لسه باقي خطوات
    if current_index < len(plan):
        return "executor"

    return END


builder = StateGraph(GraphState)

builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")

# المسار الشرطي للتنفيذ
builder.add_conditional_edges("executor", should_continue, ["executor", END])

graph = builder.compile()


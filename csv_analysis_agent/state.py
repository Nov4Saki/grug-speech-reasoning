from typing import TypedDict, Any, List, Dict


class GraphState(TypedDict):
    user_query: str
    dataframe: Any
    plan: List[Dict[str, Any]]
    current_step_index: int
    result: Any
    error: str
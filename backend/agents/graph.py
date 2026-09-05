from langgraph.graph import StateGraph, START, END

try:
    from backend.agents.state import AnalystState
    from backend.tools.data_loader import load_dataset
    from backend.tools.data_profile import get_dataset_profile
    from backend.tools.statistics import get_column_statistics
    from backend.tools.visualization import visualize_column
except ModuleNotFoundError:
    from agents.state import AnalystState
    from tools.data_loader import load_dataset
    from tools.data_profile import get_dataset_profile
    from tools.statistics import get_column_statistics
    from tools.visualization import visualize_column


tools = [
    get_column_statistics,
    visualize_column,
]


def analyze_dataset(state: AnalystState):
    file_path = state["file_path"]

    df = load_dataset(file_path)
    profile = get_dataset_profile(df)

    answer = (
        f"The dataset contains {profile['number_of_rows']} rows "
        f"and {profile['number_of_columns']} columns."
    )

    return {
        "answer": answer
    }


def build_graph():

    graph = StateGraph(AnalystState)

    graph.add_node("analyze_dataset", analyze_dataset)

    graph.add_edge(START, "analyze_dataset")
    graph.add_edge("analyze_dataset", END)

    return graph.compile()

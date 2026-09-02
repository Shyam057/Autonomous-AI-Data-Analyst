from langgraph.graph import StateGraph,START,END

from agents.state import AnalystState
from tools.data_loader import load_dataset
from tools.data_profile import get_dataset_profile

def analyze_dataset(state:AnalystState):

    file_path=state["file_path"]

    #Load dataset
    df=load_dataset(file_path)

    # Get dataset Information
    profile=get_dataset_profile(df)

    answer=(f"The dataset contains {profile["number_of_rows"]} rows"
            f"and {profile['number_of_columns']} columns.")
            

    return {
        "answer":answer
    }

def build_graph():
    graph=StateGraph(AnalystState)
    graph.add_node("analyze_dataset",analyze_dataset)

    graph.add_edge(START,"analyze_dataset")
    graph.add_edge("analyze_dataset","END")

    return graph.compile()
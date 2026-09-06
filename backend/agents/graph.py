import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

try:
    from backend.agents.state import AnalystState
    from backend.agents.tools import TOOLS
except ModuleNotFoundError:
    from agents.state import AnalystState
    from agents.tools import TOOLS


SYSTEM_PROMPT = """You are an autonomous data analyst.

Answer the user's question using the dataset at the file path provided in the
conversation. Choose the smallest useful tool first. Every tool call must
include the exact file_path from the conversation. For a complex question,
call multiple tools and use earlier results to decide what to inspect next.
Do not guess numbers or answer a different question. For "how many customers"
or "how many rows", use dataset_overview_tool with the user question, not churn_distribution. For an
average or other numeric statistic, use numeric_column_question. For a named
categorical value such as "ham" or "spam", use categorical_value_count. For
churn
rate, use churn_distribution. For "highest churn by [column]", use
churn_by_category. For a distribution request, use categorical_distribution or
numeric_column_question only when appropriate. For a question about churn
increasing, first use time_trend_limitation; do not claim a trend without a
time-based variable. Treat identifier columns such as CustomerID as identifiers,
not analytical measures, unless the user explicitly asks for that column.
After the tools return, answer every part of the question directly and
concisely. Never add unrelated drivers, charts, or findings. Never return an
empty final answer.
"""


def _create_llm():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to the .env file.")
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key,
    ).bind_tools(TOOLS)


def call_llm(state: AnalystState):
    """Ask the LLM to answer or request the next analysis tool."""
    messages = list(state.get("messages", []))
    is_first_call = not messages
    if not messages:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"file_path: {state['file_path']}\n"
                    f"user_question: {state['question']}"
                )
            ),
        ]
    response = _create_llm().invoke(messages)
    result = {"messages": [response]}
    if is_first_call:
        result["messages"] = messages + [response]
    if getattr(response, "tool_calls", None):
        result["tool_name"] = response.tool_calls[0]["name"]
    return result


def save_final_answer(state: AnalystState):
    """Copy the final LLM response into fields used by application callers."""
    message = state["messages"][-1]
    tool_name = state.get("tool_name")
    if getattr(message, "tool_calls", None):
        tool_name = message.tool_calls[0]["name"]
    answer = message.content if isinstance(message.content, str) else str(message.content)
    if not answer.strip():
        tool_results = [
            item.content for item in state["messages"]
            if item.__class__.__name__ == "ToolMessage" and isinstance(item.content, str)
        ]
        answer = " ".join(tool_results) or "The analysis completed without a readable answer."
    return {
        "answer": answer,
        "tool_name": tool_name,
    }


def build_graph():
    """Build START -> LLM -> tools -> LLM ... -> END workflow."""
    graph = StateGraph(AnalystState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("final_answer", save_final_answer)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: "final_answer"})
    graph.add_edge("tools", "llm")
    graph.add_edge("final_answer", END)
    return graph.compile()

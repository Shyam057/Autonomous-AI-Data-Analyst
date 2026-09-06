"""State passed between nodes in the data analyst graph."""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages

class AnalystState(TypedDict):
    question: str
    file_path: str
    messages: Annotated[list, add_messages]
    answer: Optional[str]
    tool_name: Optional[str]
"""
This defines the information that our langgraph agent will maintain
"""


from typing import TypedDict, Optional

class AnalystState(TypedDict):
    question:str
    file_path:str
    answer: Optional[str]
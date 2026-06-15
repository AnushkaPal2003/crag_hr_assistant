from typing import TypedDict, Annotated, List
from pydantic import BaseModel
from typing import Literal
from langgraph.graph.message import add_messages


# CRAG Agent State
# Passed between every node in the pipeline
class CRAGState(TypedDict):
    question        : str          # original user question
    documents       : List[str]    # retrieved doc contents (vectorstore or web)
    generation      : str          # final LLM answer
    web_search_used : bool         # True if fallback to web search happened
    rewritten_query : str          # rewritten query used for web search
    messages        : Annotated[list, add_messages]  # chat history for memory


# Structured Output — Document Grader
# Forces LLM to return only "yes" or "no"
class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"]

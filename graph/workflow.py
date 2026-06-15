from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import CRAGState
from nodes.nodes import (
    retrieve,
    grade_documents,
    rewrite_query,
    web_search,
    generate,
    route_after_grading
)


def build_graph():
    """
    Build and compile the CRAG (Corrective RAG) pipeline for HR policies.
    """
    graph = StateGraph(CRAGState)

    # Register all nodes
    graph.add_node("retrieve",        retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query",   rewrite_query)
    graph.add_node("web_search",      web_search)
    graph.add_node("generate",        generate)

    # Define edges 

    # always: start --> retrieve --> grade
    graph.add_edge(START,      "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    # conditional: after grading decide path
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate"    : "generate",       # relevant docs --> answer directly
            "rewrite_query": "rewrite_query"  # no relevant docs --> web search path
        }
    )

    # web search path: rewrite --> search --> generate
    graph.add_edge("rewrite_query", "web_search")
    graph.add_edge("web_search",    "generate")

    # always: generate --> end
    graph.add_edge("generate", END)

    # Compile with MemorySaver for session persistence 
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

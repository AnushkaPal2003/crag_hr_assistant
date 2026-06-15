import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # loads LangSmith tracing
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from graph.workflow import build_graph
import uvicorn
from config import API_HOST, API_PORT

# FastAPI app
app = FastAPI(
    title="CRAG HR Policy Assistant API",
    description="Corrective RAG pipeline for HR policies — LangGraph + ChromaDB + openai + Tavily",
    version="1.0.0",
    docs_url="/docs",   # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"]
)

# Build agent once at startup
print("Building CRAG agent...")
agent = build_graph()
print("Agent ready.")


# Pydantic models
class QueryRequest(BaseModel):
    question  : str
    session_id: str = "default-session"

class QueryResponse(BaseModel):
    answer         : str
    web_search_used: bool
    session_id     : str
    source         : str


# Routes 
@app.get("/")
def root():
    """Root — health check"""
    return {
        "status" : "ok",
        "service": "CRAG HR Policy Assistant",
        "docs"   : "/docs"
    }


@app.get("/health")
def health():
    """Detailed health check — used by Docker"""
    return {
        "status"   : "healthy",
        "llm"      : "openai/gpt-4o-mini",
        "vectordb" : "ChromaDB",
        "search"   : "Tavily",
        "tracing"  : "LangSmith"
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main CRAG query endpoint.

    Flow:
    1. Retrieve from HR knowledge base
    2. Grade documents for relevance
    3. If relevant → generate answer
    4. If not relevant → rewrite query → web search → generate answer

    Returns answer + whether web search was used + source info.
    """
    try:
        cfg    = {"configurable": {"thread_id": request.session_id}}
        result = agent.invoke(
            {
                "question"        : request.question,
                "documents"       : [],
                "generation"      : "",
                "web_search_used" : False,
                "rewritten_query" : "",
                "messages"        : [HumanMessage(content=request.question)]
            },
            cfg
        )

        web_used = result.get("web_search_used", False)
        source   = "Web Search (Tavily)" if web_used else "HR Knowledge Base (ChromaDB)"

        return QueryResponse(
            answer         = result["generation"],
            web_search_used= web_used,
            session_id     = request.session_id,
            source         = source
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)

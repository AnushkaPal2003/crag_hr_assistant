# HR Policy Assistant — Corrective RAG (CRAG) System

An intelligent HR policy Q&A assistant built with **Corrective RAG (CRAG)** using LangGraph.

Unlike standard RAG, CRAG adds an LLM-based document grading step that filters out irrelevant retrievals and falls back to web search when needed — preventing hallucinations.

---

## Architecture

```
User Question
      ↓
  retrieve           →  ChromaDB semantic search (HR dataset)
      ↓
  grade_documents    →  LLM grades each doc: relevant or not?
      ↓
  [conditional routing]
      ├── relevant docs  →  generate  →  Answer  →  END
      └── no relevant docs
                ↓
          rewrite_query  →  optimize query for web search
                ↓
          web_search     →  Tavily fetches live results
                ↓
          generate       →  Answer using web context  →  END
```

---

## Tech Stack

| Component       | Tool                            |
|----------------|---------------------------------|
| Agent Framework | LangGraph (StateGraph)          |
| LLM             | OpenAI( gpt-4o-mini)              |
| Vector DB       | ChromaDB (cosine similarity)    |
| Embeddings      | HuggingFace all-MiniLM-L6-v2   |
| Web Search      | Tavily Search API               |
| Observability   | LangSmith                       |
| Backend API     | FastAPI                         |
| Frontend        | Streamlit                       |
| Dataset         | strova-ai/hr-policies-qa-dataset|
| Container       | Docker + Docker Compose         |

---

## Folder Structure

```
crag_hr/
├── config.py              →  API keys, LangSmith, constants
├── app.py                 →  Streamlit frontend
├── setup_vectorstore.py   →  run ONCE to build ChromaDB
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/                  →  ChromaDB persisted here
├── vectorstore/
│   └── setup.py           →  dataset loading + ChromaDB + retriever
├── nodes/
│   └── nodes.py           →  all 5 CRAG node functions + routing
└── graph/
    ├── state.py           →  CRAGState TypedDict + GradeDocuments Pydantic
    └── workflow.py        →  graph builder + conditional edges
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Create `.env` file**
```
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langsmith_key
```

**3. Build vectorstore (run once)**
```bash
python setup_vectorstore.py
```

**4a. Run Streamlit**
```bash
streamlit run app.py
```

**4b. Run FastAPI**
```bash
uvicorn api.main:app --reload
```
Swagger UI: http://localhost:8000/docs

**4c. Run with Docker**
```bash
docker-compose up --build
```

---
Dockerized deployment with separate containers for FastAPI backend (port 8000) and Streamlit frontend (port 8501)

## Key Concepts

**Why CRAG over standard RAG?**
Standard RAG blindly uses whatever docs are retrieved — even irrelevant ones.
CRAG adds an LLM grader that filters docs before generation.
If no relevant docs --> rewrites query --> web search fallback.
Result: fewer hallucinations, more accurate answers.

**LangSmith Observability**
All runs traced at `smith.langchain.com` under project `hr_assistant`.
Every node execution, LLM call, retrieval step, and routing decision is logged.

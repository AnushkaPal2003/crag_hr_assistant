from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from graph.state import CRAGState, GradeDocuments
from vectorstore.setup import get_retriever
from config import LLM_MODEL

# LLM 
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Tavily Web Search
from langchain_tavily import TavilySearch
web_search_tool = TavilySearch(max_results=3)



# Node 1 — Retrieve documents from ChromaDB

def retrieve(state: CRAGState) -> CRAGState:
    """
    Retrieve top-K documents from ChromaDB vectorstore.
    Uses semantic similarity search against HR policies dataset.
    Returns raw document contents — not yet graded for relevance.
    """
    print("[retrieve] Searching HR knowledge base...")
    question  = state["question"]
    retriever = get_retriever()
    docs      = retriever.invoke(question)

    doc_contents = [doc.page_content for doc in docs]
    print(f"[retrieve] Found {len(doc_contents)} candidate documents")

    return {
        "documents"     : doc_contents,
        "web_search_used": False
    }

# Node 2 — Grade documents for relevance

def grade_documents(state: CRAGState) -> CRAGState:
    """
    Core CRAG step — LLM grades each retrieved document.

    Why this matters:
    Vector similarity finds "similar" text but not always "relevant" text.
    LLM grading adds an intelligent quality filter on top of retrieval.

    Uses structured output (Pydantic) to force binary yes/no decision.
    Only relevant docs proceed to generation — irrelevant ones are dropped.
    """
    print("[grade_documents] Grading retrieved documents for relevance...")
    question  = state["question"]
    documents = state["documents"]

    GRADER_PROMPT = """You are an HR document relevance grader.

Assess whether the document is relevant to answer the HR policy question.

Rules:
- Grade as 'yes' if the document contains information that helps answer the question.
- Grade as 'no' if the document is completely unrelated.
- Be lenient — partial relevance counts as 'yes'.

HR Question: {question}
Document: {document}

Return ONLY 'yes' or 'no'."""

    grader_llm    = llm.with_structured_output(GradeDocuments)
    relevant_docs = []

    for i, doc in enumerate(documents):
        prompt = GRADER_PROMPT.format(question=question, document=doc)
        result = grader_llm.invoke(prompt)

        if result.binary_score == "yes":
            relevant_docs.append(doc)
            print(f"[grade_documents] Doc {i+1}: RELEVANT")
        else:
            print(f"[grade_documents] Doc {i+1}: NOT relevant — filtered out")

    print(f"[grade_documents] {len(relevant_docs)}/{len(documents)} docs passed grading")
    return {"documents": relevant_docs}



# Node 3 — Rewrite query for better web search

def rewrite_query(state: CRAGState) -> CRAGState:
    """
    When no relevant docs found in knowledge base,
    rewrite the original question into a better web search query.

    Example:
    Original: "how many days off do I get for bereavement?"
    Rewritten: "corporate bereavement leave policy days US"
    """
    print("[rewrite_query] Rewriting query for web search...")
    question = state["question"]

    REWRITE_PROMPT = """You are a search query optimizer for HR policy questions.

Rewrite the following HR question into a concise, effective web search query.
Focus on key HR/policy terms. Remove conversational language.
Return ONLY the rewritten query — no explanation, no quotes.

Original Question: {question}
Rewritten Query:"""

    response        = llm.invoke(REWRITE_PROMPT.format(question=question))
    rewritten_query = response.content.strip()

    print(f"[rewrite_query] Original: {question}")
    print(f"[rewrite_query] Rewritten: {rewritten_query}")
    return {"rewritten_query": rewritten_query}



# Node 4 — Web search fallback (Tavily)

def web_search(state: CRAGState) -> CRAGState:
    """
    Fallback to Tavily web search when HR knowledge base has no relevant docs.
    Uses rewritten query for better results.
    Marks web_search_used=True so UI can show the source to the user.
    """
    print("[web_search] Falling back to Tavily web search...")
    query   = state.get("rewritten_query") or state["question"]
    results = web_search_tool.invoke(query)

    web_docs = [r["content"] for r in results if "content" in r]
    print(f"[web_search] Retrieved {len(web_docs)} web results")

    return {
        "documents"     : web_docs,
        "web_search_used": True
    }


# Node 5 — Generate final answer

def generate(state: CRAGState) -> CRAGState:
    """
    Generate final answer using graded and relevant documents as context.
    Works for both HR knowledge base docs and web search fallback docs.
    Clearly states when no relevant context was found.
    """
    print("[generate] Generating final answer...")
    question  = state["question"]
    documents = state["documents"]
    source    = "web search results" if state.get("web_search_used") else "HR knowledge base"

    context = "\n\n---\n\n".join(documents) if documents else ""

    GENERATE_PROMPT = ChatPromptTemplate.from_template(
        """You are an intelligent HR policy assistant for an enterprise organization.

Answer the employee's HR question clearly and professionally.
Use the provided context from {source}.
If context is empty or insufficient, say so honestly and suggest contacting HR directly.

Context:
{context}

Employee Question: {question}

Answer:"""
    )

    chain  = GENERATE_PROMPT | llm
    result = chain.invoke({
        "question": question,
        "context" : context,
        "source"  : source
    })

    print(f"[generate] Answer generated using {source}")
    return {
        "generation": result.content,
        "messages"  : [AIMessage(content=result.content)]
    }


# Routing function — after grading decides path

def route_after_grading(state: CRAGState) -> str:
    """
    Core CRAG routing decision:
    - Relevant docs found  --> generate directly from knowledge base
    - No relevant docs     --> rewrite query --> web search --> generate

    This is what makes CRAG "corrective" — it self-corrects
    poor retrieval by falling back to web search.
    """
    if state["documents"]:
        print("[route] Relevant docs found --> generating from knowledge base")
        return "generate"
    else:
        print("[route] No relevant docs --> routing to web search fallback")
        return "rewrite_query"

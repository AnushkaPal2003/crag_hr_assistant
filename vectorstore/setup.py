from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, TOP_K_DOCS, SCORE_THRESHOLD

# Embedding model 
embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_hr_dataset() -> list:
    """
    Load HR policies Q&A dataset from HuggingFace.
    Dataset: strova-ai/hr-policies-qa-dataset
    Each row has: question, answer, context fields
    """
    from datasets import load_dataset
    print("Loading HR dataset from HuggingFace...")
    ds = load_dataset("strova-ai/hr-policies-qa-dataset")
    return ds


def prepare_documents(ds) -> list:
    """
    Convert HR dataset rows into LangChain Document format.
    Dataset structure: each row has 'messages' list with system/user/assistant roles.
    We extract user question + assistant answer and combine into page_content.
    """
    docs  = []
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    for row in tqdm(split, desc="Preparing documents"):
        messages = row.get("messages", [])

        # extract user question and assistant answer from messages list
        question = ""
        answer   = ""
        for msg in messages:
            if msg.get("role") == "user":
                question = msg.get("content", "")
            elif msg.get("role") == "assistant":
                answer = msg.get("content", "")

        # skip empty rows
        if not question and not answer:
            continue

        # combine Q + A for richer retrieval
        page_content = f"Q: {question}\nA: {answer}"

        metadata = {
            "source"  : "hr-policies-qa-dataset",
            "category": "hr_policy"
        }

        docs.append(Document(page_content=page_content, metadata=metadata))

    print(f"Prepared {len(docs)} documents.")
    return docs


def create_vectorstore() -> Chroma:
    """
    Build ChromaDB vectorstore from HR dataset.
    Run ONCE — saved to disk at CHROMA_DB_PATH.
    """
    ds   = load_hr_dataset()
    docs = prepare_documents(ds)

    print("Creating ChromaDB vectorstore...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        collection_name="hr_knowledge_base",
        embedding=embed_model,
        collection_metadata={"hnsw:space": "cosine"},  # cosine similarity
        persist_directory=CHROMA_DB_PATH
    )
    print(f"Vectorstore created with {len(docs)} documents at {CHROMA_DB_PATH}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load existing vectorstore from disk"""
    return Chroma(
        collection_name="hr_knowledge_base",
        embedding_function=embed_model,
        persist_directory=CHROMA_DB_PATH
    )


def get_retriever():
    """Get similarity-threshold retriever"""
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k"              : TOP_K_DOCS,
            "score_threshold": SCORE_THRESHOLD
        }
    )
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL      = "gpt-4o-mini" 

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector DB 
CHROMA_DB_PATH  = "./data/chroma_db"
TOP_K_DOCS      = 4
SCORE_THRESHOLD = 0.3

# Tavily Web Search 
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LangSmith Observability 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = "hr_assistant"

# FastAPI 
API_HOST = "0.0.0.0"
API_PORT = 8000

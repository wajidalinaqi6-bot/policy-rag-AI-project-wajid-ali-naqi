import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "policies"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

MODEL_NAME = "gpt-3.5-turbo"
MAX_TOKENS = 1000
TEMPERATURE = 0.3

RANDOM_SEED = 42

ALLOWED_TOPICS = [
    "pto", "paid time off", "vacation", "leave", "holiday",
    "remote work", "work from home", "hybrid",
    "security", "password", "mfa", "confidential",
    "expense", "reimbursement", "travel",
    "conduct", "ethics", "behavior",
    "benefits", "insurance", "health", "retirement", "401k",
    "performance", "review", "goals",
    "technology", "email", "internet", "software",
    "leave of absence", "fmla", "parental", "bereavement"
]

MAX_ANSWER_LENGTH = 500

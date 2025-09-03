import os
from dotenv import load_dotenv


load_dotenv()


CONFIG = {
"EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "openai"),
"LLM_PROVIDER": os.getenv("LLM_PROVIDER", "openai"),
"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
"OPENAI_EMBED_MODEL": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
"OPENAI_CHAT_MODEL": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
"HF_TOKEN": os.getenv("HF_TOKEN", ""),
"HF_EMBED_MODEL": os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
"HF_GENERATION_MODEL": os.getenv("HF_GENERATION_MODEL", "HuggingFaceH4/zephyr-7b-beta"),
"CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 1200)),
"CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 120)),
"VECTOR_DIR": os.getenv("VECTOR_DIR", "storage/faiss_index"),
"DOCS_DIR": os.getenv("DOCS_DIR", "storage/docs"),
"SUMMARY_MAX_TOKENS": int(os.getenv("SUMMARY_MAX_TOKENS", 800)),
}


os.makedirs(CONFIG["VECTOR_DIR"], exist_ok=True)
os.makedirs(CONFIG["DOCS_DIR"], exist_ok=True)
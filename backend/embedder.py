from typing import List
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
import os


def get_embedder(model: str = "openai") -> object:
    """
    Returns an embedding model.
    model = "openai" → OpenAI Embeddings (requires OPENAI_API_KEY)
    model = "hf" → HuggingFace Embeddings (all-MiniLM-L6-v2)
    """
    if model == "openai":
        return OpenAIEmbeddings()
    elif model == "hf":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError("Unknown model type. Use 'openai' or 'hf'.")


def embed_texts(texts: List[str], model: str = "openai") -> List[List[float]]:
    """
    Converts a list of texts into embeddings.
    """
    embedder = get_embedder(model)
    return embedder.embed_documents(texts)


def embed_query(query: str, model: str = "openai") -> List[float]:
    """
    Embeds a single query string.
    """
    embedder = get_embedder(model)
    return embedder.embed_query(query)

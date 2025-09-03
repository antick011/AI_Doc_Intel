import os
import faiss
import pickle
from typing import List, Dict
from pathlib import Path
from langchain_openai import OpenAIEmbeddings


VECTOR_DIR = Path("storage/faiss_index")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = VECTOR_DIR / "index.faiss"
META_FILE = VECTOR_DIR / "metas.pkl"


class VectorIndex:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.index = None
        self.metas = []

    def create(self, chunks: List[str], metas: List[Dict]):
        """
        Create a FAISS index from text chunks and metadata.
        """
        vectors = self.embeddings.embed_documents(chunks)
        dim = len(vectors[0])

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        self.metas = metas

        faiss.write_index(self.index, str(INDEX_FILE))
        with open(META_FILE, "wb") as f:
            pickle.dump(self.metas, f)

    def load(self):
        """
        Load an existing FAISS index and metadata.
        """
        if not INDEX_FILE.exists() or not META_FILE.exists():
            raise FileNotFoundError("No FAISS index found. Please build it first.")

        self.index = faiss.read_index(str(INDEX_FILE))
        with open(META_FILE, "rb") as f:
            self.metas = pickle.load(f)

    def search(self, query: str, k: int = 5):
        """
        Search the FAISS index with a query and return top-k results with metadata.
        """
        if self.index is None:
            self.load()

        q_vector = self.embeddings.embed_query(query)
        D, I = self.index.search([q_vector], k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metas):
                continue
            results.append({
                "score": float(dist),
                "doc": self.metas[idx].get("source", "unknown"),
                "preview": self.metas[idx].get("preview", "")[:200],
                "i": idx
            })

        return results

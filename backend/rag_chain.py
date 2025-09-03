import os
from typing import List, Tuple
from backend.vectorstore import VectorIndex
from backend.utils import get_embedding, get_llm_response


def answer_question(query: str, k: int = 5) -> Tuple[str, List[dict]]:
    """
    Given a query, retrieves top-k chunks from the vector store and asks the LLM to answer.
    
    Returns:
        - answer: str
        - sources: List of dicts with doc info
    """
    # Load existing vector index
    index = VectorIndex()
    if not index.exists():
        return "❌ No index found. Please upload documents and build index first.", []

    # Embed query
    q_emb = get_embedding(query)

    # Retrieve
    results = index.search(q_emb, k=k)

    if not results:
        return "❌ No relevant context found.", []

    # Prepare context
    context = "\n\n".join([r["text"] for r in results])

    # Ask LLM
    prompt = f"""You are an assistant. 
Use the following context from uploaded documents to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    answer = get_llm_response(prompt)

    return answer, results

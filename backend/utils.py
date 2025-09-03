import os
import openai
from typing import List
import numpy as np

# Load API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Default models (configurable via .env)
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for a text using OpenAI embeddings API.
    """
    if not openai.api_key:
        raise ValueError("❌ OPENAI_API_KEY is missing. Please set it in .env")

    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def get_llm_response(prompt: str) -> str:
    """
    Get a response from the OpenAI chat model.
    """
    if not openai.api_key:
        raise ValueError("❌ OPENAI_API_KEY is missing. Please set it in .env")

    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for document Q&A."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip()

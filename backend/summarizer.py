import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def summarize(text: str, mode: str = "short") -> str:
    """
    Summarizes a given text using OpenAI.
    
    :param text: The input text to summarize.
    :param mode: "short" or "detailed" (controls the summary length).
    :return: Summary string.
    """
    if not openai.api_key:
        raise ValueError("❌ OPENAI_API_KEY is missing. Please set it in .env")

    if mode == "short":
        instruction = "Summarize the following text briefly in a few sentences:"
    else:
        instruction = "Provide a detailed summary of the following text:"

    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert text summarizer."},
            {"role": "user", "content": f"{instruction}\n\n{text}"},
        ],
        temperature=0.3,
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()

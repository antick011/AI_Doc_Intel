from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Initialize a text splitter that breaks large documents into chunks.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split given text into chunks.
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks


# Example test
if __name__ == "__main__":
    sample_text = """Artificial Intelligence is a branch of computer science that 
    aims to create machines that can perform tasks that typically require 
    human intelligence. It includes problem-solving, decision-making, 
    understanding natural language, and more."""
    
    parts = split_text(sample_text, chunk_size=50, chunk_overlap=10)
    for i, chunk in enumerate(parts, 1):
        print(f"Chunk {i}:", chunk)

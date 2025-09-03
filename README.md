# 📑 AI Document Intelligence

AI-powered document intelligence system with **RAG (Retrieval-Augmented Generation)**, **summarization**, and **question answering**.  
Supports PDF/TXT ingestion, vector search with FAISS, and multiple LLM backends (OpenAI / Hugging Face).

---

## 🚀 Features
- 📂 Load and process **PDF** and **TXT** documents.
- ✂️ Automatic **chunking** and embedding generation.
- 🔎 Vector search with **FAISS**.
- 🤖 **Question answering** over documents using RAG.
- 📝 Document **summarization**.
- 🌐 Supports **OpenAI** and **Hugging Face** providers.
- 🎨 Simple **Streamlit UI** for interaction.

---

## 📦 Installation

### 1️⃣ Clone repo
```bash
git clone https://github.com/yourusername/ai_doc_intel.git
cd ai_doc_intel

2️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Linux / macOS
.venv\Scripts\activate      # On Windows

3️⃣ Install dependencies
pip install -r requirements.txt

⚙️ Configuration
Create a .env file in the project root:
# Choose providers
EMBEDDING_PROVIDER=openai   # options: openai | hf
LLM_PROVIDER=openai         # options: openai | hf_local | hf_inference

# OpenAI
OPENAI_API_KEY=your-openai-key
OPENAI_EMBED_MODEL=text-embedding-3-large
OPENAI_CHAT_MODEL=gpt-4o-mini

# Hugging Face
HF_TOKEN=your-hf-token
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_GENERATION_MODEL=HuggingFaceH4/zephyr-7b-beta

# Chunking
CHUNK_SIZE=1200
CHUNK_OVERLAP=120

# Storage
VECTOR_DIR=storage/faiss_index
DOCS_DIR=storage/docs

# Summarization
SUMMARY_MAX_TOKENS=800


▶️ Running the App
Start Streamlit UI
streamlit run frontend/app.py


📂 Project Structure
ai_doc_intel/
│── backend/
│   ├── document_loader.py   # Load PDFs / TXTs
│   ├── vectorstore.py       # FAISS vector DB
│   ├── rag_chain.py         # Retrieval + QA pipeline
│   ├── summarizer.py        # Summarization logic
│   ├── utils.py             # Embedding + LLM utils
│   └── config.py            # Loads .env configs
│
│── frontend/
│   └── app.py               # Streamlit interface
│
│── storage/
│   ├── docs/                # Uploaded docs
│   └── faiss_index/         # Vector store
│
│── .env                     # API keys & settings
│── requirements.txt
│── README.md


📌 Usage

Upload a PDF/TXT file in the UI.
The system chunks and embeds the content.
You can:
    ❓ Ask questions about the document.
    📝 Generate a summary.

🔮 Roadmap
    Support for DOCX
    Multi-document RAG
    Advanced UI with chat history
    Docker deployment

📜 License

MIT License © 2025 Antick Bhattacharjee
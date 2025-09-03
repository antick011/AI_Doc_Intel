import argparse
from pathlib import Path

from backend.document_loader import load_documents
from backend.text_splitter import split_text
from backend.vectorstore import VectorStoreHandler
from backend.utils import logger

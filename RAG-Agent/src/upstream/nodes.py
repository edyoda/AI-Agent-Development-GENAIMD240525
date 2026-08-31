import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.log_utils import get_logger

logger = get_logger(__name__)


def parse_docs(state: dict) -> dict:
    doc_path = state["doc_path"]
    logger.info(f"Parsing document: {doc_path}")

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found: {doc_path}")

    reader = PdfReader(doc_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"

    doc_name = os.path.basename(doc_path)
    logger.info(f"Extracted {len(raw_text)} characters from '{doc_name}'")

    return {"raw_text": raw_text, "doc_name": doc_name}


def chunk_docs(state: dict) -> dict:
    logger.info(f"Chunking document: {state['doc_name']}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.create_documents(
        texts=[state["raw_text"]],
        metadatas=[{"doc_name": state["doc_name"]}],
    )

    logger.info(f"Created {len(chunks)} chunks from '{state['doc_name']}'")
    return {"chunks": chunks}


def add_to_vectordb(state: dict) -> dict:
    from src.vectordb import VectorDB

    db = VectorDB()
    doc_name = state["doc_name"]

    if db.doc_exists(doc_name):
        msg = f"Document '{doc_name}' already exists in the vector database. Upload denied."
        logger.warning(msg)
        return {"message": msg}

    db.add_documents(state["chunks"])
    msg = f"Document '{doc_name}' processed and added to vector database successfully."
    logger.info(msg)
    return {"message": msg}

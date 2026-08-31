from langchain_core.tools import tool

from src.config import SIMILARITY_THRESHOLD, TOP_K
from src.log_utils import get_logger
from src.vectordb import VectorDB

logger = get_logger(__name__)


@tool
def validate_db() -> str:
    """Check if the vector database collection exists and has documents ready for querying."""
    db = VectorDB()
    if not db.collection_exists():
        return "Database collection does not exist. Please upload documents first."

    count = db.get_collection_count()
    if count == 0:
        return "Database collection exists but is empty. Please upload documents first."

    return (
        f"Database is ready. Collection '{db.collection_name}' has {count} document chunks."
    )


@tool
def retrieve_info(query: str) -> str:
    """Retrieve relevant information from the vector database based on the user query."""
    db = VectorDB()

    if not db.collection_exists():
        return "No documents found in the database. Please upload documents first."

    results = db.similarity_search(query, k=TOP_K)

    if not results:
        return "No relevant information found."

    relevant_chunks = []
    for doc, score in results:
        if score >= SIMILARITY_THRESHOLD:
            relevant_chunks.append(f"[Relevance: {score:.2f}] {doc.page_content}")
        else:
            logger.info(
                f"Skipping chunk with relevance score {score:.2f} (below threshold {SIMILARITY_THRESHOLD})"
            )

    if not relevant_chunks:
        return "No relevant information found matching your query."

    return "\n\n---\n\n".join(relevant_chunks)

import chromadb
from langchain_chroma import Chroma

from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from src.embedding import get_embedding_function
from src.log_utils import get_logger

logger = get_logger(__name__)


class VectorDB:
    def __init__(self):
        self.embedding_function = get_embedding_function()
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection_name = COLLECTION_NAME
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    def collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def doc_exists(self, doc_name: str) -> bool:
        try:
            collection = self.client.get_collection(self.collection_name)
            results = collection.get(where={"doc_name": doc_name})
            return len(results["ids"]) > 0
        except Exception:
            return False

    def add_documents(self, documents):
        self._store.add_documents(documents)
        count = self.get_collection_count()
        logger.info(
            f"Added {len(documents)} chunks. Collection '{self.collection_name}' now has {count} documents."
        )

    def similarity_search(self, query: str, k: int):
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def get_collection_count(self) -> int:
        try:
            collection = self.client.get_collection(self.collection_name)
            return collection.count()
        except Exception:
            return 0

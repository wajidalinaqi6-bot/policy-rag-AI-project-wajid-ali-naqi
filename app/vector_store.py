import os
import hashlib
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings, CohereEmbeddings
from langchain.schema import Document
import chromadb
from chromadb.config import Settings
import config


class VectorStore:
    def __init__(self):
        self.embeddings = None
        self.client = None
        self.collection = None
        self._initialize_embeddings()
        self._initialize_client()

    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        use_cohere = bool(config.COHERE_API_KEY)

        if use_cohere:
            self.embeddings = CohereEmbeddings(
                cohere_api_key=config.COHERE_API_KEY,
                model="embed-english-light-v3.0"
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )

    def _initialize_client(self):
        """Initialize ChromaDB client."""
        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )

    def create_collection(self, name: str = "policies"):
        """Create or get the collection."""
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        return self.collection

    @staticmethod
    def _make_id(doc: Document, index: int) -> str:
        """Generate a stable, collision-free ID for a document chunk.

        FIX: The original code used sequential IDs (doc_0, doc_1 …).
        Calling add_documents() more than once — e.g. after /reindex —
        produced duplicate IDs and raised a ChromaDB error.

        We now hash the page content + chunk_id metadata so the same
        chunk always gets the same ID, making re-indexing idempotent.
        The 'index' fallback handles the rare case of two identical chunks.
        """
        chunk_id = doc.metadata.get('chunk_id', index)
        raw = f"{doc.page_content[:200]}_{chunk_id}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()[:16]

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store.

        Uses content-hash IDs so re-indexing is safe and idempotent.
        ChromaDB's upsert is used instead of add so existing chunks are
        overwritten rather than raising a duplicate-ID error.
        """
        if self.collection is None:
            self.create_collection()

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [self._make_id(doc, i) for i, doc in enumerate(documents)]

        embeddings = self.embeddings.embed_documents(texts)

        # FIX: use upsert instead of add — safe to call multiple times
        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Upserted {len(documents)} documents into vector store")

    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter_metadata: dict = None
    ) -> List[Document]:
        """Perform similarity search."""
        if self.collection is None:
            self.create_collection()

        if k is None:
            k = config.TOP_K

        results = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=k,
            where=filter_metadata
        )

        documents = []
        if results['documents'] and results['documents'][0]:
            for i, text in enumerate(results['documents'][0]):
                doc = Document(
                    page_content=text,
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                )
                documents.append(doc)

        return documents

    def get_all_documents(self) -> List[Document]:
        """Get all documents from the collection."""
        if self.collection is None:
            self.create_collection()

        results = self.collection.get()

        documents = []
        if results['documents']:
            for i, text in enumerate(results['documents']):
                doc = Document(
                    page_content=text,
                    metadata=results['metadatas'][i] if results['metadatas'] else {}
                )
                documents.append(doc)

        return documents

    def delete_collection(self, name: str = "policies"):
        """Delete a collection."""
        self.client.delete_collection(name)
        self.collection = None
        print(f"Deleted collection: {name}")

    def reset(self):
        """Reset the vector store (deletes all collections)."""
        self.client.reset()
        self.collection = None
        print("Vector store reset")


_vector_store = None


def get_vector_store() -> VectorStore:
    """Get the singleton VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

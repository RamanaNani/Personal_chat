import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        """Initialize the vector store with ChromaDB."""
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        ids = [str(uuid.uuid4()) for _ in documents]
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Search for relevant documents."""
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

    def clear(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Delete the existing collection
            self.client.delete_collection(self.collection_name)
            # Create a new empty collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"Warning: Error clearing collection: {str(e)}")
            # If deletion fails, try to create a new collection with a different name
            self.collection_name = f"{self.collection_name}_{uuid.uuid4().hex[:8]}"
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

    def is_empty(self) -> bool:
        """Returns True if the collection is empty."""
        count = self.collection.count()
        return count == 0 
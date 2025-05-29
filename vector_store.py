import chromadb
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        """Initialize the vector store with ChromaDB."""
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create collection without embedding function to avoid OpenAI conflicts
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        logger.info("Vector store initialized")

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        try:
            # Generate embeddings manually
            embeddings = self.sentence_transformer.encode(documents).tolist()
            
            # Process documents in smaller batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                ids = [str(uuid.uuid4()) for _ in batch_docs]
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=batch_embeddings,
                    ids=ids
                )
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def search(self, query: str, n_results: int = 2) -> Dict[str, Any]:
        """Search for relevant documents with caching."""
        try:
            logger.info(f"Searching for query: {query}")
            # Generate query embedding manually
            query_embedding = self.sentence_transformer.encode([query]).tolist()[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            logger.info(f"Found {len(results['documents'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return {"documents": [[]], "metadatas": [[]]}

    def clear(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Delete the existing collection
            self.client.delete_collection(self.collection_name)
            # Create a new empty collection
            self.collection = self.client.create_collection(
                name=self.collection_name
            )
            # Clear the search cache
            self.search.cache_clear()
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            # If deletion fails, try to create a new collection with a different name
            self.collection_name = f"{self.collection_name}_{uuid.uuid4().hex[:8]}"
            self.collection = self.client.create_collection(
                name=self.collection_name
            )
            logger.info(f"Created new collection: {self.collection_name}")

    def is_empty(self) -> bool:
        """Returns True if the collection is empty."""
        try:
            count = self.collection.count()
            logger.info(f"Collection count: {count}")
            return count == 0
        except Exception as e:
            logger.error(f"Error checking collection count: {str(e)}")
            return True 
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Dict, Any
import uuid
from app.core.config import get_settings

settings = get_settings()

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.collection_name = settings.COLLECTION_NAME
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't"""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=Distance.COSINE
                )
            )

    async def add_documents(self, embeddings: List[List[float]], texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the vector store"""
        if metadata is None:
            metadata = [{} for _ in texts]

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": text,
                    **meta
                }
            )
            for embedding, text, meta in zip(embeddings, texts, metadata)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    async def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in search_result
        ]

    async def delete_document(self, document_id: str):
        """Delete a document from the vector store"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
        ) 
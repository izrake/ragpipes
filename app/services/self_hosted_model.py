import httpx
from typing import List, Dict, Any
import json
from app.core.config import get_settings

settings = get_settings()

class SelfHostedModelService:
    def __init__(self):
        self.model_url = settings.MODEL_URL
        self.embedding_url = settings.EMBEDDING_MODEL_URL
        self.model_name = settings.MODEL_NAME
        self.embedding_model_name = settings.EMBEDDING_MODEL_NAME
        
        # Get endpoints from settings
        self.model_endpoint = settings.MODEL_ENDPOINT
        self.embedding_endpoint = settings.EMBEDDING_ENDPOINT
        self.embedding_batch_endpoint = settings.EMBEDDING_BATCH_ENDPOINT
        
        # Prepare headers with API keys if provided
        self.llm_headers = {}
        self.embedding_headers = {}
        
        if settings.REMOTE_LLM_API_KEY:
            self.llm_headers["Authorization"] = f"Bearer {settings.REMOTE_LLM_API_KEY}"
        if settings.REMOTE_EMBEDDING_API_KEY:
            self.embedding_headers["Authorization"] = f"Bearer {settings.REMOTE_EMBEDDING_API_KEY}"

    async def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: str = None
    ) -> str:
        """Generate a response using the remote model"""
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question.
            If the context doesn't contain enough information to answer the question, say so.
            Always cite your sources when possible."""

        # Format context chunks
        context = "\n\n".join([
            f"Source {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])

        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.model_url}{self.model_endpoint}",
                json={
                    "prompt": prompt,
                    "model": self.model_name,
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "stop": ["</s>", "Human:", "Assistant:"]
                },
                headers=self.llm_headers
            )
            response.raise_for_status()
            result = response.json()
            return result["response"]

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using the remote embedding model"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.embedding_url}{self.embedding_endpoint}",
                json={
                    "text": text,
                    "model": self.embedding_model_name
                },
                headers=self.embedding_headers
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.embedding_url}{self.embedding_batch_endpoint}",
                json={
                    "texts": texts,
                    "model": self.embedding_model_name
                },
                headers=self.embedding_headers
            )
            response.raise_for_status()
            result = response.json()
            return result["embeddings"] 
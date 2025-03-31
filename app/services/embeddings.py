from openai import OpenAI
from typing import List
import tiktoken
from app.core.config import get_settings

settings = get_settings()

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL
        self.encoding = tiktoken.encoding_for_model(self.model)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text"""
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.encoding.encode(text)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + chunk_size
            chunk = tokens[start:end]
            chunks.append(self.encoding.decode(chunk))
            start = end - overlap
            
        return chunks 
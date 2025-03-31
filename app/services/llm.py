from openai import OpenAI
from typing import List, Dict, Any
from app.core.config import get_settings

settings = get_settings()

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    async def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: str = None
    ) -> str:
        """Generate a response using the LLM"""
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question.
            If the context doesn't contain enough information to answer the question, say so.
            Always cite your sources when possible."""

        # Format context chunks
        context = "\n\n".join([
            f"Source {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text"""
        response = self.client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding 
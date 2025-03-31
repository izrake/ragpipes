from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import APIKeyHeader
from typing import List
import time
from datetime import datetime
import uuid

from app.models.schemas import Document, IngestResponse, QueryRequest, QueryResponse, ErrorResponse
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm import LLMService
from app.services.self_hosted_model import SelfHostedModelService
from app.services.cache import CacheService
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()

# Initialize services
embedding_service = EmbeddingService()
vector_store = VectorStore()
cache_service = CacheService()

# Initialize model service based on configuration
if settings.MODEL_TYPE == "openai":
    model_service = LLMService()
else:
    model_service = SelfHostedModelService()

# Security
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    document: Document,
    api_key: str = Depends(verify_api_key)
):
    try:
        # Chunk the text
        chunks = embedding_service.chunk_text(
            document.text,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )

        # Get embeddings for chunks
        embeddings = []
        for chunk in chunks:
            # Check cache first
            cached_embedding = await cache_service.get_cached_embedding(chunk)
            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                if settings.MODEL_TYPE == "openai":
                    embedding = embedding_service.get_embedding(chunk)
                else:
                    embedding = await model_service.generate_embedding(chunk)
                embeddings.append(embedding)
                await cache_service.cache_embedding(chunk, embedding)

        # Add to vector store
        await vector_store.add_documents(
            embeddings=embeddings,
            texts=chunks,
            metadata=[document.metadata or {} for _ in chunks]
        )

        return IngestResponse(
            document_id=str(uuid.uuid4()),
            chunks=len(chunks),
            status="success",
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    query_request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        start_time = time.time()

        # Check cache first
        cached_response = await cache_service.get_cached_response(query_request.query)
        if cached_response:
            return QueryResponse(
                answer=cached_response["response"],
                relevant_chunks=[chunk["text"] for chunk in cached_response["context_chunks"]],
                metadata=cached_response["metadata"],
                processing_time=time.time() - start_time
            )

        # Get query embedding
        if settings.MODEL_TYPE == "openai":
            query_embedding = await model_service.generate_embedding(query_request.query)
        else:
            query_embedding = await model_service.generate_embedding(query_request.query)

        # Search vector store
        relevant_chunks = await vector_store.search(
            query_vector=query_embedding,
            top_k=query_request.top_k
        )

        # Generate response using model
        response = await model_service.generate_response(
            query=query_request.query,
            context_chunks=relevant_chunks
        )

        # Cache the response
        await cache_service.cache_response(
            query=query_request.query,
            response=response,
            context_chunks=relevant_chunks
        )

        return QueryResponse(
            answer=response,
            relevant_chunks=[chunk["text"] for chunk in relevant_chunks],
            metadata={"scores": [chunk["score"] for chunk in relevant_chunks]},
            processing_time=time.time() - start_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        ) 
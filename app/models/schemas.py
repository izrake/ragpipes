from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Document(BaseModel):
    text: str = Field(..., description="The text content to be processed")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata for the document")

class IngestResponse(BaseModel):
    document_id: str
    chunks: int
    status: str
    timestamp: datetime

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query")
    top_k: Optional[int] = Field(default=5, description="Number of relevant chunks to retrieve")

class QueryResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]
    metadata: Optional[dict] = None
    processing_time: float

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None 
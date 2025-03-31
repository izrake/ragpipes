# Real-time RAG System

A scalable and efficient Retrieval-Augmented Generation (RAG) system that handles real-time data ingestion and querying.

## Features

- Real-time document ingestion via API
- Efficient text chunking and embedding using OpenAI
- Vector storage using Qdrant
- Caching layer using Redis
- FastAPI-based REST API
- Rate limiting and API key authentication
- Modular and scalable architecture

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key
- Qdrant (will be run in Docker)
- Redis (will be run in Docker)

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
API_KEY=your_api_key_for_authentication
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ragpipes
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the required services using Docker Compose:
```bash
docker-compose up -d
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Document Ingestion
```http
POST /api/v1/ingest
Content-Type: application/json
X-API-Key: your_api_key

{
    "text": "Your document text here",
    "metadata": {
        "source": "example",
        "timestamp": "2024-02-20T12:00:00Z"
    }
}
```

### Query
```http
POST /api/v1/query
Content-Type: application/json
X-API-Key: your_api_key

{
    "query": "Your question here",
    "top_k": 5
}
```

## Architecture

The system consists of the following components:

1. **API Layer**: FastAPI application handling HTTP requests
2. **Embedding Service**: OpenAI-based text embedding generation
3. **Vector Store**: Qdrant for efficient similarity search
4. **LLM Service**: OpenAI GPT for response generation
5. **Cache Layer**: Redis for caching frequently accessed data

## Best Practices

1. **Chunking**: Documents are chunked with overlap to maintain context
2. **Caching**: Frequently accessed embeddings and responses are cached
3. **Rate Limiting**: API endpoints are rate-limited to prevent abuse
4. **Error Handling**: Comprehensive error handling and logging
5. **Security**: API key authentication for all endpoints

## Performance Considerations

1. The system uses async/await for non-blocking operations
2. Embeddings are cached to reduce API calls
3. Vector search is optimized for real-time queries
4. Response caching reduces latency for frequent queries

## Monitoring and Logging

The system includes basic logging using the `loguru` library. In production, you should:

1. Set up proper logging to a file or logging service
2. Monitor API usage and performance metrics
3. Set up alerts for errors and rate limit violations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 
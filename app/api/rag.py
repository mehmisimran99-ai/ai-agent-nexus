# AI Agent Nexus - RAG API Router
"""
RAG endpoints for document ingestion and querying
"""

import os
import tempfile
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.rag.pipeline import RAGPipeline

router = APIRouter()

# Global RAG pipeline instance
_rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


class IngestRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to ingest")
    metadatas: list[dict[str, Any]] | None = Field(
        None, description="Metadata for each text"
    )


class IngestResponse(BaseModel):
    status: str
    count: int
    ids: list[str]


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    top_k: int | None = Field(None, description="Number of results to retrieve")
    filter: dict[str, Any] | None = Field(None, description="Metadata filter")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    citations: list[dict[str, Any]]


class PersistRequest(BaseModel):
    path: str = Field(..., description="Path to persist vector index")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_texts(request: IngestRequest):
    """Ingest text documents into the knowledge base"""
    pipeline = get_rag_pipeline()
    try:
        ids = await pipeline.ingest_texts(request.texts, request.metadatas)
        return IngestResponse(status="success", count=len(ids), ids=ids)
    except Exception as e:  # noqa: BLE001 - API boundary: report any failure as HTTP 500
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),  # noqa: B008 - FastAPI dependency declaration idiom
    metadata: str | None = Form(None),
):
    """Ingest a file (PDF, TXT, MD) into the knowledge base"""
    pipeline = get_rag_pipeline()

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import json

        meta = json.loads(metadata) if metadata else {}
        ids = await pipeline.ingest_file(tmp_path, meta)
        return IngestResponse(status="success", count=len(ids), ids=ids)
    except Exception as e:  # noqa: BLE001 - API boundary: report any failure as HTTP 500
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the knowledge base"""
    pipeline = get_rag_pipeline()
    try:
        result = await pipeline.query(
            question=request.question,
            top_k=request.top_k or settings.top_k,
            filter=request.filter,
        )
        return QueryResponse(**result)
    except Exception as e:  # noqa: BLE001 - API boundary: report any failure as HTTP 500
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/persist")
async def persist_index(request: PersistRequest):
    """Persist vector index to disk"""
    pipeline = get_rag_pipeline()
    try:
        await pipeline.persist(request.path)
        return {"status": "success", "path": request.path}
    except Exception as e:  # noqa: BLE001 - API boundary: report any failure as HTTP 500
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_index(request: PersistRequest):
    """Load vector index from disk"""
    pipeline = get_rag_pipeline()
    try:
        await pipeline.load(request.path)
        return {"status": "success", "path": request.path}
    except Exception as e:  # noqa: BLE001 - API boundary: report any failure as HTTP 500
        raise HTTPException(status_code=500, detail=str(e))

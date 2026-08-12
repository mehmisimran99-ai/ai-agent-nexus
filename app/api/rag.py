# AI Agent Nexus - RAG API Router
"""
RAG endpoints for document ingestion and querying
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import tempfile
import os

from app.rag.pipeline import RAGPipeline
from app.config import settings

router = APIRouter()

# Global RAG pipeline instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


class IngestRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to ingest")
    metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Metadata for each text")


class IngestResponse(BaseModel):
    status: str
    count: int
    ids: List[str]


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    top_k: Optional[int] = Field(None, description="Number of results to retrieve")
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]


class PersistRequest(BaseModel):
    path: str = Field(..., description="Path to persist vector index")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_texts(request: IngestRequest):
    """Ingest text documents into the knowledge base"""
    pipeline = get_rag_pipeline()
    try:
        ids = await pipeline.ingest_texts(request.texts, request.metadatas)
        return IngestResponse(status="success", count=len(ids), ids=ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    """Ingest a file (PDF, TXT, MD) into the knowledge base"""
    pipeline = get_rag_pipeline()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        import json
        meta = json.loads(metadata) if metadata else {}
        ids = await pipeline.ingest_file(tmp_path, meta)
        return IngestResponse(status="success", count=len(ids), ids=ids)
    except Exception as e:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/persist")
async def persist_index(request: PersistRequest):
    """Persist vector index to disk"""
    pipeline = get_rag_pipeline()
    try:
        await pipeline.persist(request.path)
        return {"status": "success", "path": request.path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_index(request: PersistRequest):
    """Load vector index from disk"""
    pipeline = get_rag_pipeline()
    try:
        await pipeline.load(request.path)
        return {"status": "success", "path": request.path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
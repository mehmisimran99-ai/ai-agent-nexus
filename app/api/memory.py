# AI Agent Nexus - Memory API Router
"""
Conversation memory endpoints for thread persistence
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.memory.conversation import ConversationMemory

router = APIRouter()

_memory: ConversationMemory | None = None


def get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory


class ThreadCreateRequest(BaseModel):
    title: str | None = Field(None, description="Thread title")
    metadata: dict[str, Any] | None = Field(None, description="Thread metadata")


class ThreadResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    metadata: dict[str, Any]


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    timestamp: datetime
    tool_calls: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []


class ThreadDetailResponse(BaseModel):
    thread: ThreadResponse
    messages: list[MessageResponse]


@router.post("/threads", response_model=ThreadResponse)
async def create_thread(request: ThreadCreateRequest):
    """Create a new conversation thread"""
    memory = get_memory()
    thread = await memory.create_thread(
        title=request.title or f"Thread {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        metadata=request.metadata or {},
    )
    return ThreadResponse(**thread.to_dict())


@router.get("/threads", response_model=list[ThreadResponse])
async def list_threads(limit: int = 50, offset: int = 0):
    """List conversation threads"""
    memory = get_memory()
    threads = await memory.list_threads(limit=limit, offset=offset)
    return [ThreadResponse(**t.to_dict()) for t in threads]


@router.get("/threads/{thread_id}", response_model=ThreadDetailResponse)
async def get_thread(thread_id: str):
    """Get thread with full message history"""
    memory = get_memory()
    thread = await memory.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadDetailResponse(
        thread=ThreadResponse(**thread.to_dict()),
        messages=[MessageResponse(**m.to_dict()) for m in thread.messages],
    )


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a conversation thread"""
    memory = get_memory()
    success = await memory.delete_thread(thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"status": "deleted", "thread_id": thread_id}


@router.post("/threads/{thread_id}/summarize")
async def summarize_thread(thread_id: str):
    """Generate a summary of the thread"""
    memory = get_memory()
    summary = await memory.summarize_thread(thread_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"thread_id": thread_id, "summary": summary}

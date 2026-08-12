# AI Agent Nexus - Agents API Router
"""
Agent chat endpoints with streaming support
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json

from app.agents.graph import run_agent_stream, run_agent
from app.config import settings

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")
    stream: bool = Field(False, description="Enable SSE streaming")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration overrides")


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    tool_calls: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []


class ToolListResponse(BaseModel):
    tools: List[Dict[str, Any]]


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Non-streaming agent chat"""
    try:
        result = await run_agent(
            message=request.message,
            thread_id=request.thread_id,
            config=request.config,
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """SSE streaming agent chat"""
    async def event_generator():
        async for chunk in run_agent_stream(
            message=request.message,
            thread_id=request.thread_id,
            config=request.config,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/tools", response_model=ToolListResponse)
async def list_tools():
    """List available agent tools"""
    from app.tools.registry import get_tool_registry
    registry = get_tool_registry()
    return ToolListResponse(tools=registry.list_tools())


@router.post("/configure")
async def configure_agent(config: Dict[str, Any]):
    """Update agent configuration"""
    # In production, persist to database
    return {"status": "configured", "config": config}


@router.get("/status")
async def agent_status():
    """Get agent system status"""
    return {
        "status": "ready",
        "model": settings.llm_model,
        "max_iterations": settings.agent_max_iterations,
        "tool_approval_required": settings.tool_approval_required,
    }
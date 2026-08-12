# AI Agent Nexus - Main App Entry
"""
AI Agent Nexus — Multi-Agent Orchestration Framework
Entry point for FastAPI application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.config import settings
from app.api import agents, rag, memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting {settings.app_name} v{settings.version}")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Multi-agent orchestration with LangGraph, RAG, and governance",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agents.router, prefix="/agents", tags=["agents"])
app.include_router(rag.router, prefix="/rag", tags=["rag"])
app.include_router(memory.router, prefix="/memory", tags=["memory"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.app_name, "version": settings.version}


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.version,
        "description": "Multi-agent AI system with LangGraph orchestration",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)
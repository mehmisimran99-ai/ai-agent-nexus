# AI Agent Nexus - Configuration
"""
Configuration management using Pydantic Settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_name: str = "AI Agent Nexus"
    version: str = "1.0.0"
    port: int = 8001
    debug: bool = True

    # LLM
    llm_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    openai_api_key: str | None = None

    # RAG
    embedding_model: str = "nomic-embed-text"
    vector_store: str = "chromadb"  # chromadb, faiss
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_persist_dir: str = "./data/chroma"
    faiss_index_path: str = "./data/faiss/index"

    # Agent
    agent_max_iterations: int = 5
    tool_approval_required: bool = True

    # External APIs
    tavily_api_key: str | None = None

    # Evaluation
    eval_threshold: float = 0.8  # 80% pass rate for CI gate


settings = Settings()

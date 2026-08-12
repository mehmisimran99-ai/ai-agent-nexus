# AI Agent Nexus — Multi-Agent Orchestration Framework

A production-ready multi-agent system built with LangGraph, featuring planner/researcher/executor/critic agents with shared state, hybrid RAG, conversation memory, and enterprise governance patterns.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LangGraph State Machine                  │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│   Planner   │ Researcher  │  Executor   │      Critic       │
│  (decompose)│  (retrieve) │  (execute)  │   (evaluate)      │
└──────┬──────┴──────┬──────┴──────┬──────┴─────────┬────────┘
       │             │             │               │
       └─────────────┴─────────────┴───────────────┘
                         │
              ┌──────────▼──────────┐
              │   Shared State      │
              │ (messages, context, │
              │  tools, memory)     │
              └─────────────────────┘
```

## Features

- **4-Agent Orchestration**: Planner → Researcher → Executor → Critic with LangGraph state management
- **Hybrid RAG**: Vector + keyword search with reranking and citation tracking
- **Conversation Memory**: Thread persistence, summarization, context window management
- **Tool Approval Flows**: Human-in-the-loop governance for sensitive operations
- **REST API + CLI**: FastAPI backend with streaming responses
- **Evaluation Harness**: Faithfulness, relevance, groundedness metrics
- **Structured Logging**: Audit trails for compliance

## Quick Start

```bash
# Clone and setup
git clone https://github.com/mehmisimran99-ai/ai-agent-nexus.git
cd ai-agent-nexus

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Start Ollama (required for local LLMs)
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Run API server
uvicorn app.main:app --reload --port 8001

# Run CLI
python -m cli.main --help
```

## Project Structure

```
ai-agent-nexus/
├── app/
│   ├── main.py              # FastAPI app + routers
│   ├── config.py            # Pydantic settings
│   ├── api/
│   │   ├── agents.py        # Multi-agent endpoints
│   │   ├── rag.py           # RAG endpoints
│   │   └── memory.py        # Conversation memory endpoints
│   ├── agents/
│   │   ├── state.py         # Agent state definitions
│   │   ├── planner.py       # Planner agent
│   │   ├── researcher.py    # Researcher agent (RAG)
│   │   ├── executor.py      # Executor agent (tools)
│   │   ├── critic.py        # Critic agent (evaluation)
│   │   └── graph.py         # LangGraph orchestration
│   ├── rag/
│   │   ├── pipeline.py      # RAG orchestration
│   │   ├── hybrid_search.py # Vector + keyword + rerank
│   │   ├── vector_store.py  # ChromaDB/FAISS abstraction
│   │   └── embeddings.py    # Ollama/OpenAI embeddings
│   ├── memory/
│   │   ├── conversation.py  # Thread persistence
│   │   ├── summarizer.py    # Conversation summarization
│   │   └── context_window.py# Context management
│   ├── tools/
│   │   ├── registry.py      # Tool registry
│   │   ├── web_search.py    # Tavily web search
│   │   ├── code_exec.py     # Python code execution
│   │   ├── file_ops.py      # File read/write
│   │   └── approval.py      # Human-in-the-loop approval
│   └── evaluation/
│       ├── metrics.py       # Faithfulness, relevance, groundedness
│       └── harness.py       # Evaluation runner
├── cli/
│   └── main.py              # CLI interface
├── tests/
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## API Endpoints

### Agents
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agents/chat` | Non-streaming multi-agent chat |
| POST | `/agents/chat/stream` | SSE streaming chat |
| GET | `/agents/status` | Agent system status |

### RAG
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/rag/ingest` | Ingest documents |
| POST | `/rag/query` | Hybrid search query |
| POST | `/rag/persist` | Save vector index |
| POST | `/rag/load` | Load vector index |

### Memory
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/memory/threads` | List conversation threads |
| GET | `/memory/threads/{id}` | Get thread history |
| POST | `/memory/threads` | Create new thread |
| DELETE | `/memory/threads/{id}` | Delete thread |

## Configuration

Environment variables (`.env`):
```bash
# LLM
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.1

# RAG
EMBEDDING_MODEL=nomic-embed-text
VECTOR_STORE=chromadb
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=4
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Agent
AGENT_MAX_ITERATIONS=5
TOOL_APPROVAL_REQUIRED=true

# External APIs
TAVILY_API_KEY=***
OPENAI_API_KEY=***
```

## Evaluation

```bash
# Run evaluation suite
python -m app.evaluation.harness --suite full

# Or via API
curl -X POST http://localhost:8001/eval/run \
  -H "Content-Type: application/json" \
  -d '{"suite": "full"}'
```

**Metrics**: Faithfulness, Relevance, Groundedness, Tool Usage Accuracy

## Governance Features

- **Audit Logging**: All agent decisions, tool calls, and RAG retrievals logged
- **Tool Approval**: Sensitive tools require human confirmation
- **Data Lineage**: Citation tracking for RAG responses
- **Evaluation Reports**: Automated quality reports for compliance

## License

MIT — Portfolio project for demonstration purposes.
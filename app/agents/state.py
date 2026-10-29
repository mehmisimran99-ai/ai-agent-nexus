# AI Agent Nexus - Agent State Definitions
"""
Agent state and message types for LangGraph orchestration
"""

from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a tool invocation"""

    id: str
    name: str
    arguments: dict[str, Any]
    result: Any | None = None
    status: Literal["pending", "approved", "rejected", "completed", "failed"] = (
        "pending"
    )
    approved_by: str | None = None
    approved_at: datetime | None = None


class Citation(BaseModel):
    """Citation for RAG-sourced information"""

    source_id: str
    content: str
    metadata: dict[str, Any]
    score: float


class AgentMessage(BaseModel):
    """Message in the conversation"""

    id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_calls: list[ToolCall] = []
    citations: list[Citation] = []
    metadata: dict[str, Any] = {}


class AgentState(TypedDict):
    """LangGraph state for multi-agent orchestration"""

    # Conversation
    messages: list[AgentMessage]
    thread_id: str

    # Current task
    current_task: str | None
    task_type: Literal["chat", "research", "analysis", "code", "rag"] | None

    # Agent routing
    next_agent: Literal["planner", "researcher", "executor", "critic", "end"] | None
    agent_outputs: dict[str, Any]

    # Tools & RAG
    available_tools: list[str]
    tool_results: dict[str, Any]
    rag_context: list[Citation]

    # Governance
    requires_approval: bool
    approval_requests: list[ToolCall]
    audit_log: list[dict[str, Any]]

    # Memory
    conversation_summary: str | None
    context_window: list[AgentMessage]

    # Configuration
    config: dict[str, Any]

    # Error handling
    error: str | None
    retry_count: int


# Initial state factory
def create_initial_state(
    thread_id: str, config: dict[str, Any] | None = None
) -> AgentState:
    """Create initial agent state"""
    return AgentState(
        messages=[],
        thread_id=thread_id,
        current_task=None,
        task_type=None,
        next_agent="planner",
        agent_outputs={},
        available_tools=[
            "calculator",
            "python_execute",
            "file_read",
            "file_write",
            "web_search",
            "rag_query",
        ],
        tool_results={},
        rag_context=[],
        requires_approval=False,
        approval_requests=[],
        audit_log=[],
        conversation_summary=None,
        context_window=[],
        config=config or {},
        error=None,
        retry_count=0,
    )


# State update helpers
def add_message(state: AgentState, message: AgentMessage) -> AgentState:
    """Add message to state"""
    state["messages"].append(message)
    # Maintain context window (last 20 messages)
    state["context_window"] = state["messages"][-20:]
    return state


def add_audit_entry(
    state: AgentState, action: str, details: dict[str, Any]
) -> AgentState:
    """Add audit log entry"""
    state["audit_log"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
        }
    )
    return state


def set_next_agent(state: AgentState, agent: str) -> AgentState:
    """Set next agent to execute"""
    state["next_agent"] = agent
    return state


def record_agent_output(state: AgentState, agent: str, output: Any) -> AgentState:
    """Record agent output"""
    state["agent_outputs"][agent] = output
    return state

# AI Agent Nexus - Agent State Definitions
"""
Agent state and message types for LangGraph orchestration
"""

from typing import TypedDict, List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a tool invocation"""
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    status: Literal["pending", "approved", "rejected", "completed", "failed"] = "pending"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


class Citation(BaseModel):
    """Citation for RAG-sourced information"""
    source_id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class AgentMessage(BaseModel):
    """Message in the conversation"""
    id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_calls: List[ToolCall] = []
    citations: List[Citation] = []
    metadata: Dict[str, Any] = {}


class AgentState(TypedDict):
    """LangGraph state for multi-agent orchestration"""
    # Conversation
    messages: List[AgentMessage]
    thread_id: str
    
    # Current task
    current_task: Optional[str]
    task_type: Optional[Literal["chat", "research", "analysis", "code", "rag"]]
    
    # Agent routing
    next_agent: Optional[Literal["planner", "researcher", "executor", "critic", "end"]]
    agent_outputs: Dict[str, Any]
    
    # Tools & RAG
    available_tools: List[str]
    tool_results: Dict[str, Any]
    rag_context: List[Citation]
    
    # Governance
    requires_approval: bool
    approval_requests: List[ToolCall]
    audit_log: List[Dict[str, Any]]
    
    # Memory
    conversation_summary: Optional[str]
    context_window: List[AgentMessage]
    
    # Configuration
    config: Dict[str, Any]
    
    # Error handling
    error: Optional[str]
    retry_count: int


# Initial state factory
def create_initial_state(thread_id: str, config: Dict[str, Any] = None) -> AgentState:
    """Create initial agent state"""
    return AgentState(
        messages=[],
        thread_id=thread_id,
        current_task=None,
        task_type=None,
        next_agent="planner",
        agent_outputs={},
        available_tools=["calculator", "python_execute", "file_read", "file_write", "web_search", "rag_query"],
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


def add_audit_entry(state: AgentState, action: str, details: Dict[str, Any]) -> AgentState:
    """Add audit log entry"""
    state["audit_log"].append({
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details,
    })
    return state


def set_next_agent(state: AgentState, agent: str) -> AgentState:
    """Set next agent to execute"""
    state["next_agent"] = agent
    return state


def record_agent_output(state: AgentState, agent: str, output: Any) -> AgentState:
    """Record agent output"""
    state["agent_outputs"][agent] = output
    return state
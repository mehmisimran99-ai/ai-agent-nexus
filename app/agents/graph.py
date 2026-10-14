# AI Agent Nexus - LangGraph Orchestration
"""
LangGraph state machine for multi-agent orchestration
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import AgentState, create_initial_state
from app.agents.planner import run_planner
from app.agents.researcher import run_researcher
from app.agents.executor import run_executor
from app.agents.critic import run_critic
from app.rag.pipeline import RAGPipeline


# LLM placeholder - in production, use ChatOllama, ChatOpenAI, etc.
class MockLLM:
    """Mock LLM for development - replace with real LLM"""
    async def ainvoke(self, messages):
        class MockResponse:
            def __init__(self, content):
                self.content = content
        # Detect which agent is calling based on system prompt
        system = ""
        for m in messages:
            if hasattr(m, "content") and "Planner Agent" in str(m.content):
                system = "planner"
            elif hasattr(m, "content") and "Critic Agent" in str(m.content):
                system = "critic"
        if system == "critic":
            content = '{"quality_score": 0.85, "correctness": true, "completeness": true, "issues": [], "suggestions": ["Looks good"], "requires_retry": false, "next_agent": "end"}'
        else:
            content = '{"task_type": "chat", "steps": [{"agent": "executor", "action": "respond", "tools": []}], "required_tools": [], "requires_rag": false, "estimated_iterations": 1}'
        return MockResponse(content)


# Global instances
_llm = MockLLM()
_rag_pipeline = RAGPipeline()
_checkpointer = MemorySaver()


def should_continue(state: AgentState) -> Literal["planner", "researcher", "executor", "critic", "__end__"]:
    """Determine next node based on state"""
    next_agent = state.get("next_agent", "end")
    if next_agent == "end":
        return END
    return next_agent


async def planner_node(state: AgentState) -> AgentState:
    """Planner node"""
    return await run_planner(state, _llm)


async def researcher_node(state: AgentState) -> AgentState:
    """Researcher node"""
    return await run_researcher(state, _llm, _rag_pipeline)


async def executor_node(state: AgentState) -> AgentState:
    """Executor node"""
    return await run_executor(state, _llm)


async def critic_node(state: AgentState) -> AgentState:
    """Critic node"""
    return await run_critic(state, _llm)


# Build graph
def build_agent_graph() -> StateGraph:
    """Build the multi-agent LangGraph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("critic", critic_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {
            "researcher": "researcher",
            "executor": "executor",
            "critic": "critic",
            "end": END,
            END: END,
        }
    )
    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {
            "executor": "executor",
            "critic": "critic",
            "end": END,
            END: END,
        }
    )
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "critic": "critic",
            "end": END,
            END: END,
        }
    )
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "planner": "planner",
            "researcher": "researcher",
            "executor": "executor",
            "end": END,
            END: END,
        }
    )
    
    # Compile with checkpointer for persistence
    return workflow.compile(checkpointer=_checkpointer)


# Global graph instance
_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


async def run_agent(message: str, thread_id: str = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run agent synchronously (non-streaming)"""
    graph = get_agent_graph()
    
    if thread_id is None:
        import uuid
        thread_id = str(uuid.uuid4())
    
    # Create initial state
    initial_state = create_initial_state(thread_id, config or {})
    
    # Add user message
    from app.agents.state import AgentMessage, add_message
    from datetime import datetime
    user_msg = AgentMessage(
        id=f"msg_{len(initial_state['messages'])}",
        role="user",
        content=message,
        timestamp=datetime.now(),
    )
    initial_state = add_message(initial_state, user_msg)
    
    # Run graph
    thread_config = {"configurable": {"thread_id": thread_id}}
    final_state = await graph.ainvoke(initial_state, thread_config)
    
    # Extract response
    response = ""
    tool_calls = []
    citations = []
    
    for msg in final_state["messages"]:
        if msg.role == "assistant":
            response = msg.content
            tool_calls = [tc.model_dump() for tc in msg.tool_calls]
            citations = [c.model_dump() for c in msg.citations]
    
    return {
        "response": response or "Task completed",
        "thread_id": thread_id,
        "tool_calls": tool_calls,
        "citations": citations,
    }


async def run_agent_stream(message: str, thread_id: str = None, config: Dict[str, Any] = None):
    """Run agent with streaming updates"""
    graph = get_agent_graph()
    
    if thread_id is None:
        import uuid
        thread_id = str(uuid.uuid4())
    
    initial_state = create_initial_state(thread_id, config or {})
    
    from app.agents.state import AgentMessage, add_message
    from datetime import datetime
    user_msg = AgentMessage(
        id=f"msg_{len(initial_state['messages'])}",
        role="user",
        content=message,
        timestamp=datetime.now(),
    )
    initial_state = add_message(initial_state, user_msg)
    
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    # Stream events
    async for event in graph.astream(initial_state, thread_config, stream_mode="values"):
        # Yield state updates
        state = event
        yield {
            "thread_id": thread_id,
            "current_agent": state.get("next_agent"),
            "task_type": state.get("task_type"),
            "messages_count": len(state["messages"]),
        }
    
    # Final result
    final_state = await graph.ainvoke(initial_state, thread_config)
    response = ""
    for msg in final_state["messages"]:
        if msg.role == "assistant":
            response = msg.content
            break
    
    yield {
        "thread_id": thread_id,
        "response": response or "Task completed",
        "done": True,
    }

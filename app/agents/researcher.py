# AI Agent Nexus - Researcher Agent
"""
Researcher Agent: Retrieves information via RAG and web search
"""

from langchain_core.prompts import ChatPromptTemplate

from app.agents.state import AgentState, Citation
from app.rag.pipeline import RAGPipeline
from app.tools.registry import get_tool_registry


RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Researcher Agent. Your job is to gather relevant information for the user's query using available tools.

Available tools:
- rag_query: Query the internal knowledge base
- web_search: Search the web for current information

Use RAG for internal documents, web search for current events/external info.
Return findings as structured citations."""),
    ("human", "Task: {task}\n\nAvailable context: {context}\n\nPlan: {plan}"),
])


async def run_researcher(state: AgentState, llm, rag_pipeline: RAGPipeline) -> AgentState:
    """Execute researcher agent"""
    plan = state["agent_outputs"].get("planner", {})
    task = state["current_task"]
    
    # Determine research strategy
    citations = []
    
    # RAG query if needed
    if plan.get("requires_rag") or "rag_query" in state["available_tools"]:
        rag_result = await rag_pipeline.query(task, top_k=5)
        for source in rag_result.get("sources", []):
            citations.append(Citation(
                source_id=source.get("id", ""),
                content=source.get("content", "")[:500],
                metadata=source.get("metadata", {}),
                score=source.get("score", 0.0),
            ))
    
    # Web search if needed
    if "web_search" in state["available_tools"]:
        registry = get_tool_registry()
        web_tool = registry.get_tool("web_search")
        if web_tool:
            search_result = await web_tool.ainvoke({"query": task})
            # Parse and add as citations
            citations.append(Citation(
                source_id="web_search",
                content=str(search_result)[:500],
                metadata={"source": "web_search", "query": task},
                score=0.8,
            ))
    
    # Update state
    state["rag_context"] = citations
    state["next_agent"] = "executor"
    
    return state
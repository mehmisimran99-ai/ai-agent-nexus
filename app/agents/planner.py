# AI Agent Nexus - Planner Agent
"""
Planner Agent: Decomposes user requests into executable tasks
"""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from app.agents.state import AgentState


class Plan(BaseModel):
    """Task decomposition plan"""
    task_type: str = Field(description="Type of task: chat, research, analysis, code, rag")
    steps: List[Dict[str, Any]] = Field(description="Ordered steps to execute")
    required_tools: List[str] = Field(description="Tools needed for this plan")
    requires_rag: bool = Field(description="Whether RAG retrieval is needed")
    estimated_iterations: int = Field(description="Estimated number of agent iterations")


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Planner Agent in a multi-agent AI system. Your job is to analyze the user's request and create an execution plan.

Available agents:
- planner: You (decompose tasks)
- researcher: Retrieves information via RAG and web search
- executor: Executes code, calculations, file operations
- critic: Evaluates results for quality and correctness

Available tools: calculator, python_execute, file_read, file_write, web_search, rag_query

Analyze the request and output a JSON plan with:
- task_type: one of [chat, research, analysis, code, rag]
- steps: ordered list of {{\"agent\": \"agent_name\", \"action\": \"description\", \"tools\": [\"tool1\", \"tool2\"]}}
- required_tools: list of tool names needed
- requires_rag: boolean
- estimated_iterations: integer (1-10)"""),
    ("human", "User request: {message}\n\nConversation context: {context}"),
])


async def run_planner(state: AgentState, llm) -> AgentState:
    """Execute planner agent"""
    # Get recent context
    context_messages = state["context_window"][-5:] if state["context_window"] else []
    context = "\n".join([f"{m.role}: {m.content}" for m in context_messages])
    
    # Create prompt
    prompt = PLANNER_PROMPT.format_messages(
        message=state["messages"][-1].content if state["messages"] else "",
        context=context or "No prior context",
    )
    
    # Get LLM response
    parser = JsonOutputParser(pydantic_object=Plan)
    response = await llm.ainvoke(prompt)
    plan_dict = parser.parse(response.content)
    plan = Plan(**plan_dict) if isinstance(plan_dict, dict) else plan_dict
    
    # Update state
    state["current_task"] = state["messages"][-1].content if state["messages"] else ""
    state["task_type"] = plan.task_type
    state["available_tools"] = plan.required_tools
    state["requires_approval"] = any(
        tool in ["file_write", "web_search"] for tool in plan.required_tools
    )
    
    # Set next agent based on first step
    if plan.steps:
        state["next_agent"] = plan.steps[0]["agent"]
    else:
        state["next_agent"] = "executor"
    
    # Record plan
    state = record_agent_output(state, "planner", plan.model_dump())
    
    return state


def record_agent_output(state: AgentState, agent: str, output: Any) -> AgentState:
    state["agent_outputs"][agent] = output
    return state

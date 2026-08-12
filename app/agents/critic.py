# AI Agent Nexus - Critic Agent
"""
Critic Agent: Evaluates results for quality, correctness, and completeness
"""

from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from app.agents.state import AgentState


class Critique(BaseModel):
    """Critic evaluation output"""
    quality_score: float = Field(description="Overall quality 0-1", ge=0, le=1)
    correctness: bool = Field(description="Is the result factually correct?")
    completeness: bool = Field(description="Does it fully address the task?")
    issues: List[str] = Field(description="Identified issues or gaps")
    suggestions: List[str] = Field(description="Improvement suggestions")
    requires_retry: bool = Field(description="Should the task be retried?")
    next_agent: str = Field(description="Next agent: end, planner, researcher, executor")


CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Critic Agent. Evaluate the results of the agent execution.

Assess:
1. Quality: Overall quality of the response (0-1)
2. Correctness: Factual accuracy
3. Completeness: Fully addresses the user's request
4. Issues: Any problems, hallucinations, or gaps
5. Suggestions: How to improve

If quality < 0.7 or major issues exist, set requires_retry=true and suggest next_agent.
If good, set next_agent="end"."""),
    ("human", """Original task: {task}
Plan: {plan}
Tool results: {tool_results}
RAG context: {rag_context}
Current response: {response}"""),
])


async def run_critic(state: AgentState, llm) -> AgentState:
    """Execute critic agent"""
    task = state["current_task"]
    plan = state["agent_outputs"].get("planner", {})
    tool_results = state["tool_results"]
    rag_context = [c.content for c in state["rag_context"]]
    
    # Get the latest assistant message as response
    response = ""
    for msg in reversed(state["messages"]):
        if msg.role == "assistant":
            response = msg.content
            break
    
    prompt = CRITIC_PROMPT.format_messages(
        task=task,
        plan=str(plan),
        tool_results=str(tool_results),
        rag_context="\n".join(rag_context) if rag_context else "None",
        response=response or "No response generated yet",
    )
    
    parser = JsonOutputParser(pydantic_object=Critique)
    result = await llm.ainvoke(prompt)
    critique = parser.parse(result.content)
    
    # Update state based on critique
    if critique.requires_retry and state["retry_count"] < 3:
        state["retry_count"] += 1
        state["next_agent"] = critique.next_agent
        state["error"] = "; ".join(critique.issues)
    else:
        state["next_agent"] = "end"
    
    # Record critique
    state["agent_outputs"]["critic"] = critique.model_dump()
    
    return state
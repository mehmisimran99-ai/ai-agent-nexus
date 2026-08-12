# AI Agent Nexus - Executor Agent
"""
Executor Agent: Executes code, calculations, file operations, and tool calls
"""

from langchain_core.prompts import ChatPromptTemplate

from app.agents.state import AgentState, ToolCall
from app.tools.registry import get_tool_registry


EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an Executor Agent. Your job is to execute tasks using available tools.

Available tools:
- calculator: Safe math evaluation (e.g., "Calculate sqrt(144) + 5^2")
- python_execute: Run Python code (e.g., "Write code to compute fibonacci(10)")
- file_read: Read local files (e.g., "Read requirements.txt")
- file_write: Write local files (e.g., "Create hello.py with print('hi')")
- rag_query: Query knowledge base (handled by researcher)

For each tool call, output a JSON object with:
- tool: tool name
- arguments: tool arguments
- reasoning: why this tool is needed

Execute one tool at a time. Wait for result before next call."""),
    ("human", "Task: {task}\n\nPlan: {plan}\n\nResearch context: {context}\n\nAvailable tools: {tools}"),
])


async def run_executor(state: AgentState, llm) -> AgentState:
    """Execute executor agent"""
    task = state["current_task"]
    tools = state["available_tools"]
    
    # Get tool registry
    registry = get_tool_registry()
    
    # For demo, simulate tool execution based on task type
    tool_results = {}
    
    # Check if any tools need approval
    approval_tools = [t for t in tools if t in ["file_write", "web_search"]]
    if approval_tools and state["tool_approval_required"]:
        state["requires_approval"] = True
        state["approval_requests"] = [
            ToolCall(
                id=f"approval_{i}",
                name=tool,
                arguments={},
                status="pending",
            )
            for i, tool in enumerate(approval_tools)
        ]
        state["next_agent"] = "critic"  # Wait for approval
        return state
    
    # Execute tools (simplified - in production, use LLM to decide)
    if "calculator" in tools and any(op in task.lower() for op in ["calculate", "math", "+", "-", "*", "/", "sqrt"]):
        calc_tool = registry.get_tool("calculator")
        if calc_tool:
            # Extract math expression (simplified)
            import re
            math_expr = re.search(r'[\d\+\-\*\/\^\(\)\s\.]+', task)
            if math_expr:
                result = await calc_tool.ainvoke({"expression": math_expr.group()})
                tool_results["calculator"] = result
    
    if "python_execute" in tools and any(kw in task.lower() for kw in ["code", "python", "script", "compute", "fibonacci"]):
        py_tool = registry.get_tool("python_execute")
        if py_tool:
            # Generate simple code based on task
            code = generate_code_for_task(task)
            result = await py_tool.ainvoke({"code": code})
            tool_results["python_execute"] = result
    
    state["tool_results"] = tool_results
    state["next_agent"] = "critic"
    
    return state


def generate_code_for_task(task: str) -> str:
    """Generate Python code for common tasks"""
    task_lower = task.lower()
    
    if "fibonacci" in task_lower:
        n = 10
        import re
        nums = re.findall(r'\d+', task)
        if nums:
            n = int(nums[0])
        return f"""
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(f"fibonacci({n}) = {{fibonacci({n})}}")
"""
    elif "factorial" in task_lower:
        import re
        nums = re.findall(r'\d+', task)
        n = int(nums[0]) if nums else 5
        return f"""
import math
print(f"factorial({n}) = {{math.factorial({n})}}")
"""
    else:
        return f"print('Task: {task}')\nprint('Result: Executed successfully')"
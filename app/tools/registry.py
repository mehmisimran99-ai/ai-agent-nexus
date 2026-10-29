# AI Agent Nexus - Tool Registry
"""
Tool registry for agent tools with approval flows
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Base tool interface"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass

    @abstractmethod
    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        pass

    def invoke(self, arguments: dict[str, Any]) -> Any:
        return asyncio.run(self.ainvoke(arguments))


class CalculatorTool(BaseTool):
    """Safe calculator tool"""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate mathematical expressions safely"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate",
                }
            },
            "required": ["expression"],
        }

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        import ast
        import operator

        expression = arguments["expression"]

        # Safe evaluation using AST
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def eval_expr(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = eval_expr(node.left)
                right = eval_expr(node.right)
                return allowed_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_expr(node.operand)
                return allowed_operators[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported operation: {type(node)}")  # noqa: TRY004 - ValueError is this tool's error contract

        try:
            tree = ast.parse(expression, mode="eval")
            result = eval_expr(tree.body)
            return {"result": result, "expression": expression}
        except Exception as e:  # noqa: BLE001 - tool boundary: failures are returned as error dicts
            return {"error": str(e), "expression": expression}


class PythonExecuteTool(BaseTool):
    """Python code execution tool"""

    @property
    def name(self) -> str:
        return "python_execute"

    @property
    def description(self) -> str:
        return "Execute Python code in a sandboxed environment"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        }

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        code = arguments["code"]

        # In production: use secure sandbox (Docker, gVisor, etc.)
        # For demo: restricted exec
        try:
            # Capture stdout
            import io
            from contextlib import redirect_stdout

            stdout_capture = io.StringIO()

            # Restricted globals
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "sum": sum,
                    "max": max,
                    "min": min,
                    "abs": abs,
                    "round": round,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                },
                "math": __import__("math"),
                "random": __import__("random"),
                "datetime": __import__("datetime"),
                "json": __import__("json"),
                "re": __import__("re"),
            }

            with redirect_stdout(stdout_capture):
                exec(code, safe_globals)  # noqa: S102 - sandboxed exec is the purpose of this tool

            output = stdout_capture.getvalue()
            return {"output": output, "success": True}

        except Exception as e:  # noqa: BLE001 - tool boundary: failures are returned as error dicts
            return {"error": str(e), "success": False}


class FileReadTool(BaseTool):
    """File read tool"""

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read contents of a local file"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"],
        }

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        import os

        path = arguments["path"]

        # Security: restrict to allowed directories
        allowed_dirs = ["./data", "./workspace", "./output"]
        if not any(
            os.path.abspath(path).startswith(os.path.abspath(d)) for d in allowed_dirs
        ):
            return {"error": "Access denied: path not in allowed directories"}

        try:
            with open(path, "r") as f:  # noqa: ASYNC230 - small local file read
                content = f.read()
            return {"content": content, "path": path}
        except Exception as e:  # noqa: BLE001 - tool boundary: failures are returned as error dicts
            return {"error": str(e), "path": path}


class FileWriteTool(BaseTool):
    """File write tool (requires approval)"""

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "Write content to a local file (requires approval)"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        import os

        path = arguments["path"]
        content = arguments["content"]

        # Security check
        allowed_dirs = ["./data", "./workspace", "./output"]
        if not any(
            os.path.abspath(path).startswith(os.path.abspath(d)) for d in allowed_dirs
        ):
            return {"error": "Access denied: path not in allowed directories"}

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:  # noqa: ASYNC230 - small local file write
                f.write(content)
            return {"status": "written", "path": path, "bytes": len(content)}
        except Exception as e:  # noqa: BLE001 - tool boundary: failures are returned as error dicts
            return {"error": str(e), "path": path}


class WebSearchTool(BaseTool):
    """Web search tool using Tavily"""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        import httpx

        from app.config import settings

        query = arguments["query"]
        max_results = arguments.get("max_results", 5)

        if not settings.tavily_api_key:
            return {"error": "Tavily API key not configured", "results": []}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": settings.tavily_api_key,
                        "query": query,
                        "max_results": max_results,
                        "include_domains": [],
                        "exclude_domains": [],
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0),
                    }
                    for r in data.get("results", [])
                ]

                return {"results": results, "query": query}
        except Exception as e:  # noqa: BLE001 - tool boundary: failures are returned as error dicts
            return {"error": str(e), "results": []}


class RAGQueryTool(BaseTool):
    """RAG query tool"""

    @property
    def name(self) -> str:
        return "rag_query"

    @property
    def description(self) -> str:
        return "Query the internal knowledge base"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question to ask the knowledge base",
                },
                "top_k": {"type": "integer", "default": 4},
            },
            "required": ["question"],
        }

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        from app.rag.pipeline import RAGPipeline

        question = arguments["question"]
        top_k = arguments.get("top_k", 4)

        pipeline = RAGPipeline()
        result = await pipeline.query(question, top_k=top_k)

        return result


# Tool registry
_tool_registry: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool):
    """Register a tool"""
    _tool_registry[tool.name] = tool


def get_tool_registry() -> "ToolRegistry":
    return ToolRegistry()


class ToolRegistry:
    """Tool registry for managing available tools"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._initialize_default_tools()

    def _initialize_default_tools(self):
        """Register default tools"""
        self.register(CalculatorTool())
        self.register(PythonExecuteTool())
        self.register(FileReadTool())
        self.register(FileWriteTool())
        self.register(WebSearchTool())
        self.register(RAGQueryTool())

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def get_tools_for_agent(self, tool_names: list[str]) -> list[BaseTool]:
        return [self._tools[name] for name in tool_names if name in self._tools]

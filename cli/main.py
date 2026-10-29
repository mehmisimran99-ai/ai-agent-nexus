# AI Agent Nexus - CLI Interface
"""
CLI interface for AI Agent Nexus
"""

import argparse
import asyncio
import sys

from app.agents.graph import run_agent, run_agent_stream
from app.rag.pipeline import RAGPipeline


async def chat_command(args):
    """Interactive chat with the agent"""
    thread_id = args.thread_id
    print(f"Starting chat session (thread: {thread_id or 'new'})")
    print("Type 'exit' to quit, 'stream' to toggle streaming\n")

    stream_mode = args.stream

    while True:
        try:
            message = input("You: ").strip()
            if message.lower() in ("exit", "quit"):
                break
            if message.lower() == "stream":
                stream_mode = not stream_mode
                print(f"Streaming mode: {'ON' if stream_mode else 'OFF'}")
                continue
            if not message:
                continue

            if stream_mode:
                print("Agent: ", end="", flush=True)
                async for chunk in run_agent_stream(
                    message=message, thread_id=thread_id
                ):
                    if chunk.get("done"):
                        print(chunk.get("response", ""))
                    else:
                        # Show agent status
                        agent = chunk.get("current_agent")
                        if agent and agent != "end":
                            print(f"[{agent}]", end=" ", flush=True)
                print()
            else:
                result = await run_agent(message=message, thread_id=thread_id)
                print(f"Agent: {result['response']}")
                if result.get("tool_calls"):
                    print(
                        f"  Tools used: {[tc['name'] for tc in result['tool_calls']]}"
                    )
                if thread_id is None:
                    thread_id = result["thread_id"]

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("\nGoodbye!")


async def rag_ingest_command(args):
    """Ingest documents into RAG"""
    pipeline = RAGPipeline()

    if args.file:
        # Ingest file
        import json

        meta = json.loads(args.metadata) if args.metadata else {}
        ids = await pipeline.ingest_file(args.file, meta)
        print(f"Ingested {len(ids)} chunks from {args.file}")
    elif args.text:
        # Ingest text
        ids = await pipeline.ingest_texts(
            [args.text], [args.metadata] if args.metadata else None
        )
        print("Ingested 1 text chunk")
    else:
        print("Error: --file or --text required")
        sys.exit(1)


async def rag_query_command(args):
    """Query RAG knowledge base"""
    pipeline = RAGPipeline()
    result = await pipeline.query(args.question, top_k=args.top_k)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")
    for i, src in enumerate(result["sources"]):
        print(f"  {i + 1}. {src['content'][:100]}... (score: {src['score']:.2f})")


async def eval_command(args):
    """Run evaluation suite"""
    from app.evaluation.harness import run_suite, save_run_results

    suite_name = args.suite
    threshold = args.threshold

    print(f"Running evaluation suite: {suite_name}")
    print(f"Pass threshold: {threshold:.0%}")
    print("-" * 50)

    run = await run_suite(suite_name, threshold)

    json_path, md_path = save_run_results(run)

    print("-" * 50)
    print(
        f"Results: {run.passed_cases}/{run.total_cases} passed ({run.overall_score:.2%})"
    )
    print(f"Threshold: {'PASSED' if run.passed_threshold else 'FAILED'}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")

    sys.exit(0 if run.passed_threshold else 1)


async def serve_command(args):
    """Start API server"""
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(description="AI Agent Nexus CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with agent")
    chat_parser.add_argument("--thread-id", help="Conversation thread ID")
    chat_parser.add_argument(
        "--stream", action="store_true", help="Enable streaming mode"
    )

    # RAG commands
    rag_parser = subparsers.add_parser("rag", help="RAG operations")
    rag_sub = rag_parser.add_subparsers(dest="rag_command")

    ingest_parser = rag_sub.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--file", help="File path to ingest")
    ingest_parser.add_argument("--text", help="Text to ingest")
    ingest_parser.add_argument("--metadata", help="JSON metadata")

    query_parser = rag_sub.add_parser("query", help="Query knowledge base")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=4, help="Number of results")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation suite")
    eval_parser.add_argument(
        "--suite", default="full", choices=["math", "reasoning", "tools", "full"]
    )
    eval_parser.add_argument(
        "--threshold", type=float, default=0.8, help="Pass threshold"
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8001, help="Port to bind")
    serve_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route commands
    if args.command == "chat":
        asyncio.run(chat_command(args))
    elif args.command == "rag":
        if args.rag_command == "ingest":
            asyncio.run(rag_ingest_command(args))
        elif args.rag_command == "query":
            asyncio.run(rag_query_command(args))
        else:
            rag_parser.print_help()
    elif args.command == "eval":
        asyncio.run(eval_command(args))
    elif args.command == "serve":
        serve_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

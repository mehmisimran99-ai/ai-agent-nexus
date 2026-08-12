# AI Agent Nexus - Evaluation Harness
"""
Evaluation harness for running test suites and CI gates
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
import json
import asyncio
from datetime import datetime

from app.evaluation.metrics import (
    run_evaluation_suite,
)
from app.agents.graph import run_agent


@dataclass
class TestCase:
    """Single test case"""
    id: str
    name: str
    question: str
    expected_answer: str = ""
    expected_substrings: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    category: str = "general"


@dataclass
class TestSuite:
    """Collection of test cases"""
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)


@dataclass
class EvaluationRun:
    """Results of an evaluation run"""
    run_id: str
    suite_name: str
    timestamp: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    overall_score: float
    passed_threshold: bool
    case_results: List[Dict[str, Any]] = field(default_factory=list)


# Test suites
MATH_SUITE = TestSuite(
    name="math",
    description="Arithmetic, calculus, algebra evaluation",
    test_cases=[
        TestCase(
            id="math_001",
            name="Basic arithmetic",
            question="Calculate 15 * 23 + 100",
            expected_substrings=["445"],
            expected_tools=["calculator"],
            category="math",
        ),
        TestCase(
            id="math_002",
            name="Square root",
            question="Calculate sqrt(144) + 5^2",
            expected_substrings=["37"],
            expected_tools=["calculator"],
            category="math",
        ),
        TestCase(
            id="math_003",
            name="Power calculation",
            question="What is 2 to the power of 10?",
            expected_substrings=["1024"],
            expected_tools=["calculator"],
            category="math",
        ),
        TestCase(
            id="math_004",
            name="Fibonacci via code",
            question="Write Python code to compute fibonacci(10)",
            expected_substrings=["55", "fibonacci"],
            expected_tools=["python_execute"],
            category="math",
        ),
    ],
)


REASONING_SUITE = TestSuite(
    name="reasoning",
    description="Logic and general knowledge reasoning",
    test_cases=[
        TestCase(
            id="reasoning_001",
            name="General knowledge",
            question="What is the capital of Canada?",
            expected_substrings=["Ottawa"],
            category="reasoning",
        ),
        TestCase(
            id="reasoning_002",
            name="Logical reasoning",
            question="If all A are B, and all B are C, are all A C?",
            expected_substrings=["yes", "true"],
            category="reasoning",
        ),
    ],
)


TOOLS_SUITE = TestSuite(
    name="tools",
    description="Tool usage verification",
    test_cases=[
        TestCase(
            id="tools_001",
            name="Calculator tool usage",
            question="Calculate 25 * 4",
            expected_tools=["calculator"],
            expected_substrings=["100"],
            category="tools",
        ),
        TestCase(
            id="tools_002",
            name="Python execution tool",
            question="Write code to compute factorial of 5",
            expected_tools=["python_execute"],
            expected_substrings=["120"],
            category="tools",
        ),
        TestCase(
            id="tools_003",
            name="File read tool",
            question="Read the requirements.txt file",
            expected_tools=["file_read"],
            category="tools",
        ),
    ],
)


FULL_SUITE = TestSuite(
    name="full",
    description="Complete evaluation suite",
    test_cases=MATH_SUITE.test_cases + REASONING_SUITE.test_cases + TOOLS_SUITE.test_cases,
)


SUITES = {
    "math": MATH_SUITE,
    "reasoning": REASONING_SUITE,
    "tools": TOOLS_SUITE,
    "full": FULL_SUITE,
}


async def run_test_case(test_case: TestCase, thread_id: str = None) -> Dict[str, Any]:
    """Run a single test case"""
    result = await run_agent(
        message=test_case.question,
        thread_id=thread_id or f"eval_{test_case.id}",
    )
    
    # Run evaluations
    eval_results = await run_evaluation_suite(
        response=result.get("response", ""),
        question=test_case.question,
        context=test_case.context,
        citations=result.get("citations", []),
        tool_calls=result.get("tool_calls", []),
        expected_tools=test_case.expected_tools,
        expected_answer=test_case.expected_answer,
        expected_substrings=test_case.expected_substrings,
    )
    
    # Determine if passed
    passed = all(r.passed for r in eval_results)
    overall_score = sum(r.score for r in eval_results) / len(eval_results) if eval_results else 0
    
    return {
        "test_case": test_case.id,
        "name": test_case.name,
        "question": test_case.question,
        "response": result.get("response", ""),
        "passed": passed,
        "score": overall_score,
        "evaluations": [
            {
                "metric": r.metric,
                "score": r.score,
                "passed": r.passed,
                "details": r.details,
            }
            for r in eval_results
        ],
    }


async def run_suite(suite_name: str, threshold: float = 0.8) -> EvaluationRun:
    """Run a complete test suite"""
    suite = SUITES.get(suite_name)
    if not suite:
        raise ValueError(f"Unknown suite: {suite_name}")
    
    run_id = f"eval_{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    case_results = []
    passed_count = 0
    
    for test_case in suite.test_cases:
        print(f"Running {test_case.id}: {test_case.name}...")
        result = await run_test_case(test_case)
        case_results.append(result)
        if result["passed"]:
            passed_count += 1
    
    total = len(suite.test_cases)
    overall_score = passed_count / total if total > 0 else 0
    passed_threshold = overall_score >= threshold
    
    return EvaluationRun(
        run_id=run_id,
        suite_name=suite_name,
        timestamp=datetime.now().isoformat(),
        total_cases=total,
        passed_cases=passed_count,
        failed_cases=total - passed_count,
        overall_score=overall_score,
        passed_threshold=passed_threshold,
        case_results=case_results,
    )


def save_run_results(run: EvaluationRun, output_dir: str = "./eval_results"):
    """Save evaluation results to disk"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON results
    json_path = os.path.join(output_dir, f"{run.run_id}.json")
    with open(json_path, "w") as f:
        json.dump({
            "run_id": run.run_id,
            "suite_name": run.suite_name,
            "timestamp": run.timestamp,
            "total_cases": run.total_cases,
            "passed_cases": run.passed_cases,
            "failed_cases": run.failed_cases,
            "overall_score": run.overall_score,
            "passed_threshold": run.passed_threshold,
            "case_results": run.case_results,
        }, f, indent=2)
    
    # Markdown summary
    md_path = os.path.join(output_dir, f"{run.run_id}.md")
    with open(md_path, "w") as f:
        f.write(f"# Evaluation Run: {run.run_id}\n\n")
        f.write(f"**Suite:** {run.suite_name}\n")
        f.write(f"**Timestamp:** {run.timestamp}\n")
        f.write(f"**Overall Score:** {run.overall_score:.2%}\n")
        f.write(f"**Threshold (80%):** {'✅ PASSED' if run.passed_threshold else '❌ FAILED'}\n\n")
        
        f.write("## Test Results\n\n")
        f.write("| Test | Status | Score |\n")
        f.write("|------|--------|-------|\n")
        
        for case in run.case_results:
            status = "✅ PASS" if case["passed"] else "❌ FAIL"
            f.write(f"| {case['name']} | {status} | {case['score']:.2%} |\n")
        
        f.write("\n## Details\n\n")
        for case in run.case_results:
            f.write(f"### {case['name']} ({case['test_case']})\n")
            f.write(f"**Question:** {case['question']}\n")
            f.write(f"**Response:** {case['response'][:200]}...\n")
            f.write("**Evaluations:**\n")
            for eval in case["evaluations"]:
                status = "✅" if eval["passed"] else "❌"
                f.write(f"- {status} {eval['metric']}: {eval['score']:.2%}\n")
            f.write("\n")
    
    return json_path, md_path


async def main():
    """CLI entry point"""
    import sys
    
    suite_name = sys.argv[1] if len(sys.argv) > 1 else "full"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    
    print(f"Running evaluation suite: {suite_name}")
    print(f"Pass threshold: {threshold:.0%}")
    print("-" * 50)
    
    run = await run_suite(suite_name, threshold)
    
    json_path, md_path = save_run_results(run)
    
    print("-" * 50)
    print(f"Results: {run.passed_cases}/{run.total_cases} passed ({run.overall_score:.2%})")
    print(f"Threshold: {'PASSED' if run.passed_threshold else 'FAILED'}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    
    # Exit code for CI
    sys.exit(0 if run.passed_threshold else 1)


if __name__ == "__main__":
    asyncio.run(main())
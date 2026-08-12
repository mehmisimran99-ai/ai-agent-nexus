# AI Agent Nexus - Evaluation Metrics
"""
Evaluation metrics for agent quality: faithfulness, relevance, groundedness
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class EvaluationResult:
    """Result of a single evaluation"""
    metric: str
    score: float  # 0-1
    passed: bool
    details: Dict[str, Any]


class BaseEvaluator(ABC):
    """Base evaluator interface"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    async def evaluate(self, **kwargs) -> EvaluationResult:
        pass


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluates if response is faithful to retrieved context"""
    
    @property
    def name(self) -> str:
        return "faithfulness"
    
    async def evaluate(self, response: str, context: List[str], **kwargs) -> EvaluationResult:
        """
        Faithfulness: Does the response contain only information from the context?
        Score based on: claims in response that are supported by context / total claims
        """
        # In production: use LLM-as-judge or NLI model
        # For demo: simple heuristic
        
        if not context:
            return EvaluationResult(
                metric="faithfulness",
                score=0.0,
                passed=False,
                details={"reason": "No context provided"},
            )
        
        # Simple keyword overlap heuristic
        response_words = set(response.lower().split())
        context_words = set(" ".join(context).lower().split())
        
        if not response_words:
            return EvaluationResult(
                metric="faithfulness",
                score=1.0,
                passed=True,
                details={"reason": "Empty response"},
            )
        
        overlap = len(response_words & context_words)
        score = overlap / len(response_words)
        
        return EvaluationResult(
            metric="faithfulness",
            score=score,
            passed=score >= 0.7,
            details={
                "response_word_count": len(response_words),
                "context_word_count": len(context_words),
                "overlap_count": overlap,
            },
        )


class RelevanceEvaluator(BaseEvaluator):
    """Evaluates if response is relevant to the question"""
    
    @property
    def name(self) -> str:
        return "relevance"
    
    async def evaluate(self, response: str, question: str, **kwargs) -> EvaluationResult:
        """
        Relevance: Does the response answer the question?
        """
        # In production: use LLM-as-judge or embedding similarity
        # For demo: keyword overlap with question
        
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Remove stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "what", "when", "where", "who", "why", "how"}
        question_words = question_words - stop_words
        
        if not question_words:
            return EvaluationResult(
                metric="relevance",
                score=1.0,
                passed=True,
                details={"reason": "No meaningful question words"},
            )
        
        overlap = len(question_words & response_words)
        score = overlap / len(question_words)
        
        return EvaluationResult(
            metric="relevance",
            score=score,
            passed=score >= 0.5,
            details={
                "question_keywords": list(question_words),
                "matched_keywords": list(question_words & response_words),
            },
        )


class GroundednessEvaluator(BaseEvaluator):
    """Evaluates if response is grounded in evidence"""
    
    @property
    def name(self) -> str:
        return "groundedness"
    
    async def evaluate(self, response: str, citations: List[Dict[str, Any]], **kwargs) -> EvaluationResult:
        """
        Groundedness: Are claims in response backed by citations?
        """
        if not citations:
            return EvaluationResult(
                metric="groundedness",
                score=0.0,
                passed=False,
                details={"reason": "No citations provided"},
            )
        
        # Check if response has citation markers or references
        citation_indicators = ["[", "]", "source", "reference", "citation", "according to"]
        response_lower = response.lower()
        
        has_citation_markers = any(indicator in response_lower for indicator in citation_indicators)
        
        # Score based on citation presence and count
        base_score = 0.5 if has_citation_markers else 0.2
        citation_bonus = min(len(citations) * 0.1, 0.5)
        score = min(base_score + citation_bonus, 1.0)
        
        return EvaluationResult(
            metric="groundedness",
            score=score,
            passed=score >= 0.6,
            details={
                "citation_count": len(citations),
                "has_citation_markers": has_citation_markers,
            },
        )


class ToolUsageEvaluator(BaseEvaluator):
    """Evaluates if tools were used correctly"""
    
    @property
    def name(self) -> str:
        return "tool_usage"
    
    async def evaluate(self, tool_calls: List[Dict[str, Any]], expected_tools: List[str], **kwargs) -> EvaluationResult:
        """
        Tool Usage: Were the right tools called with correct arguments?
        """
        if not expected_tools:
            return EvaluationResult(
                metric="tool_usage",
                score=1.0,
                passed=True,
                details={"reason": "No tools expected"},
            )
        
        called_tools = [tc.get("name", "") for tc in tool_calls]
        expected_set = set(expected_tools)
        called_set = set(called_tools)
        
        # Check if all expected tools were called
        missing = expected_set - called_set
        extra = called_set - expected_set
        
        score = 1.0 - (len(missing) * 0.3) - (len(extra) * 0.1)
        score = max(score, 0.0)
        
        return EvaluationResult(
            metric="tool_usage",
            score=score,
            passed=score >= 0.7 and len(missing) == 0,
            details={
                "expected_tools": list(expected_set),
                "called_tools": list(called_set),
                "missing_tools": list(missing),
                "extra_tools": list(extra),
            },
        )


class ExactMatchEvaluator(BaseEvaluator):
    """Exact match evaluation for deterministic tasks"""
    
    @property
    def name(self) -> str:
        return "exact_match"
    
    async def evaluate(self, response: str, expected: str, **kwargs) -> EvaluationResult:
        """Check if response exactly matches expected"""
        score = 1.0 if response.strip() == expected.strip() else 0.0
        
        return EvaluationResult(
            metric="exact_match",
            score=score,
            passed=score == 1.0,
            details={
                "expected": expected[:100],
                "actual": response[:100],
            },
        )


class SubstringEvaluator(BaseEvaluator):
    """Substring containment evaluation"""
    
    @property
    def name(self) -> str:
        return "substring"
    
    async def evaluate(self, response: str, expected_substrings: List[str], **kwargs) -> EvaluationResult:
        """Check if response contains expected substrings"""
        if not expected_substrings:
            return EvaluationResult(
                metric="substring",
                score=1.0,
                passed=True,
                details={"reason": "No substrings expected"},
            )
        
        found = sum(1 for s in expected_substrings if s.lower() in response.lower())
        score = found / len(expected_substrings)
        
        return EvaluationResult(
            metric="substring",
            score=score,
            passed=score >= 0.8,
            details={
                "expected": expected_substrings,
                "found": found,
                "total": len(expected_substrings),
            },
        )


# Evaluator registry
_EVALUATORS = {
    "faithfulness": FaithfulnessEvaluator(),
    "relevance": RelevanceEvaluator(),
    "groundedness": GroundednessEvaluator(),
    "tool_usage": ToolUsageEvaluator(),
    "exact_match": ExactMatchEvaluator(),
    "substring": SubstringEvaluator(),
}


def get_evaluator(name: str) -> Optional[BaseEvaluator]:
    return _EVALUATORS.get(name)


def get_all_evaluators() -> List[BaseEvaluator]:
    return list(_EVALUATORS.values())


async def run_evaluation_suite(
    response: str,
    question: str = "",
    context: List[str] = None,
    citations: List[Dict[str, Any]] = None,
    tool_calls: List[Dict[str, Any]] = None,
    expected_tools: List[str] = None,
    expected_answer: str = "",
    expected_substrings: List[str] = None,
) -> List[EvaluationResult]:
    """Run full evaluation suite"""
    results = []
    
    # Faithfulness
    if context:
        results.append(await _EVALUATORS["faithfulness"].evaluate(response=response, context=context))
    
    # Relevance
    if question:
        results.append(await _EVALUATORS["relevance"].evaluate(response=response, question=question))
    
    # Groundedness
    if citations:
        results.append(await _EVALUATORS["groundedness"].evaluate(response=response, citations=citations))
    
    # Tool usage
    if tool_calls is not None and expected_tools is not None:
        results.append(await _EVALUATORS["tool_usage"].evaluate(tool_calls=tool_calls, expected_tools=expected_tools))
    
    # Exact match
    if expected_answer:
        results.append(await _EVALUATORS["exact_match"].evaluate(response=response, expected=expected_answer))
    
    # Substring
    if expected_substrings:
        results.append(await _EVALUATORS["substring"].evaluate(response=response, expected_substrings=expected_substrings))
    
    return results
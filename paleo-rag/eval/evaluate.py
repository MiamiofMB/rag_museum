"""
Evaluation module for Paleo RAG system.

Provides basic evaluation metrics for RAG performance including
HitRate@K and Answer Relevancy using simple heuristics or ragas.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import config

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Holds evaluation metrics for a single query."""
    
    query: str
    hit_at_k: bool  # Whether relevant doc was in top-k
    relevancy_score: float  # 0-1 score for answer relevance
    retrieval_time_sec: float
    num_sources: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "hit_at_k": self.hit_at_k,
            "relevancy_score": self.relevancy_score,
            "retrieval_time_sec": self.retrieval_time_sec,
            "num_sources": self.num_sources,
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""
    
    total_queries: int
    hit_rate: float  # Fraction of queries with hit@k
    avg_relevancy: float
    avg_retrieval_time_sec: float
    results: list[EvaluationResult]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_queries": self.total_queries,
            "hit_rate": round(self.hit_rate, 4),
            "avg_relevancy": round(self.avg_relevancy, 4),
            "avg_retrieval_time_sec": round(self.avg_retrieval_time_sec, 4),
            "results": [r.to_dict() for r in self.results],
        }


# Test questions with expected keywords for evaluation
TEST_QUESTIONS = [
    {
        "question": "Какой динозавр был самым большим?",
        "expected_keywords": ["зауропод", "Argentinosaurus", "Patagotitan", "крупн", "размер", "вес", "тонн"],
    },
    {
        "question": "Почему вымерли динозавры?",
        "expected_keywords": ["астероид", "вымер", "удар", "климат", "вулкан", "мел", "палеоген"],
    },
    {
        "question": "Как определяют возраст окаменелостей?",
        "expected_keywords": ["датирован", "радиометрич", "изотоп", "биостратиграф", "возраст"],
    },
    {
        "question": "Где можно найти скелеты тираннозавра?",
        "expected_keywords": ["США", "Монтана", "Канада", "Альберта", "найд", "экспонат"],
    },
    {
        "question": "Что такое аммониты?",
        "expected_keywords": ["аммонит", "моллюск", "раковин", "мор", "биостратиграф"],
    },
    {
        "question": "Какие динозавры имели перья?",
        "expected_keywords": ["перь", "теропод", "Velociraptor", "Microraptor", "оперен"],
    },
    {
        "question": "Как работают палеонтологические раскопки?",
        "expected_keywords": ["раскоп", "квадрат", "слой", "документ", "гипс", "лаборатор"],
    },
    {
        "question": "В какую эпоху жили трицератопсы?",
        "expected_keywords": ["мелов", "мел", "период", "эпох", "66"],
    },
    {
        "question": "Что такое биостратиграфия?",
        "expected_keywords": ["биостратиграф", "индекс-фоссили", "слой", "корреляц", "аммонит"],
    },
    {
        "question": "Можно ли клонировать динозавра?",
        "expected_keywords": ["клон", "ДНК", "нет", "сохран", "разруш"],
    },
]


def check_hit_at_k(
    retrieved_docs: list[dict[str, Any]],
    expected_keywords: list[str],
) -> bool:
    """
    Check if any retrieved document contains expected keywords.
    
    Args:
        retrieved_docs: List of retrieved document dictionaries.
        expected_keywords: List of keywords that should appear.
    
    Returns:
        True if at least one document contains keywords.
    """
    for doc in retrieved_docs:
        content = doc.get("content", "").lower()
        title = doc.get("title", "").lower()
        combined = content + " " + title
        
        matches = sum(1 for kw in expected_keywords if kw.lower() in combined)
        if matches >= 1:
            return True
    
    return False


def compute_relevancy_score(
    answer: str,
    retrieved_docs: list[dict[str, Any]],
    expected_keywords: list[str],
) -> float:
    """
    Compute a simple relevancy score based on keyword overlap.
    
    Args:
        answer: Generated answer text.
        retrieved_docs: List of retrieved documents.
        expected_keywords: Expected keywords for the question.
    
    Returns:
        Score between 0 and 1.
    """
    answer_lower = answer.lower()
    
    # Count keyword matches in answer
    answer_matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    
    # Count keyword matches in retrieved docs
    doc_matches = 0
    for doc in retrieved_docs:
        content = doc.get("content", "").lower()
        title = doc.get("title", "").lower()
        combined = content + " " + title
        doc_matches += sum(1 for kw in expected_keywords if kw.lower() in combined)
    
    # Normalize
    max_possible = len(expected_keywords)
    if max_possible == 0:
        return 0.5  # Neutral if no keywords defined
    
    # Weight: 60% answer relevance, 40% retrieval relevance
    answer_score = min(answer_matches / max_possible, 1.0)
    doc_score = min(doc_matches / (max_possible * 3), 1.0)  # Expect ~3 docs
    
    final_score = 0.6 * answer_score + 0.4 * doc_score
    
    return round(final_score, 4)


def evaluate_rag_chain(
    rag_chain: Any,  # RAGChain type to avoid circular import
    num_questions: int | None = None,
    top_k: int | None = None,
) -> EvaluationReport:
    """
    Evaluate the RAG chain on test questions.
    
    Args:
        rag_chain: Initialized RAGChain instance.
        num_questions: Number of test questions to use.
        top_k: Number of documents to retrieve.
    
    Returns:
        EvaluationReport with aggregated metrics.
    """
    num_questions = num_questions or config.EVAL_NUM_QUESTIONS
    top_k = top_k or config.TOP_K
    
    # Limit to available questions
    num_questions = min(num_questions, len(TEST_QUESTIONS))
    
    logger.info(f"Evaluating on {num_questions} questions with top_k={top_k}")
    
    results = []
    
    for i, test_item in enumerate(TEST_QUESTIONS[:num_questions]):
        question = test_item["question"]
        expected_keywords = test_item["expected_keywords"]
        
        logger.debug(f"Processing question {i+1}/{num_questions}: {question[:40]}...")
        
        try:
            import time
            start_time = time.time()
            
            # Get response from RAG chain
            response = rag_chain.invoke(question, top_k=top_k)
            
            elapsed = time.time() - start_time
            
            # Extract sources
            sources = [s.to_dict() for s in response.sources]
            
            # Compute metrics
            hit = check_hit_at_k(sources, expected_keywords)
            relevancy = compute_relevancy_score(response.answer, sources, expected_keywords)
            
            result = EvaluationResult(
                query=question,
                hit_at_k=hit,
                relevancy_score=relevancy,
                retrieval_time_sec=round(elapsed, 3),
                num_sources=len(sources),
            )
            results.append(result)
            
            logger.debug(f"  Hit: {hit}, Relevancy: {relevancy:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate question {i+1}: {e}")
            # Add failed result
            results.append(EvaluationResult(
                query=question,
                hit_at_k=False,
                relevancy_score=0.0,
                retrieval_time_sec=0.0,
                num_sources=0,
            ))
    
    # Aggregate metrics
    total = len(results)
    hits = sum(1 for r in results if r.hit_at_k)
    hit_rate = hits / total if total > 0 else 0.0
    
    avg_relevancy = sum(r.relevancy_score for r in results) / total if total > 0 else 0.0
    avg_time = sum(r.retrieval_time_sec for r in results) / total if total > 0 else 0.0
    
    report = EvaluationReport(
        total_queries=total,
        hit_rate=hit_rate,
        avg_relevancy=avg_relevancy,
        avg_retrieval_time_sec=avg_time,
        results=results,
    )
    
    logger.info(f"Evaluation complete: HitRate={hit_rate:.4f}, AvgRelevancy={avg_relevancy:.4f}")
    
    return report


def print_report(report: EvaluationReport) -> None:
    """Print evaluation report to console."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\nTotal Queries: {report.total_queries}")
    print(f"Hit Rate @ {config.TOP_K}: {report.hit_rate:.2%}")
    print(f"Average Relevancy: {report.avg_relevancy:.4f}")
    print(f"Avg Retrieval Time: {report.avg_retrieval_time_sec:.3f}s")
    
    print("\n" + "-" * 70)
    print("DETAILED RESULTS")
    print("-" * 70)
    
    for i, result in enumerate(report.results, 1):
        status = "✓" if result.hit_at_k else "✗"
        print(f"\n{i}. {result.query[:50]}...")
        print(f"   Status: {status} Hit@{config.TOP_K}")
        print(f"   Relevancy: {result.relevancy_score:.4f}")
        print(f"   Time: {result.retrieval_time_sec:.3f}s")
        print(f"   Sources: {result.num_sources}")
    
    print("\n" + "=" * 70)


def main() -> None:
    """Run evaluation on the RAG chain."""
    logging.basicConfig(level=logging.INFO)
    
    # Check if index exists
    if not config.FAISS_INDEX_PATH.exists():
        print(f"Vector store not found at {config.FAISS_INDEX_PATH}")
        print("Please run main.py first to build the index.")
        return
    
    # Import here to avoid circular dependency
    from rag.rag_chain import RAGChain
    
    print("Initializing RAG chain for evaluation...")
    chain = RAGChain()
    
    print(f"Running evaluation on {config.EVAL_NUM_QUESTIONS} test questions...\n")
    
    report = evaluate_rag_chain(chain)
    print_report(report)
    
    # Save report to file
    report_path = config.INDEX_DIR / "evaluation_report.json"
    import json
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()

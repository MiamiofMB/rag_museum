"""
Retriever module for Paleo RAG system.

Wraps FAISS vector store to provide semantic search functionality
with HyDE query rewriting support.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import config
from pipeline.chunker import Chunk
from pipeline.embedder import Embedder
from pipeline.vector_store import VectorStore, load_vector_store
from rag.hyde_rewriter import HyDERewriter

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a single retrieval result with metadata."""
    
    chunk: Chunk
    score: float
    rank: int
    
    @property
    def title(self) -> str:
        """Get the source document title."""
        return self.chunk.metadata.get("source_title", "Неизвестно")
    
    @property
    def epoch(self) -> str:
        """Get the epoch from metadata."""
        return self.chunk.metadata.get("epoch", "Н/Д")
    
    @property
    def hall(self) -> str:
        """Get the hall from metadata."""
        return self.chunk.metadata.get("hall", "Н/Д")
    
    @property
    def doc_type(self) -> str:
        """Get the document type."""
        return self.chunk.metadata.get("source_type", "unknown")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for UI display."""
        return {
            "title": self.title,
            "content": self.chunk.content,
            "epoch": self.epoch,
            "hall": self.hall,
            "doc_type": self.doc_type,
            "score": round(self.score, 4),
            "rank": self.rank,
        }


class Retriever:
    """
    Semantic retriever using FAISS and HyDE.
    
    Combines HyDE query rewriting with vector search to find
    relevant document chunks for a given query.
    """
    
    def __init__(
        self,
        vector_store_path: Path | None = None,
        top_k: int | None = None,
        use_hyde: bool = True,
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store_path: Path to saved FAISS index.
            top_k: Number of results to retrieve.
            use_hyde: Whether to use HyDE query rewriting.
        """
        self.vector_store_path = vector_store_path or config.FAISS_INDEX_PATH
        self.top_k = top_k or config.TOP_K
        self.use_hyde = use_hyde
        
        self._vector_store: VectorStore | None = None
        self._embedder: Embedder | None = None
        self._hyde_rewriter: HyDERewriter | None = None
        
        logger.info(f"Retriever initialized with top_k={self.top_k}, hyde={self.use_hyde}")
    
    @property
    def vector_store(self) -> VectorStore:
        """Get the vector store, loading if necessary."""
        if self._vector_store is None:
            logger.info(f"Loading vector store from {self.vector_store_path}")
            self._vector_store = load_vector_store(self.vector_store_path)
        return self._vector_store
    
    @property
    def embedder(self) -> Embedder:
        """Get the embedder, initializing if necessary."""
        if self._embedder is None:
            logger.info("Initializing embedder")
            self._embedder = Embedder()
        return self._embedder
    
    @property
    def hyde_rewriter(self) -> HyDERewriter:
        """Get the HyDE rewriter, initializing if necessary."""
        if self._hyde_rewriter is None:
            logger.info("Initializing HyDE rewriter")
            self._hyde_rewriter = HyDERewriter()
        return self._hyde_rewriter
    
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_hyde: bool | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query string.
            top_k: Optional override for number of results.
            use_hyde: Optional override for HyDE usage.
        
        Returns:
            List of RetrievalResult objects sorted by relevance.
        """
        actual_top_k = top_k if top_k is not None else self.top_k
        actual_use_hyde = use_hyde if use_hyde is not None else self.use_hyde
        
        logger.info(f"Retrieving for query: {query[:50]}... (top_k={actual_top_k})")
        
        try:
            # Step 1: Apply HyDE if enabled
            if actual_use_hyde:
                search_query = self.hyde_rewriter.rewrite(query)
                logger.debug(f"HyDE rewritten query length: {len(search_query)}")
            else:
                search_query = query
                logger.debug("Using original query (HyDE disabled)")
            
            # Step 2: Encode the query
            query_embedding = self.embedder.encode_query(search_query)
            
            # Step 3: Search vector store
            results = self.vector_store.search(query_embedding, top_k=actual_top_k)
            
            # Step 4: Convert to RetrievalResult objects
            retrieval_results = [
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=i + 1,
                )
                for i, (chunk, score) in enumerate(results)
            ]
            
            logger.info(f"Retrieved {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """
        Retrieve results with additional metadata.
        
        Args:
            query: User query string.
            top_k: Number of results to retrieve.
        
        Returns:
            Tuple of (results list, metadata dict with timing and stats).
        """
        import time
        
        start_time = time.time()
        
        results = self.retrieve(query, top_k=top_k)
        
        elapsed = time.time() - start_time
        
        metadata = {
            "query": query,
            "retrieval_time_sec": round(elapsed, 3),
            "num_results": len(results),
            "hyde_used": self.use_hyde,
            "top_k": top_k or self.top_k,
        }
        
        return results, metadata


def create_retriever(
    vector_store_path: Path | None = None,
    top_k: int | None = None,
    use_hyde: bool = True,
) -> Retriever:
    """
    Factory function to create a Retriever instance.
    
    Args:
        vector_store_path: Path to FAISS index.
        top_k: Number of results to return.
        use_hyde: Whether to enable HyDE.
    
    Returns:
        Initialized Retriever instance.
    """
    return Retriever(
        vector_store_path=vector_store_path,
        top_k=top_k,
        use_hyde=use_hyde,
    )


def main() -> None:
    """Test the retriever on sample queries."""
    logging.basicConfig(level=logging.INFO)
    
    # Check if index exists
    if not config.FAISS_INDEX_PATH.exists():
        print(f"Vector store not found at {config.FAISS_INDEX_PATH}")
        print("Please run main.py first to build the index.")
        return
    
    retriever = Retriever()
    
    test_queries = [
        "Какой динозавр был самым большим?",
        "Почему вымерли динозавры?",
        "Как определяют возраст окаменелостей?",
    ]
    
    print("Testing retriever...\n")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        try:
            results = retriever.retrieve(query, top_k=3)
            
            for result in results:
                print(f"\n[{result.rank}] Score: {result.score:.4f}")
                print(f"    Title: {result.title}")
                print(f"    Epoch: {result.epoch}")
                print(f"    Hall: {result.hall}")
                print(f"    Content: {result.chunk.content[:150]}...")
        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n\nRetriever test completed!")


if __name__ == "__main__":
    main()

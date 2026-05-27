"""
RAG chain module for Paleo RAG system.

Assembles the full retrieval-augmented generation pipeline:
query → HyDE → retrieval → context assembly → LLM response.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from config import config
from rag.retriever import Retriever, RetrievalResult

logger = logging.getLogger(__name__)


# Final answer generation prompt in Russian
RAG_PROMPT_TEMPLATE = """Ты — научный эксперт палеонтологического музея. 
Отвечай на вопросы посетителей точно, научно, но доступно.

ИНСТРУКЦИИ:
- Используй только предоставленный контекст для ответа
- Если ответа нет в контексте, скажи об этом честно
- Цитируй источники (название экспоната, эпоху)
- Отвечай на русском языке
- Будь краток, но информативен (3-5 предложений)
- Избегай выдумок и спекуляций

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}

ВОПРОС ПОСЕТИТЕЛЯ:
{question}

ОТВЕТ ЭКСПЕРТА:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)


@dataclass
class RAGResponse:
    """Represents a complete RAG response with metadata."""
    
    answer: str
    sources: list[RetrievalResult]
    query: str
    hyde_query: Optional[str] = None
    processing_time_sec: float = 0.0
    num_sources: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for UI/API response."""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "query": self.query,
            "hyde_query": self.hyde_query,
            "processing_time_sec": self.processing_time_sec,
            "num_sources": self.num_sources,
        }


class RAGChain:
    """
    Complete RAG pipeline with HyDE support.
    
    Orchestrates the flow from user query through HyDE rewriting,
    document retrieval, context assembly, and final answer generation.
    """
    
    def __init__(
        self,
        retriever: Retriever | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """
        Initialize the RAG chain.
        
        Args:
            retriever: Pre-configured Retriever instance.
            model_name: Ollama model name for answer generation.
            base_url: Ollama server URL.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        self._retriever = retriever
        self.model_name = model_name or config.LLM_MODEL
        self.base_url = base_url or config.LLM_BASE_URL
        self.temperature = temperature if temperature is not None else config.TEMPERATURE
        self.max_tokens = max_tokens or config.MAX_TOKENS
        
        self._llm: Optional[OllamaLLM] = None
        self._prompt = RAG_PROMPT
        
        logger.info(f"RAGChain initialized with model: {self.model_name}")
    
    @property
    def retriever(self) -> Retriever:
        """Get or create the retriever."""
        if self._retriever is None:
            logger.info("Creating default retriever")
            self._retriever = Retriever()
        return self._retriever
    
    @retriever.setter
    def retriever(self, value: Retriever) -> None:
        """Set the retriever."""
        self._retriever = value
    
    @property
    def llm(self) -> OllamaLLM:
        """Lazy-load the LLM."""
        if self._llm is None:
            logger.info(f"Initializing Ollama LLM: {self.model_name}")
            try:
                self._llm = OllamaLLM(
                    model=self.model_name,
                    base_url=self.base_url,
                    temperature=self.temperature,
                    num_predict=self.max_tokens,
                )
            except Exception as e:
                logger.error(f"Failed to initialize Ollama LLM: {e}")
                raise
        return self._llm
    
    def _build_context(self, results: list[RetrievalResult]) -> str:
        """
        Build context string from retrieval results.
        
        Args:
            results: List of retrieval results.
        
        Returns:
            Formatted context string for the LLM.
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            part = f"""[Источник {i}]
Название: {result.title}
Эпоха: {result.epoch}
Зал: {result.hall}
Тип: {result.doc_type}
Содержание: {result.chunk.content}
---"""
            context_parts.append(part)
        
        return "\n\n".join(context_parts)
    
    def invoke(
        self,
        query: str,
        top_k: int | None = None,
        use_hyde: bool | None = None,
    ) -> RAGResponse:
        """
        Process a query through the full RAG pipeline.
        
        Args:
            query: User query string.
            top_k: Number of documents to retrieve.
            use_hyde: Whether to use HyDE query rewriting.
        
        Returns:
            RAGResponse with answer and metadata.
        """
        import time
        
        start_time = time.time()
        
        logger.info(f"Processing query: {query[:50]}...")
        
        try:
            # Step 1: Retrieve relevant documents
            logger.debug("Step 1: Retrieving documents...")
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                use_hyde=use_hyde,
            )
            
            # Get HyDE query if used
            hyde_query = None
            if use_hyde or (use_hyde is None and self.retriever.use_hyde):
                # The retriever already applied HyDE internally
                # We can't easily get the rewritten query, so we'll leave it None
                # or we could modify the retriever to return it
                hyde_query = "HyDE applied (internal)"
            
            # Step 2: Build context from results
            logger.debug(f"Step 2: Building context from {len(results)} results...")
            context = self._build_context(results)
            
            # Step 3: Generate answer using LLM
            logger.debug("Step 3: Generating answer...")
            prompt_text = self._prompt.format(context=context, question=query)
            answer = self.llm.invoke(prompt_text)
            
            # Clean answer
            answer = answer.strip()
            
            # Calculate timing
            elapsed = time.time() - start_time
            
            logger.info(f"Generated answer in {elapsed:.2f}s")
            
            return RAGResponse(
                answer=answer,
                sources=results,
                query=query,
                hyde_query=hyde_query,
                processing_time_sec=round(elapsed, 3),
                num_sources=len(results),
            )
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            raise
    
    def stream(
        self,
        query: str,
        top_k: int | None = None,
    ):
        """
        Stream the answer generation (if supported by LLM).
        
        Args:
            query: User query string.
            top_k: Number of documents to retrieve.
        
        Yields:
            Chunks of the generated answer.
        """
        # First retrieve documents
        results = self.retriever.retrieve(query=query, top_k=top_k)
        context = self._build_context(results)
        
        # Generate answer with streaming
        prompt_text = self._prompt.format(context=context, question=query)
        
        for chunk in self.llm.stream(prompt_text):
            yield chunk


def create_rag_chain(
    retriever: Retriever | None = None,
    model_name: str | None = None,
) -> RAGChain:
    """
    Factory function to create a RAGChain instance.
    
    Args:
        retriever: Optional pre-configured retriever.
        model_name: Optional model name override.
    
    Returns:
        Initialized RAGChain instance.
    """
    return RAGChain(retriever=retriever, model_name=model_name)


def main() -> None:
    """Test the RAG chain on sample queries."""
    logging.basicConfig(level=logging.INFO)
    
    # Check if index exists
    if not config.FAISS_INDEX_PATH.exists():
        print(f"Vector store not found at {config.FAISS_INDEX_PATH}")
        print("Please run main.py first to build the index.")
        return
    
    chain = RAGChain()
    
    test_queries = [
        "Какой динозавр был самым большим?",
        "Почему вымерли динозавры?",
        "Как палеонтологи определяют возраст окаменелостей?",
    ]
    
    print("Testing RAG chain...\n")
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print("=" * 60)
        
        try:
            response = chain.invoke(query, top_k=3)
            
            print(f"\nANSWER ({response.processing_time_sec}s):")
            print(response.answer)
            
            print(f"\nSOURCES ({response.num_sources}):")
            for source in response.sources:
                print(f"  [{source.rank}] {source.title} ({source.epoch})")
        
        except Exception as e:
            print(f"ERROR: {e}")
    
    print(f"\n\n{'='*60}")
    print("RAG chain test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

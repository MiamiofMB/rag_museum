"""
HyDE (Hypothetical Document Embeddings) rewriter for Paleo RAG system.

Generates hypothetical scientific documents from user queries to improve
retrieval quality. Includes fallback to original query if LLM fails.
"""

import logging
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from config import config

logger = logging.getLogger(__name__)


# HyDE prompt in Russian for paleontology context
HYDE_PROMPT_TEMPLATE = """Ты — научный ассистент палеонтологического музея. 
Твоя задача — написать краткий гипотетический научный текст, который мог бы содержать ответ на вопрос посетителя.

ИНСТРУКЦИЯ:
- Напиши научно-популярный текст объёмом 3-5 предложений на русском языке
- Текст должен быть похож на описание экспоната или научную статью
- Используй терминологию палеонтологии и геологии
- Не отвечай на вопрос напрямую, а создай контекстный документ
- Избегай фраз вроде "ответ на ваш вопрос" или "гипотетический документ"

ВОПРОС ПОСЕТИТЕЛЯ:
{question}

ГИПОТЕТИЧЕСКИЙ НАУЧНЫЙ ТЕКСТ:"""

HYDE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=HYDE_PROMPT_TEMPLATE,
)


class HyDERewriter:
    """
    HyDE query rewriter using Ollama LLM.
    
    Generates a hypothetical document from the user query,
    which is then used for embedding-based retrieval.
    Falls back to the original query if generation fails.
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """
        Initialize the HyDE rewriter.
        
        Args:
            model_name: Ollama model name (e.g., "qwen2.5:7b").
            base_url: Ollama server URL.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        self.model_name = model_name or config.LLM_MODEL
        self.base_url = base_url or config.LLM_BASE_URL
        self.temperature = temperature if temperature is not None else config.TEMPERATURE
        self.max_tokens = max_tokens or config.MAX_TOKENS
        
        self._llm: Optional[OllamaLLM] = None
        self._prompt = HYDE_PROMPT
        
        logger.info(f"HyDERewriter initialized with model: {self.model_name}")
    
    @property
    def llm(self) -> OllamaLLM:
        """Lazy-load the LLM."""
        if self._llm is None:
            logger.info(f"Initializing Ollama LLM: {self.model_name} at {self.base_url}")
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
    
    def rewrite(self, query: str, use_hyde: bool = True) -> str:
        """
        Rewrite a query using HyDE or return original.
        
        Args:
            query: Original user query.
            use_hyde: Whether to apply HyDE transformation.
        
        Returns:
            Rewritten query (hypothetical document) or original query.
        """
        if not use_hyde:
            logger.debug("HyDE disabled, using original query")
            return query
        
        logger.debug(f"Applying HyDE to query: {query[:50]}...")
        
        try:
            # Generate hypothetical document
            hypothetical_doc = self._generate_hypothetical_document(query)
            
            # Validate response
            if self._is_valid_response(hypothetical_doc):
                logger.info(f"HyDE successful, generated {len(hypothetical_doc)} chars")
                return hypothetical_doc
            else:
                logger.warning(
                    f"HyDE response invalid ({len(hypothetical_doc)} chars), "
                    "falling back to original query"
                )
                return query
                
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}, falling back to original query")
            return query
    
    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document from the query.
        
        Args:
            query: User query.
        
        Returns:
            Generated hypothetical document text.
        """
        # Format prompt
        prompt_text = self._prompt.format(question=query)
        
        # Call LLM
        response = self.llm.invoke(prompt_text)
        
        # Clean response
        if isinstance(response, str):
            cleaned = response.strip()
            # Remove common prefixes that might appear
            for prefix in ["ГИПОТЕТИЧЕСКИЙ НАУЧНЫЙ ТЕКСТ:", "Ответ:", "Текст:"]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            return cleaned
        else:
            return str(response)
    
    def _is_valid_response(self, text: str) -> bool:
        """
        Check if the generated response is valid.
        
        Args:
            text: Generated text to validate.
        
        Returns:
            True if valid, False otherwise.
        """
        if not text:
            return False
        
        # Check minimum length (~30 tokens ≈ 120 characters)
        if len(text) < 120:
            return False
        
        # Check for obvious failure patterns
        failure_patterns = [
            "я не могу",
            "не знаю",
            "извините",
            "error",
            "exception",
        ]
        
        text_lower = text.lower()
        for pattern in failure_patterns:
            if pattern in text_lower:
                return False
        
        return True
    
    def count_tokens_approx(self, text: str) -> int:
        """
        Approximate token count for Russian text.
        
        Args:
            text: Text to count tokens.
        
        Returns:
            Approximate token count.
        """
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4


def create_hyde_rewriter(
    model_name: str | None = None,
    base_url: str | None = None,
) -> HyDERewriter:
    """
    Factory function to create a HyDERewriter instance.
    
    Args:
        model_name: Optional model name override.
        base_url: Optional base URL override.
    
    Returns:
        Initialized HyDERewriter instance.
    """
    return HyDERewriter(model_name=model_name, base_url=base_url)


def main() -> None:
    """Test HyDE rewriting on sample queries."""
    logging.basicConfig(level=logging.INFO)
    
    rewriter = HyDERewriter()
    
    test_queries = [
        "Какой динозавр был самым большим?",
        "Почему вымерли динозавры?",
        "Как определяют возраст окаменелостей?",
        "Где можно найти скелеты тираннозавра?",
    ]
    
    print("Testing HyDE rewriter...\n")
    
    for query in test_queries:
        print(f"Original query: {query}")
        try:
            rewritten = rewriter.rewrite(query)
            print(f"Rewritten ({len(rewritten)} chars):")
            print(f"  {rewritten[:200]}..." if len(rewritten) > 200 else f"  {rewritten}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 60)
    
    print("\nHyDE test completed!")


if __name__ == "__main__":
    main()

"""
Embedding module for Paleo RAG system.

Provides embedding generation using sentence-transformers with
the BAAI/bge-small-ru-v1.5 model optimized for Russian text.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import config
from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wrapper around sentence-transformers for generating embeddings.
    
    Uses the BAAI/bge-small-ru-v1.5 model which is optimized
    for Russian language semantic search.
    """
    
    def __init__(self, model_name: str | None = None):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None
        
        logger.info(f"Initializing embedder with model: {self.model_name}")
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model with retry logic."""
        if self._model is None:
            logger.info(f"Loading model: {self.model_name}")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self._model = SentenceTransformer(
                        self.model_name,
                        trust_remote_code=True,
                    )
                    self._model.eval()
                    logger.info(f"Model loaded successfully on attempt {attempt + 1}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Failed to load model (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {2 ** attempt} seconds..."
                        )
                        import time
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(f"Failed to load model after {max_retries} attempts: {e}")
                        raise
        return self._model
    
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
        
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([])
        
        logger.debug(f"Encoding {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Important for cosine similarity
        )
        
        logger.debug(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.
        
        Args:
            query: Query text to encode.
        
        Returns:
            1D numpy array of the embedding.
        """
        embeddings = self.encode([query])
        return embeddings[0]
    
    def encode_chunks(self, chunks: list[Chunk]) -> tuple[np.ndarray, list[str]]:
        """
        Encode a list of Chunk objects.
        
        Args:
            chunks: List of Chunk objects to encode.
        
        Returns:
            Tuple of (embeddings array, chunk IDs list).
        """
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.id for chunk in chunks]
        
        embeddings = self.encode(texts, show_progress=True)
        
        return embeddings, chunk_ids
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        # bge-small-ru-v1.5 has 512 dimensions
        sample = self.encode(["test"])
        return sample.shape[1]


def create_embedder(model_name: str | None = None) -> Embedder:
    """
    Factory function to create an Embedder instance.
    
    Args:
        model_name: Optional model name override.
    
    Returns:
        Initialized Embedder instance.
    """
    return Embedder(model_name)


def main() -> None:
    """Test the embedder on sample texts."""
    logging.basicConfig(level=logging.INFO)
    
    embedder = Embedder()
    
    # Test texts in Russian
    test_texts = [
        "Тираннозавр рекс был крупным хищным динозавром.",
        "Аммониты вымерли вместе с динозаврами 66 миллионов лет назад.",
        "Радиометрическое датирование позволяет определить возраст пород.",
    ]
    
    print("Encoding test texts...")
    embeddings = embedder.encode(test_texts, show_progress=True)
    
    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Compute similarity between first two texts
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"\nCosine similarity between text 1 and 2: {similarity:.4f}")
    
    similarity_13 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
    print(f"Cosine similarity between text 1 and 3: {similarity_13:.4f}")


if __name__ == "__main__":
    main()

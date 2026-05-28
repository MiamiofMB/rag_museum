"""
FAISS vector store module for Paleo RAG system.

Handles creation, saving, loading, and searching of FAISS indices
for efficient similarity search over document embeddings.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from config import config
from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for semantic search.
    
    Supports creating an index from embeddings, saving/loading
    the index, and performing similarity search.
    """
    
    def __init__(
        self,
        index_path: Path | None = None,
        embedding_dim: int | None = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            index_path: Path to save/load the FAISS index.
            embedding_dim: Dimension of embeddings (auto-detected if not provided).
        """
        self.index_path = index_path or config.FAISS_INDEX_PATH
        self.embedding_dim = embedding_dim or 384  # Default for all-MiniLM-L6-v2
        self._index: faiss.Index | None = None
        self._chunk_map: dict[str, Chunk] = {}
        self._id_to_idx: dict[str, int] = {}
        
        logger.info(f"VectorStore initialized with path: {self.index_path}")
    
    @property
    def index(self) -> faiss.Index:
        """Get the FAISS index, raising error if not loaded."""
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load() or create_index() first.")
        return self._index
    
    def create_index(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """
        Create a FAISS index from embeddings and chunks.
        
        Args:
            embeddings: Numpy array of shape (n_chunks, embedding_dim).
            chunks: List of Chunk objects corresponding to embeddings.
        """
        if len(embeddings) == 0:
            raise ValueError("Cannot create index from empty embeddings")
        
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and chunks ({len(chunks)}) "
                "must have the same length"
            )
        
        # Update embedding dimension from actual embeddings
        actual_dim = embeddings.shape[1]
        if self.embedding_dim != actual_dim:
            logger.info(f"Updating embedding dimension from {self.embedding_dim} to {actual_dim}")
            self.embedding_dim = actual_dim
        
        logger.info(f"Creating FAISS index with {len(embeddings)} vectors of dimension {self.embedding_dim}...")
        
        # Use IndexFlatIP for inner product (works with normalized embeddings)
        # which is equivalent to cosine similarity
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)
        
        # Add embeddings to index
        self._index.add(np.ascontiguousarray(embeddings_normalized, dtype=np.float32))
        
        # Build chunk mapping
        self._chunk_map = {}
        self._id_to_idx = {}
        
        for idx, chunk in enumerate(chunks):
            self._chunk_map[chunk.id] = chunk
            self._id_to_idx[chunk.id] = idx
        
        logger.info(f"Index created with {self._index.ntotal} vectors")
    
    def save(self, path: Path | None = None) -> None:
        """
        Save the FAISS index and chunk map to disk.
        
        Args:
            path: Optional path override.
        """
        save_path = path or self.index_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self._index is None:
            raise RuntimeError("No index to save")
        
        logger.info(f"Saving index to {save_path}...")
        
        # Save FAISS index
        faiss.write_index(self._index, str(save_path / "index.faiss"))
        
        # Save chunk map and ID mapping
        metadata = {
            "chunk_map": self._chunk_map,
            "id_to_idx": self._id_to_idx,
            "embedding_dim": self.embedding_dim,
        }
        
        with open(save_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved to {save_path}")
    
    def load(self, path: Path | None = None) -> None:
        """
        Load the FAISS index and chunk map from disk.
        
        Args:
            path: Optional path override.
        """
        load_path = path or self.index_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Index path does not exist: {load_path}")
        
        index_file = load_path / "index.faiss"
        metadata_file = load_path / "metadata.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        logger.info(f"Loading index from {load_path}...")
        
        # Load FAISS index
        self._index = faiss.read_index(str(index_file))
        
        # Load metadata
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        
        self._chunk_map = metadata.get("chunk_map", {})
        self._id_to_idx = metadata.get("id_to_idx", {})
        self.embedding_dim = metadata.get("embedding_dim", self.embedding_dim)
        
        logger.info(f"Loaded index with {self._index.ntotal} vectors")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for most similar chunks to a query embedding.
        
        Args:
            query_embedding: 1D numpy array of query embedding.
            top_k: Number of results to return.
        
        Returns:
            List of (Chunk, score) tuples sorted by similarity.
        """
        if self._index is None:
            raise RuntimeError("Index not loaded")
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        
        # Limit k to available vectors
        actual_k = min(top_k, self._index.ntotal)
        
        # Search
        distances, indices = self.index.search(query_embedding, actual_k)
        
        # Convert to list of (Chunk, score) tuples
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                # Find chunk by index
                chunk = None
                for chunk_id, chunk_idx in self._id_to_idx.items():
                    if chunk_idx == idx:
                        chunk = self._chunk_map[chunk_id]
                        break
                
                if chunk:
                    results.append((chunk, float(dist)))
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by its ID."""
        return self._chunk_map.get(chunk_id)
    
    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal


def build_vector_store(
    embeddings: np.ndarray,
    chunks: list[Chunk],
    save_path: Path | None = None,
) -> VectorStore:
    """
    Build and save a vector store from embeddings and chunks.
    
    Args:
        embeddings: Numpy array of embeddings.
        chunks: List of Chunk objects.
        save_path: Optional path to save the index.
    
    Returns:
        Initialized and populated VectorStore.
    """
    embedding_dim = embeddings.shape[1]
    store = VectorStore(index_path=save_path, embedding_dim=embedding_dim)
    store.create_index(embeddings, chunks)
    
    if save_path:
        store.save()
    
    return store


def load_vector_store(path: Path | None = None) -> VectorStore:
    """
    Load a vector store from disk.
    
    Args:
        path: Path to the saved index.
    
    Returns:
        Loaded VectorStore instance.
    """
    store = VectorStore(index_path=path)
    store.load()
    return store


def main() -> None:
    """Test vector store creation and search."""
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    from pipeline.embedder import Embedder
    
    embedder = Embedder()
    
    test_texts = [
        "Тираннозавр рекс — хищный динозавр мелового периода.",
        "Трицератопс был травоядным динозавром с тремя рогами.",
        "Аммониты вымерли 66 миллионов лет назад.",
        "Радиометрическое датирование определяет возраст пород.",
        "Палеонтология изучает ископаемые остатки древних организмов.",
    ]
    
    print("Creating test chunks...")
    chunks = [
        Chunk(
            id=f"test_{i}",
            content=text,
            metadata={"source": "test"},
            source_doc_id=i,
            chunk_index=0,
            total_chunks=1,
        )
        for i, text in enumerate(test_texts)
    ]
    
    print("Encoding texts...")
    embeddings, _ = embedder.encode_chunks(chunks)
    
    print("Building vector store...")
    store = VectorStore(embedding_dim=embeddings.shape[1])
    store.create_index(embeddings, chunks)
    
    # Test search
    print("\nSearching for 'хищный динозавр'...")
    query_emb = embedder.encode_query("хищный динозавр")
    results = store.search(query_emb, top_k=3)
    
    for chunk, score in results:
        print(f"  Score: {score:.4f} - {chunk.content[:60]}...")
    
    print("\nVector store test completed!")


if __name__ == "__main__":
    main()

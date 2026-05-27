"""
Text chunking module for Paleo RAG system.

Splits documents into chunks with configurable size and overlap,
preserving metadata for each chunk.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import config


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    
    id: str
    content: str
    metadata: dict[str, Any]
    source_doc_id: int
    chunk_index: int
    total_chunks: int


def count_tokens(text: str) -> int:
    """
    Approximate token count for Russian text.
    
    Uses simple heuristic: ~4 characters per token on average.
    For more accuracy, use a real tokenizer.
    """
    # Rough estimate: 1 token ≈ 4 characters for mixed text
    return len(text) // 4


def split_text_into_chunks(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks by sentences.
    
    Args:
        text: Input text to split.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Number of overlapping tokens between chunks.
    
    Returns:
        List of text chunks.
    """
    if not text:
        return []
    
    # Split by sentences (Russian punctuation)
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in ".!?。\n":
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # Add remaining text
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    if not sentences:
        return [text] if text else []
    
    # Group sentences into chunks
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    # Convert token sizes to character sizes (approximate)
    chunk_chars = chunk_size * 4
    overlap_chars = chunk_overlap * 4
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If single sentence is too long, split it
        if sentence_tokens > chunk_size:
            # If we have accumulated chunk, save it first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0
            
            # Split long sentence by words
            words = sentence.split()
            temp_chunk = ""
            temp_tokens = 0
            
            for word in words:
                word_tokens = count_tokens(word)
                if temp_tokens + word_tokens > chunk_size:
                    chunks.append(temp_chunk.strip())
                    # Keep overlap
                    temp_chunk = word + " "
                    temp_tokens = word_tokens
                else:
                    temp_chunk += word + " "
                    temp_tokens += word_tokens
            
            if temp_chunk.strip():
                current_chunk = temp_chunk
                current_tokens = temp_tokens
        else:
            # Check if adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if chunk_overlap > 0 and chunks:
                    # Get last portion of previous chunk for overlap
                    prev_chunk = chunks[-1]
                    overlap_text = prev_chunk[-overlap_chars:] if len(prev_chunk) > overlap_chars else prev_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                current_tokens = count_tokens(current_chunk)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Chunk a list of documents.
    
    Args:
        documents: List of document dictionaries with 'content' and 'metadata'.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between chunks in tokens.
    
    Returns:
        List of Chunk objects.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    all_chunks = []
    
    for doc in documents:
        doc_id = doc.get("id", 0)
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        
        # Merge document metadata with chunk-specific info
        base_metadata = {
            **metadata,
            "source_title": doc.get("title", ""),
            "source_type": doc.get("type", "unknown"),
        }
        
        # Split content into chunks
        text_chunks = split_text_into_chunks(
            text=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Create Chunk objects
        for idx, chunk_text in enumerate(text_chunks):
            chunk = Chunk(
                id=f"{doc_id}_{idx}",
                content=chunk_text,
                metadata=base_metadata,
                source_doc_id=doc_id,
                chunk_index=idx,
                total_chunks=len(text_chunks),
            )
            all_chunks.append(chunk)
    
    return all_chunks


def load_and_chunk(
    input_path: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Load documents from JSONL and chunk them.
    
    Args:
        input_path: Path to JSONL file.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between chunks in tokens.
    
    Returns:
        List of Chunk objects.
    """
    from data.generate_data import load_documents
    
    documents = load_documents(input_path)
    return chunk_documents(documents, chunk_size, chunk_overlap)


def main() -> None:
    """Test chunking on sample data."""
    print("Loading documents...")
    chunks = load_and_chunk(config.RAW_DATA_FILE)
    
    print(f"Generated {len(chunks)} chunks from {config.RAW_DATA_FILE}")
    
    # Print sample
    if chunks:
        print("\n=== Sample chunk ===")
        sample = chunks[0]
        print(f"ID: {sample.id}")
        print(f"Content: {sample.content[:200]}...")
        print(f"Metadata: {sample.metadata}")
        print(f"Source doc ID: {sample.source_doc_id}")
        print(f"Chunk {sample.chunk_index + 1}/{sample.total_chunks}")


if __name__ == "__main__":
    main()

"""Text chunking module for Paleo RAG system."""

from dataclasses import dataclass
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
    """Approximate token count for Russian text."""
    return len(text) // 4


def split_text_into_chunks(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks by sentences.

    Args:
        text: Input text to split.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Number of overlapping tokens between chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []
    
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in ".!?。\n":
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    if not sentences:
        return [text] if text else []
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    chunk_chars = chunk_size * 4
    overlap_chars = chunk_overlap * 4
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        if sentence_tokens > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0
            
            words = sentence.split()
            temp_chunk = ""
            temp_tokens = 0
            
            for word in words:
                word_tokens = count_tokens(word)
                if temp_tokens + word_tokens > chunk_size:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = word + " "
                    temp_tokens = word_tokens
                else:
                    temp_chunk += word + " "
                    temp_tokens += word_tokens
            
            if temp_chunk.strip():
                current_chunk = temp_chunk
                current_tokens = temp_tokens
        else:
            if current_tokens + sentence_tokens > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                if chunk_overlap > 0 and chunks:
                    prev_chunk = chunks[-1]
                    overlap_text = prev_chunk[-overlap_chars:] if len(prev_chunk) > overlap_chars else prev_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                current_tokens = count_tokens(current_chunk)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk a list of documents.

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
        
        base_metadata = {
            **metadata,
            "source_title": doc.get("title", ""),
            "source_type": doc.get("type", "unknown"),
        }
        
        text_chunks = split_text_into_chunks(
            text=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
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
    input_path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Load documents from JSONL and chunk them.

    Args:
        input_path: Path to JSONL file.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between chunks in tokens.

    Returns:
        List of Chunk objects.
    """
    import json
    
    documents = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    
    return chunk_documents(documents, chunk_size, chunk_overlap)


def main() -> None:
    """Test chunking on sample data."""
    print("Loading documents...")
    chunks = load_and_chunk(config.RAW_DATA_FILE)
    
    print(f"Generated {len(chunks)} chunks from {config.RAW_DATA_FILE}")
    
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

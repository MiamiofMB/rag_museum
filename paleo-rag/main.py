#!/usr/bin/env python3
"""
Main entry point for Paleo RAG system.

Orchestrates the full pipeline:
1. Check/load configuration
2. Generate synthetic data (if not exists)
3. Build FAISS index
4. Launch Gradio UI

Usage:
    python main.py [--skip-data] [--skip-index] [--eval-only]
"""

import argparse
import logging
import sys
from pathlib import Path

from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_env_file() -> bool:
    """Check if .env file exists."""
    env_file = Path(".env")
    
    if not env_file.exists():
        logger.warning(f".env file not found at {env_file}")
        logger.info("Using default configuration values")
        logger.info(f"Copy .env.example to .env to customize: cp .env.example .env")
        return False
    
    logger.info(f"Loaded configuration from {env_file}")
    return True


def generate_data(force: bool = False) -> bool:
    """
    Generate synthetic data if it doesn't exist.
    
    Args:
        force: Force regeneration even if data exists.
    
    Returns:
        True if data was generated or already exists.
    """
    if config.RAW_DATA_FILE.exists() and not force:
        logger.info(f"Data file already exists: {config.RAW_DATA_FILE}")
        
        # Count documents
        with open(config.RAW_DATA_FILE, "r", encoding="utf-8") as f:
            num_docs = sum(1 for _ in f)
        
        logger.info(f"Found {num_docs} documents")
        return True
    
    logger.info(f"Generating {config.NUM_DOCUMENTS} synthetic documents...")
    
    try:
        from data.generate_data import main as generate_main
        generate_main()
        logger.info("Data generation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        return False


def build_index(force: bool = False) -> bool:
    """
    Build FAISS index from data.
    
    Args:
        force: Force rebuild even if index exists.
    
    Returns:
        True if index was built successfully.
    """
    index_exists = config.FAISS_INDEX_PATH.exists()
    
    if index_exists and not force:
        logger.info(f"Index already exists: {config.FAISS_INDEX_PATH}")
        return True
    
    logger.info("Building FAISS index...")
    
    try:
        # Import required modules
        from data.generate_data import load_documents
        from pipeline.chunker import chunk_documents
        from pipeline.embedder import Embedder
        from pipeline.vector_store import build_vector_store
        
        # Load documents
        logger.info(f"Loading documents from {config.RAW_DATA_FILE}...")
        documents = load_documents(config.RAW_DATA_FILE)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Chunk documents
        logger.info("Chunking documents...")
        chunks = chunk_documents(documents)
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Encode chunks
        logger.info("Encoding chunks...")
        embedder = Embedder()
        embeddings, chunk_ids = embedder.encode_chunks(chunks)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Build and save vector store
        logger.info("Building FAISS index...")
        store = build_vector_store(
            embeddings=embeddings,
            chunks=chunks,
            save_path=config.FAISS_INDEX_PATH,
        )
        
        logger.info(f"Index built successfully with {store.size} vectors")
        logger.info(f"Index saved to: {config.FAISS_INDEX_PATH}")
        
        return True
        
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_evaluation() -> bool:
    """
    Run evaluation on the RAG system.
    
    Returns:
        True if evaluation completed successfully.
    """
    logger.info("Running evaluation...")
    
    try:
        from eval.evaluate import main as eval_main
        eval_main()
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False


def launch_ui() -> None:
    """Launch the Gradio UI."""
    logger.info("Launching UI...")
    
    try:
        from ui.app import launch_ui
        launch_ui()
        
    except Exception as e:
        logger.error(f"UI launch failed: {e}")
        import traceback
        traceback.print_exc()


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Paleo RAG — Палеонтологический музей RAG система",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                  # Full pipeline: data → index → UI
    python main.py --skip-data      # Skip data generation
    python main.py --skip-index     # Skip index building
    python main.py --eval-only      # Run evaluation only
    python main.py --force          # Force regenerate data and index
        """,
    )
    
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data generation step",
    )
    
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip index building step",
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (requires existing index)",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate data and rebuild index",
    )
    
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Don't launch UI after building",
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("🦕 Paleo RAG — Палеонтологический Музей")
    print("=" * 60)
    print()
    
    # Step 0: Check environment
    logger.info("Step 0: Checking configuration...")
    check_env_file()
    print()
    
    # Step 1: Generate data
    if args.eval_only:
        logger.info("Skipping data generation (--eval-only)")
    elif args.skip_data and not args.force:
        logger.info("Skipping data generation (--skip-data)")
    else:
        logger.info("Step 1: Generating data...")
        if not generate_data(force=args.force):
            logger.error("Failed to generate data")
            return 1
        print()
    
    # Step 2: Build index
    if args.eval_only:
        logger.info("Skipping index building (--eval-only)")
    elif args.skip_index and not args.force:
        logger.info("Skipping index building (--skip-index)")
    else:
        logger.info("Step 2: Building index...")
        if not build_index(force=args.force):
            logger.error("Failed to build index")
            return 1
        print()
    
    # Step 3: Run evaluation (if requested)
    if args.eval_only:
        logger.info("Step 3: Running evaluation...")
        if not run_evaluation():
            logger.error("Evaluation failed")
            return 1
        return 0
    
    # Step 4: Launch UI
    if args.no_ui:
        logger.info("Skipping UI launch (--no-ui)")
        logger.info("Pipeline completed successfully!")
        return 0
    
    logger.info("Step 4: Launching UI...")
    print()
    print("=" * 60)
    print("🚀 Запуск веб-интерфейса...")
    print("Откройте http://localhost:7860 в браузере")
    print("=" * 60)
    print()
    
    launch_ui()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

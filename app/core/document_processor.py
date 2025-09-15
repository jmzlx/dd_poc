#!/usr/bin/env python3
"""
Streamlined Document Processing Module

This module provides a document processing pipeline with:
- Direct LangChain loader integration with glob patterns
- Built-in FAISS vector storage without external file tracking
- Semantic text chunking using RecursiveCharacterTextSplitter
- Consolidated document metadata handling
"""

import os
import time

# Enable tokenizers parallelism for better performance
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Import configuration and utilities from app modules
from app.core.config import get_app_config
from app.core.model_cache import get_cached_embeddings
from app.core.logging import logger
from app.core.performance import get_performance_manager, monitor_performance, cached_by_content

# Optional accelerate import
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None


# =============================================================================
# ERROR HANDLING UTILITIES - Merged from error_handlers.py
# =============================================================================

def safe_execute(func: Callable, default: Any = None, context: str = "", log_errors: bool = True) -> Any:
    """
    Execute a function with basic error handling and logging

    Args:
        func: Function to execute
        default: Value to return on error
        context: Brief description for logs
        log_errors: Whether to log errors

    Returns:
        Function result or default value on error
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"{context or func.__name__}: {e}")
        return default


def escape_markdown_math(text: str) -> str:
    """Escape dollar signs and other LaTeX-like patterns to prevent Streamlit from interpreting them as math."""
    if not text:
        return text
    # Replace dollar signs with escaped version
    text = text.replace('$', '\\$')
    # Also escape other potential math delimiters
    text = text.replace('\\(', '\\\\(')
    text = text.replace('\\)', '\\\\)')
    text = text.replace('\\[', '\\\\[')
    text = text.replace('\\]', '\\\\]')
    return text


class DocumentProcessor:
    """
    Streamlined document processing class with integrated FAISS vector storage

    This class consolidates all document processing functionality including:
    - Document loading using LangChain's DirectoryLoader with glob patterns
    - Semantic text chunking with RecursiveCharacterTextSplitter
    - FAISS vector storage for similarity search
    - Document metadata handling
    """

    def __init__(self, model_name: Optional[str] = None, store_name: Optional[str] = None):
        """
        Initialize the document processor

        Args:
            model_name: Name of the sentence transformer model for embeddings (optional)
            store_name: Name for the FAISS store (optional, uses config default)
        """
        config = get_app_config()
        self.model_name = model_name or config.model['sentence_transformer_model']
        self.store_name = store_name or config.processing['faiss_store_name']

        # Initialize components
        self.documents: List[Document] = []
        self.vector_store: Optional[FAISS] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self.performance_stats = {}

        # Convenience properties for backward compatibility
        self.chunks = []  # Will be populated after processing

        # Initialize text splitter with semantic boundaries
        self._init_text_splitter()

        # Initialize embeddings if model name provided
        if self.model_name:
            self.embeddings = get_cached_embeddings(self.model_name)
            logger.info(f"Initialized cached embeddings with model: {self.model_name}")

            # Setup accelerate for GPU optimization if available
            if ACCELERATE_AVAILABLE:
                try:
                    self.accelerator = Accelerator()
                    logger.info(f"Accelerate initialized with device: {self.accelerator.device}")
                except Exception as e:
                    logger.warning(f"Failed to initialize accelerate: {e}")
                    self.accelerator = None
            else:
                self.accelerator = None
        else:
            logger.warning("No model name provided - embeddings not initialized")
            self.accelerator = None

        # Try to load existing FAISS store
        self._load_existing_store()

    def _init_text_splitter(self):
        """Initialize the text splitter with optimal settings for semantic chunking"""
        config = get_app_config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.processing['chunk_size'],
            chunk_overlap=config.processing['chunk_overlap'],
            # Better separators for business documents with semantic boundaries
            separators=[
                "\n\n\n",  # Triple newlines (major section breaks)
                "\n\n",    # Double newlines (paragraph breaks)
                "\n",      # Single newlines
                ". ",      # Sentences
                ".\n",     # Sentences with newlines
                "! ",      # Exclamations
                "? ",      # Questions
                "; ",      # Semicolons (common in legal/business docs)
                ", ",      # Commas (last resort for long sentences)
                " ",       # Spaces
                "",        # Character level (absolute last resort)
            ],
            length_function=len,
            is_separator_regex=False,
            # Keep related content together
            keep_separator=True,  # Keep separators to maintain context
        )
        logger.info(f"Initialized semantic text splitter: {config.processing['chunk_size']} chars, {config.processing['chunk_overlap']} overlap")

    def _load_existing_store(self):
        """Load existing FAISS store if available"""
        if not self.embeddings:
            return

        config = get_app_config()
        faiss_dir = config.paths['faiss_dir']
        faiss_index_path = faiss_dir / f"{self.store_name}.faiss"
        faiss_pkl_path = faiss_dir / f"{self.store_name}.pkl"

        try:
            if faiss_index_path.exists() and faiss_pkl_path.exists():
                self.vector_store = FAISS.load_local(
                    str(faiss_dir),
                    self.embeddings,
                    index_name=self.store_name,
                    allow_dangerous_deserialization=True  # Safe: we created these files ourselves
                )
                logger.info(f"Loaded existing FAISS store: {self.store_name} with {self.vector_store.index.ntotal} vectors")
            else:
                logger.info(f"No existing FAISS store found for: {self.store_name}")
        except Exception as e:
            logger.error(f"Failed to load FAISS store: {e}")
            self.vector_store = None

    @monitor_performance
    def load_data_room(self, data_room_path: str, progress_bar=None) -> Dict[str, Any]:
        """
        Load and process an entire data room using DirectoryLoader with glob patterns

        Args:
            data_room_path: Path to the data room directory
            progress_bar: Optional Streamlit progress bar object

        Returns:
            Dictionary with processing results including performance metrics
        """
        import time
        start_time = time.time()

        config = get_app_config()
        data_room_path = Path(data_room_path)

        if not data_room_path.exists():
            logger.error(f"Data room path does not exist: {data_room_path}")
            return {'documents_count': 0, 'chunks_count': 0, 'has_embeddings': False}

        logger.info(f"Starting streamlined data room processing: {data_room_path}")

        # Clear existing documents
        self.documents = []

    @monitor_performance
    def load_data_room(self, data_room_path: str, progress_bar=None) -> Dict[str, Any]:
        start_time = time.time()
        documents_loaded = 0
        config = get_app_config()

        # Load documents by file type using DirectoryLoader with glob patterns
        supported_extensions = config.processing['supported_file_extensions']
        perf_manager = get_performance_manager()

        # Get memory info for batch optimization
        mem_info = perf_manager.monitor_memory_usage()
        logger.info(f"Memory usage at start: {mem_info['percent']:.1f}%")
        logger.info(f"Available memory: {mem_info['rss']:.1f}MB")

        for ext in supported_extensions:
            try:
                # Create glob pattern for this extension
                glob_pattern = f"**/*{ext}"

                # Choose appropriate loader based on extension
                if ext == '.pdf':
                    loader_cls = PyPDFLoader
                elif ext in ['.docx', '.doc']:
                    loader_cls = Docx2txtLoader
                elif ext in ['.txt', '.md']:
                    loader_cls = TextLoader
                else:
                    continue

                # Use DirectoryLoader with glob pattern
                loader = DirectoryLoader(
                    str(data_room_path),
                    glob=glob_pattern,
                    loader_cls=loader_cls,
                    loader_kwargs={'encoding': 'utf-8'} if ext in ['.txt', '.md'] else {},
                    recursive=True,
                    show_progress=False,  # Disable verbose progress output
                    use_multithreading=True
                )

                # Load documents for this extension
                docs = safe_execute(
                    lambda: loader.load(),
                    default=[],
                    context=f"Loading {ext} files"
                )

                if docs:
                    # Add relative path information to metadata
                    for doc in docs:
                        if 'source' in doc.metadata:
                            source_path = Path(doc.metadata['source'])
                            if source_path.exists():
                                try:
                                    rel_path = source_path.relative_to(data_room_path)
                                    doc.metadata['path'] = str(rel_path)
                                    doc.metadata['name'] = source_path.name
                                except ValueError:
                                    # If relative path fails, use original source
                                    doc.metadata['path'] = doc.metadata['source']
                                    doc.metadata['name'] = source_path.name

                    self.documents.extend(docs)
                    documents_loaded += len(docs)
                    logger.info(f"Loaded {len(docs)} {ext} documents")

                    # Monitor memory usage and trigger GC if needed
                    mem_usage = perf_manager.monitor_memory_usage()
                    if perf_manager.should_gc_collect(mem_usage):
                        import gc
                        gc.collect()
                        logger.debug(f"GC triggered - memory usage: {mem_usage['rss']:.1f}MB")
            except Exception as e:
                logger.error(f"Error loading {ext} files: {e}")

        scan_time = time.time() - start_time
        logger.info(f"Document loading completed in {scan_time:.2f} seconds")

        # Split documents into chunks using the text splitter
        chunk_start = time.time()
        if self.documents and self.text_splitter:
            # Track original documents to identify first chunks
            original_docs = {doc.metadata.get('source', ''): True for doc in self.documents}

            self.documents = self.text_splitter.split_documents(self.documents)

            # Add chunk metadata and populate chunks for backward compatibility
            # Track which documents we've seen to mark first chunks
            seen_documents = {}
            self.chunks = []

            for i, doc in enumerate(self.documents):
                doc.metadata['chunk_id'] = f"chunk_{i}"
                doc.metadata['processed_at'] = datetime.now().isoformat()

                # Mark first chunks for each document (critical for document type matching)
                doc_source = doc.metadata.get('source', '')
                if doc_source not in seen_documents:
                    doc.metadata['is_first_chunk'] = True
                    seen_documents[doc_source] = True
                    logger.debug(f"First chunk marked for: {doc_source}")
                else:
                    doc.metadata['is_first_chunk'] = False

                # Add citation information if available
                if 'page' in doc.metadata:
                    doc.metadata['citation'] = f"page {doc.metadata['page']}"
                else:
                    doc.metadata['citation'] = doc.metadata.get('name', 'document')

                # Create chunk dict for backward compatibility
                chunk_dict = {
                    'text': doc.page_content,
                    'source': doc.metadata.get('name', ''),
                    'path': doc.metadata.get('path', ''),
                    'full_path': doc.metadata.get('source', ''),
                    'metadata': doc.metadata
                }
                self.chunks.append(chunk_dict)

            first_chunks_count = len([doc for doc in self.documents if doc.metadata.get('is_first_chunk', False)])
            logger.info(f"Marked {first_chunks_count} first chunks out of {len(self.documents)} total chunks")

        chunk_time = time.time() - chunk_start
        logger.info(f"Text splitting completed in {chunk_time:.2f} seconds")

                # FAISS vector store should be loaded from pre-built indices
        embedding_time = 0
        if self.embeddings and self.documents:
            embedding_start = time.time()

            if self.vector_store is None:
                logger.debug("FAISS store not pre-loaded (expected during index building)")
            else:
                logger.info(f"Using pre-loaded FAISS store with {self.vector_store.index.ntotal} vectors")

            embedding_time = time.time() - embedding_start
            logger.info(f"FAISS check completed in {embedding_time:.2f} seconds")

        total_time = time.time() - start_time
        logger.info(f"Total data room processing completed in {total_time:.2f} seconds")

        # Store performance stats
        self.performance_stats = {
            'total_time': total_time,
            'scan_time': scan_time,
            'chunk_time': chunk_time,
            'embedding_time': embedding_time,
            'documents_per_second': documents_loaded / scan_time if scan_time > 0 else 0
        }

        return {
            'documents_count': documents_loaded,
            'chunks_count': len(self.documents),
            'total_chunks_in_store': self.vector_store.index.ntotal if self.vector_store else 0,
            'has_embeddings': self.vector_store is not None,
            'performance': self.performance_stats
        }

    def search(self, query: str, top_k: int = 5, threshold: Optional[float] = None) -> List[Dict]:
        """
        Search documents using FAISS similarity search

        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of search results with scores and metadata
        """
        if not self.vector_store:
            logger.warning("FAISS vector store not available for search")
            return []

        config = get_app_config()
        if threshold is None:
            threshold = config.processing['similarity_threshold']

        try:
            # Perform similarity search with scores - get more candidates for reranking
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=max(20, top_k*3))

            # VECTORIZED: Initial filtering and conversion to candidates format
            import numpy as np
            
            # Extract documents and scores for vectorized processing
            docs = [doc for doc, score in docs_and_scores]
            scores = np.array([score for doc, score in docs_and_scores])
            
            # VECTORIZED: Convert FAISS distances to similarity scores in batch
            similarity_scores = np.where(scores <= 2.0, 1.0 - (scores / 2.0), 0.0)
            
            # VECTORIZED: Filter by threshold using boolean mask
            threshold_mask = similarity_scores >= threshold
            valid_indices = np.where(threshold_mask)[0]
            
            # Build candidates list for all valid documents (no duplicate filtering needed)
            # Note: Removed duplicate checking as it was removing valuable overlapping chunks
            # that are intentionally created by the 200-character chunk overlap setting
            candidates = []
            
            for idx in valid_indices:
                doc = docs[idx]
                similarity_score = similarity_scores[idx]

                candidates.append({
                    'text': doc.page_content,
                    'source': doc.metadata.get('name', ''),
                    'path': doc.metadata.get('path', ''),
                    'score': float(similarity_score),
                    'metadata': doc.metadata
                })

            # Apply reranking if we have candidates
            if candidates:
                try:
                    # Import rerank_results from ranking module to avoid circular import
                    from app.core.ranking import rerank_results

                    # Rerank the top candidates (limit to reasonable number for performance)
                    candidates_to_rerank = candidates[:min(15, len(candidates))]  # Rerank up to 15 candidates

                    reranked_results = rerank_results(query, candidates_to_rerank)
                    results = reranked_results[:top_k]  # Take top_k after reranking
                    logger.info(f"Reranked {len(reranked_results)} search results for query: {query[:50]}...")
                except Exception as e:
                    # Reranking failed - use original results without reranking
                    logger.warning(f"Reranking failed for search query '{query}': {e}. Using original similarity scores.")
                    results = candidates[:top_k]
            else:
                results = []

            return results

        except Exception as e:
            logger.error(f"Failed to search FAISS store: {e}")
            raise RuntimeError(f"Document search failed for query '{query}': {e}") from e

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = {
            'total_documents': len(self.documents),
            'total_vectors_in_store': self.vector_store.index.ntotal if self.vector_store else 0,
            'has_embeddings': self.vector_store is not None,
            'store_name': self.store_name,
            'model_name': self.model_name
        }

        # Add performance metrics if available
        if self.performance_stats:
            stats['performance'] = self.performance_stats

        return stats

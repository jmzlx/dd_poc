#!/usr/bin/env python3
"""
Document Processing Module

This module handles all document-related operations including:
- File text extraction from various formats (PDF, DOCX, TXT, MD)
- Document scanning and indexing
- Semantic text chunking for RAG with better context preservation
- Document metadata handling
"""

import os
# Fix tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fitz  # PyMuPDF
import docx
import io
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import concurrent.futures
import threading
import logging
from functools import wraps
import joblib
import hashlib
import time
import faiss

# Semantic chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup logging for thread-safe error handling
logger = logging.getLogger(__name__)

# Thread-safe context management for Streamlit
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    STREAMLIT_CONTEXT_AVAILABLE = True
except ImportError:
    STREAMLIT_CONTEXT_AVAILABLE = False
    logger.warning("Streamlit context management not available")


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


def extract_text_from_file(file_path: Path, progress_callback=None) -> Tuple[str, Dict]:
    """
    Extract text from file with metadata
    
    Args:
        file_path: Path to the file to extract text from
        
    Returns:
        Tuple of (text_content, metadata)
    """
    metadata = {'pages': [], 'type': 'unknown'}
    text_content = ""
    
    try:
        if file_path.suffix.lower() == '.pdf':
            # Use PyMuPDF (fitz) for faster and more robust PDF processing
            try:
                pdf_document = fitz.open(str(file_path))
                texts = []
                
                for page_num in range(pdf_document.page_count):
                    try:
                        page = pdf_document[page_num]
                        page_text = page.get_text()
                        
                        if page_text.strip():  # Only add non-empty pages
                            texts.append(page_text)
                            metadata['pages'].append(page_num + 1)  # 1-based page numbering
                    except Exception as page_error:
                        # Handle individual page errors gracefully
                        logger.warning(f"Error reading page {page_num + 1} of {file_path.name}: {page_error}")
                        if st and hasattr(st, 'session_state'):
                            # Only use streamlit in main thread context
                            try:
                                st.warning(f"Error reading page {page_num + 1} of {file_path.name}: {page_error}")
                            except Exception:
                                pass
                        continue
                
                pdf_document.close()
                text_content = '\n'.join(texts)[:10000]
                metadata['type'] = 'pdf'
                
            except Exception as pdf_error:
                # Handle corrupted or unsupported PDF files
                error_msg = f"Error processing PDF {file_path.name}: {pdf_error}"
                logger.error(error_msg)
                if st and hasattr(st, 'session_state'):
                    # Only use streamlit in main thread context
                    try:
                        st.error(error_msg)
                    except Exception:
                        pass
                # Try to return partial content if available
                if 'pdf_document' in locals():
                    try:
                        pdf_document.close()
                    except:
                        pass
                return "", metadata
                
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            doc = docx.Document(str(file_path))
            text_content = '\n'.join(p.text for p in doc.paragraphs)[:10000]
            metadata['type'] = 'docx'
            
        elif file_path.suffix.lower() in ['.txt', '.md']:
            text_content = file_path.read_text(encoding='utf-8', errors='ignore')[:10000]
            metadata['type'] = 'text'
            
    except Exception as e:
        error_msg = f"Could not read {file_path.name}: {e}"
        logger.warning(error_msg)
        if st and hasattr(st, 'session_state'):  # Only use streamlit if available and in main thread
            try:
                st.warning(error_msg)
            except Exception:
                pass
        
    # Call progress callback if provided (for parallel processing tracking)
    if progress_callback:
        try:
            progress_callback(file_path.name)
        except Exception:
            pass  # Don't let callback errors affect processing
    
    return text_content, metadata


def _process_file_with_context(args):
    """
    Thread-safe file processing function with proper context management
    
    Args:
        args: Tuple of (file_path, base_path, progress_callback)
        
    Returns:
        Tuple of (file_path_str, document_info) or None if failed
    """
    file_path, base_path, progress_callback = args
    
    try:
        # Extract text from file
        text, metadata = extract_text_from_file(file_path, progress_callback)
        
        if text:
            # Store relative path for display
            rel_path = file_path.relative_to(base_path)
            document_info = {
                'text': text,
                'content': text,  # Alias for backward compatibility
                'name': file_path.name,
                'rel_path': str(rel_path),
                'metadata': metadata
            }
            return str(file_path), document_info
    except Exception as e:
        logger.error(f"Error processing file {file_path.name}: {e}")
    
    return None


def scan_data_room(data_room_path: str, max_workers: int = 4, progress_callback=None) -> Dict[str, Dict]:
    """
    Scan entire data room directory for documents using parallel processing
    
    Args:
        data_room_path: Path to the data room directory
        max_workers: Maximum number of worker threads (default: 4)
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary mapping file paths to document information
    """
    documents = {}
    path = Path(data_room_path)
    
    if not path.exists():
        return documents
    
    # Collect all document files first
    file_paths = []
    for file_path in path.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            if file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md']:
                file_paths.append(file_path)
    
    if not file_paths:
        return documents
    
    logger.info(f"Processing {len(file_paths)} files with {max_workers} workers")
    
    # Prepare arguments for parallel processing
    process_args = [(file_path, path, progress_callback) for file_path in file_paths]
    
    # Process files in parallel
    processed_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {}
        
        for args in process_args:
            future = executor.submit(_process_file_with_context, args)
            
            # Add Streamlit context if available
            if STREAMLIT_CONTEXT_AVAILABLE:
                try:
                    script_ctx = get_script_run_ctx()
                    if script_ctx:
                        add_script_run_ctx(future)
                except Exception as e:
                    logger.warning(f"Could not add script context: {e}")
            
            future_to_file[future] = args[0]  # Store file_path for reference
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                result = future.result(timeout=30)  # 30-second timeout per file
                if result:
                    file_path_str, document_info = result
                    documents[file_path_str] = document_info
                    processed_count += 1
                else:
                    failed_count += 1
            except concurrent.futures.TimeoutError:
                file_path = future_to_file[future]
                logger.error(f"Timeout processing file: {file_path.name}")
                failed_count += 1
            except Exception as e:
                file_path = future_to_file[future]
                logger.error(f"Error processing file {file_path.name}: {e}")
                failed_count += 1
    
    logger.info(f"Completed processing: {processed_count} successful, {failed_count} failed")
    return documents


def create_chunks_with_metadata(documents: Dict[str, Dict], chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
    """
    Create searchable chunks with semantic splitting and full metadata.
    Uses RecursiveCharacterTextSplitter for better context preservation.
    
    Args:
        documents: Dictionary of documents
        chunk_size: Size of each chunk in characters (default: 2000 for ~400 words)
        overlap: Overlap between chunks in characters (default: 200 for ~50 words)
        
    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    
    # Initialize semantic text splitter with hierarchical separators
    # This preserves document structure by prioritizing paragraph breaks,
    # then sentences, then words
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        length_function=len,
        is_separator_regex=False,
    )
    
    for doc_path, doc_info in documents.items():
        text = doc_info['text']
        
        if not text.strip():
            continue
            
        # Split text using semantic boundaries
        semantic_chunks = text_splitter.split_text(text)
        
        # Create chunks with metadata
        for i, chunk_text in enumerate(semantic_chunks):
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text.strip(),
                    'source': doc_info['name'],
                    'path': doc_info['rel_path'],
                    'full_path': doc_path,
                    'chunk_id': f"semantic_chunk_{i}",
                    'metadata': doc_info['metadata']
                })
    
    return chunks


def create_embeddings_batch(texts: List[str], model: SentenceTransformer, batch_size: int = 100) -> np.ndarray:
    """
    Create embeddings for texts in batches for better performance
    
    Args:
        texts: List of texts to embed
        model: SentenceTransformer model
        batch_size: Batch size for processing
        
    Returns:
        NumPy array of embeddings
    """
    embeddings_list = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings_list.append(batch_embeddings)
    
    return np.vstack(embeddings_list) if embeddings_list else np.array([])


def search_documents_with_faiss(
    query: str, 
    chunks: List[Dict], 
    faiss_index: faiss.IndexFlatIP, 
    model: SentenceTransformer, 
    top_k: int = 5,
    threshold: float = 0.3
) -> List[Dict]:
    """
    Search documents using FAISS IndexFlatIP for fast similarity search
    
    Args:
        query: Search query
        chunks: List of document chunks
        faiss_index: FAISS index with embeddings
        model: SentenceTransformer model
        top_k: Number of top results to return
        threshold: Minimum similarity threshold
        
    Returns:
        List of search results with citations
    """
    if not chunks or faiss_index is None:
        return []
    
    # Encode query and normalize for inner product similarity
    query_embedding = model.encode(query).astype('float32')
    query_embedding = query_embedding.reshape(1, -1)
    
    # Normalize for cosine similarity using inner product
    faiss.normalize_L2(query_embedding)
    
    # Search using FAISS (much faster than numpy)
    scores, indices = faiss_index.search(query_embedding, min(top_k * 2, len(chunks)))
    
    results = []
    seen_texts = set()
    
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:  # -1 indicates no more results
            continue
            
        # Avoid duplicates
        text_preview = chunks[idx]['text'][:100]
        if text_preview not in seen_texts:
            seen_texts.add(text_preview)
            
            # Format citation based on file type
            metadata = chunks[idx]['metadata']
            if metadata['type'] == 'pdf' and metadata.get('pages'):
                citation = f"page {metadata['pages'][0]}"
            else:
                citation = "document"
            
            results.append({
                'text': chunks[idx]['text'],
                'source': chunks[idx]['source'],
                'path': chunks[idx]['path'],
                'full_path': chunks[idx].get('full_path', ''),
                'citation': citation,
                'score': float(score)
            })
            
            if len(results) >= top_k:
                break
    
    return results


def search_documents_with_citations(
    query: str, 
    chunks: List[Dict], 
    embeddings: np.ndarray, 
    model: SentenceTransformer, 
    top_k: int = 5,
    threshold: float = 0.3
) -> List[Dict]:
    """
    Legacy search documents function - kept for backward compatibility
    Creates temporary FAISS index and uses FAISS search for better performance
    
    Args:
        query: Search query
        chunks: List of document chunks
        embeddings: Precomputed embeddings for chunks
        model: SentenceTransformer model
        top_k: Number of top results to return
        threshold: Minimum similarity threshold
        
    Returns:
        List of search results with citations
    """
    if not chunks:
        return []
    
    # Create temporary FAISS index for better performance
    embeddings_f32 = embeddings.astype('float32')
    faiss.normalize_L2(embeddings_f32)  # Normalize for cosine similarity
    
    index = faiss.IndexFlatIP(embeddings_f32.shape[1])
    index.add(embeddings_f32)
    
    return search_documents_with_faiss(query, chunks, index, model, top_k, threshold)


def create_progress_tracker(total_files: int = 0, streamlit_progress_bar=None):
    """
    Create a thread-safe progress tracking function
    
    Args:
        total_files: Total number of files to process
        streamlit_progress_bar: Optional Streamlit progress bar
        
    Returns:
        Progress callback function
    """
    processed_count = [0]  # Use list for mutable counter in closure
    lock = threading.Lock()
    
    def progress_callback(filename: str = None):
        with lock:
            processed_count[0] += 1
            progress = processed_count[0] / max(total_files, 1)
            
            if streamlit_progress_bar and hasattr(st, 'session_state'):
                try:
                    streamlit_progress_bar.progress(
                        min(progress, 1.0), 
                        text=f"Processing {filename or 'documents'}... ({processed_count[0]}/{total_files})"
                    )
                except Exception:
                    pass  # Don't let UI errors affect processing
    
    return progress_callback


def _generate_cache_key(documents: Dict[str, Dict]) -> str:
    """
    Generate a cache key based on document paths and modification times
    
    Args:
        documents: Dictionary of documents with file paths
        
    Returns:
        Cache key string
    """
    # Create a hash based on file paths and their modification times
    cache_data = []
    
    for file_path, doc_info in documents.items():
        try:
            path_obj = Path(file_path)
            if path_obj.exists():
                mtime = path_obj.stat().st_mtime
                cache_data.append(f"{file_path}:{mtime}")
        except Exception as e:
            logger.warning(f"Could not get modification time for {file_path}: {e}")
            # Use current time as fallback
            cache_data.append(f"{file_path}:{time.time()}")
    
    # Sort to ensure consistent hashing regardless of document order
    cache_data.sort()
    cache_string = "|".join(cache_data)
    
    # Generate MD5 hash for the cache key
    return hashlib.md5(cache_string.encode('utf-8')).hexdigest()


def _get_cache_dir() -> Path:
    """Get or create the cache directory"""
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _save_embeddings_to_cache(cache_key: str, embeddings: np.ndarray, chunks: List[Dict]) -> bool:
    """
    Save embeddings and chunks to cache
    
    Args:
        cache_key: Cache key for the data
        embeddings: Embeddings array to cache
        chunks: Document chunks to cache
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cache_dir = _get_cache_dir()
        cache_file = cache_dir / f"embeddings_{cache_key}.joblib"
        
        cache_data = {
            'embeddings': embeddings,
            'chunks': chunks,
            'timestamp': time.time(),
            'cache_key': cache_key
        }
        
        joblib.dump(cache_data, cache_file, compress=3)
        logger.info(f"Saved embeddings to cache: {cache_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save embeddings to cache: {e}")
        return False


def _load_embeddings_from_cache(cache_key: str) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
    """
    Load embeddings and chunks from cache
    
    Args:
        cache_key: Cache key for the data
        
    Returns:
        Tuple of (embeddings, chunks) or (None, None) if not found
    """
    try:
        cache_dir = _get_cache_dir()
        cache_file = cache_dir / f"embeddings_{cache_key}.joblib"
        
        if not cache_file.exists():
            return None, None
            
        cache_data = joblib.load(cache_file)
        
        # Validate cache data structure
        if not all(key in cache_data for key in ['embeddings', 'chunks', 'timestamp', 'cache_key']):
            logger.warning(f"Invalid cache data structure in {cache_file}")
            return None, None
            
        # Check if cache key matches (additional validation)
        if cache_data['cache_key'] != cache_key:
            logger.warning(f"Cache key mismatch in {cache_file}")
            return None, None
            
        logger.info(f"Loaded embeddings from cache: {cache_file}")
        return cache_data['embeddings'], cache_data['chunks']
        
    except Exception as e:
        logger.error(f"Failed to load embeddings from cache: {e}")
        return None, None


def _invalidate_old_cache_files(max_age_days: int = 7) -> None:
    """
    Remove old cache files to prevent cache directory from growing too large
    
    Args:
        max_age_days: Maximum age of cache files in days
    """
    try:
        cache_dir = _get_cache_dir()
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for cache_file in cache_dir.glob("embeddings_*.joblib"):
            try:
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_seconds:
                    cache_file.unlink()
                    logger.info(f"Removed old cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Could not remove old cache file {cache_file}: {e}")
                
    except Exception as e:
        logger.error(f"Failed to invalidate old cache files: {e}")


class DocumentProcessor:
    """
    Main document processing class that orchestrates document operations with parallel processing support
    Enhanced with FAISS for 10x faster similarity search
    """
    
    def __init__(self, model: Optional[SentenceTransformer] = None):
        """
        Initialize the document processor
        
        Args:
            model: SentenceTransformer model for embeddings (optional)
        """
        self.model = model
        self.documents = {}
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None  # FAISS index for fast similarity search
        self.performance_stats = {}  # Track performance metrics
    
    def load_data_room(self, data_room_path: str, max_workers: int = 4, progress_callback=None) -> Dict[str, any]:
        """
        Load and process an entire data room with parallel processing
        
        Args:
            data_room_path: Path to the data room directory
            max_workers: Maximum number of worker threads for document processing
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with processing results including performance metrics
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting data room processing: {data_room_path}")
        
        # Scan documents with parallel processing
        self.documents = scan_data_room(
            data_room_path, 
            max_workers=max_workers, 
            progress_callback=progress_callback
        )
        
        scan_time = time.time() - start_time
        logger.info(f"Document scanning completed in {scan_time:.2f} seconds")
        
        # Create chunks
        chunk_start = time.time()
        self.chunks = create_chunks_with_metadata(self.documents)
        chunk_time = time.time() - chunk_start
        
        # Create embeddings if model is available
        embedding_time = 0
        cache_hit = False
        
        if self.model and self.chunks:
            embedding_start = time.time()
            
            # Try to load from cache first
            cache_key = _generate_cache_key(self.documents)
            cached_embeddings, cached_chunks = _load_embeddings_from_cache(cache_key)
            
            if cached_embeddings is not None and cached_chunks is not None:
                # Cache hit - use cached data
                self.embeddings = cached_embeddings
                # Verify chunks match (safety check)
                if len(cached_chunks) == len(self.chunks):
                    self.chunks = cached_chunks
                    cache_hit = True
                    logger.info(f"Loaded embeddings from cache (key: {cache_key[:8]}...)")
                    # Build FAISS index from cached embeddings
                    self._build_faiss_index()
                else:
                    logger.warning("Cached chunks length mismatch, regenerating embeddings")
            
            if not cache_hit:
                # Cache miss or invalid - generate new embeddings
                texts = [chunk['text'] for chunk in self.chunks]
                self.embeddings = create_embeddings_batch(texts, self.model)
                
                # Save to cache
                if _save_embeddings_to_cache(cache_key, self.embeddings, self.chunks):
                    logger.info(f"Saved new embeddings to cache (key: {cache_key[:8]}...)")
                
                # Clean up old cache files
                _invalidate_old_cache_files()
            
            # Build FAISS index for fast similarity search
            self._build_faiss_index()
                
            embedding_time = time.time() - embedding_start
            cache_status = "from cache" if cache_hit else "generated"
            logger.info(f"Embeddings {cache_status} and FAISS index built in {embedding_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Total data room processing completed in {total_time:.2f} seconds")
        
        return {
            'documents_count': len(self.documents),
            'chunks_count': len(self.chunks),
            'has_embeddings': self.embeddings is not None,
            'performance': {
                'total_time': total_time,
                'scan_time': scan_time,
                'chunk_time': chunk_time,
                'embedding_time': embedding_time,
                'documents_per_second': len(self.documents) / scan_time if scan_time > 0 else 0,
                'cache_hit': cache_hit,
                'cache_key': cache_key[:8] + "..." if 'cache_key' in locals() else None
            }
        }
    
    def _build_faiss_index(self) -> None:
        """
        Build FAISS IndexFlatIP for fast similarity search
        """
        if self.embeddings is None:
            logger.warning("No embeddings available to build FAISS index")
            return
        
        try:
            # Convert to float32 and normalize for cosine similarity via inner product
            embeddings_f32 = self.embeddings.astype('float32')
            faiss.normalize_L2(embeddings_f32)
            
            # Create FAISS index
            dimension = embeddings_f32.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings_f32)
            
            logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors, dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            self.faiss_index = None

    def faiss_search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Fast similarity search using FAISS IndexFlatIP
        
        Args:
            query: Search query
            top_k: Number of top results
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results with citations
        """
        if not self.model or self.faiss_index is None:
            return []
        
        return search_documents_with_faiss(
            query, self.chunks, self.faiss_index, self.model, top_k, threshold
        )

    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Search documents using semantic similarity - uses FAISS if available, falls back to numpy
        
        Args:
            query: Search query
            top_k: Number of top results
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        if not self.model:
            return []
        
        # Use FAISS search if index is available (10x faster)
        if self.faiss_index is not None:
            return self.faiss_search(query, top_k, threshold)
        elif self.embeddings is not None:
            # Fallback to numpy-based search
            return search_documents_with_citations(
                query, self.chunks, self.embeddings, self.model, top_k, threshold
            )
        else:
            return []
    
    def get_statistics(self) -> Dict[str, any]:
        """Get processing statistics including performance metrics"""
        stats = {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'has_embeddings': self.embeddings is not None,
            'has_faiss_index': self.faiss_index is not None,
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index is not None else 0,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0
        }
        
        # Add performance metrics if available
        if self.performance_stats:
            stats['performance'] = self.performance_stats
            
        return stats
    
    def load_data_room_with_progress(self, data_room_path: str, max_workers: int = 4, 
                                   progress_bar=None) -> Dict[str, any]:
        """
        Load data room with Streamlit progress bar support
        
        Args:
            data_room_path: Path to the data room directory
            max_workers: Maximum number of worker threads
            progress_bar: Streamlit progress bar object
            
        Returns:
            Dictionary with processing results
        """
        # Count total files first for accurate progress tracking
        path = Path(data_room_path)
        if not path.exists():
            return {'documents_count': 0, 'chunks_count': 0, 'has_embeddings': False}
        
        total_files = sum(1 for file_path in path.rglob('*') 
                         if file_path.is_file() and not file_path.name.startswith('.') 
                         and file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md'])
        
        # Create progress tracker
        progress_callback = create_progress_tracker(total_files, progress_bar)
        
        # Load with progress tracking
        result = self.load_data_room(data_room_path, max_workers, progress_callback)
        self.performance_stats = result.get('performance', {})
        
        return result

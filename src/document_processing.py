#!/usr/bin/env python3
"""
Document Processing Module

This module handles all document-related operations including:
- File text extraction from various formats (PDF, DOCX, TXT, MD)
- Document scanning and indexing
- Text chunking for RAG
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


def create_chunks_with_metadata(documents: Dict[str, Dict], chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """
    Create searchable chunks with full metadata
    
    Args:
        documents: Dictionary of documents
        chunk_size: Size of each chunk in words
        overlap: Overlap between chunks in words
        
    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    
    for doc_path, doc_info in documents.items():
        text = doc_info['text']
        words = text.split()
        
        # Create overlapping chunks
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = ' '.join(words[i:i + chunk_size])
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'source': doc_info['name'],
                    'path': doc_info['rel_path'],
                    'full_path': doc_path,
                    'chunk_id': f"chunk_{i}",
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


def search_documents_with_citations(
    query: str, 
    chunks: List[Dict], 
    embeddings: np.ndarray, 
    model: SentenceTransformer, 
    top_k: int = 5,
    threshold: float = 0.3
) -> List[Dict]:
    """
    Search documents and return with citations
    
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
    
    query_embedding = model.encode(query)
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = []
    seen_texts = set()
    
    for idx in top_indices:
        if similarities[idx] > threshold:
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
                    'score': float(similarities[idx])
                })
    
    return results


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


class DocumentProcessor:
    """
    Main document processing class that orchestrates document operations with parallel processing support
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
        if self.model and self.chunks:
            embedding_start = time.time()
            texts = [chunk['text'] for chunk in self.chunks]
            self.embeddings = create_embeddings_batch(texts, self.model)
            embedding_time = time.time() - embedding_start
            logger.info(f"Embeddings created in {embedding_time:.2f} seconds")
        
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
                'documents_per_second': len(self.documents) / scan_time if scan_time > 0 else 0
            }
        }
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Search documents using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of top results
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        if not self.model or self.embeddings is None:
            return []
        
        return search_documents_with_citations(
            query, self.chunks, self.embeddings, self.model, top_k, threshold
        )
    
    def get_statistics(self) -> Dict[str, any]:
        """Get processing statistics including performance metrics"""
        stats = {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'has_embeddings': self.embeddings is not None,
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

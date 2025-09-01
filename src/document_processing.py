#!/usr/bin/env python3
"""
Streamlined Document Processing Module

This module provides a simplified document processing pipeline with:
- Direct LangChain loader integration with glob patterns  
- Built-in FAISS vector storage without external file tracking
- Semantic text chunking using RecursiveCharacterTextSplitter
- Consolidated document metadata handling
"""

import os
import logging

# Fix tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import re

from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import configuration
from .config import get_config

# Import error handling  


logger = logging.getLogger(__name__)


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
        config = get_config()
        self.model_name = model_name or config.model.sentence_transformer_model
        self.store_name = store_name or config.processing.faiss_store_name
        
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
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            logger.info(f"Initialized embeddings with model: {self.model_name}")
        else:
            logger.warning("No model name provided - embeddings not initialized")
        
        # Try to load existing FAISS store
        self._load_existing_store()
    
    def _init_text_splitter(self):
        """Initialize the text splitter with optimal settings for semantic chunking"""
        config = get_config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.processing.chunk_size,
            chunk_overlap=config.processing.chunk_overlap,
            separators=["\\n\\n", "\\n", ".", "!", "?", ",", " "],
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(f"Initialized text splitter: {config.processing.chunk_size} chars, {config.processing.chunk_overlap} overlap")
    
    def _load_existing_store(self):
        """Load existing FAISS store if available"""
        if not self.embeddings:
            return
        
        config = get_config()
        faiss_dir = Path(config.paths.data_dir) / "enhanced_faiss"
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
    
    def _save_store(self):
        """Save FAISS store to disk"""
        if not self.vector_store:
            return
        
        try:
            config = get_config()
            faiss_dir = Path(config.paths.data_dir) / "enhanced_faiss"
            faiss_dir.mkdir(parents=True, exist_ok=True)
            
            self.vector_store.save_local(
                str(faiss_dir),
                index_name=self.store_name
            )
            logger.info(f"Saved FAISS store: {self.store_name} with {self.vector_store.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save FAISS store: {e}")
    
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
        
        config = get_config()
        data_room_path = Path(data_room_path)
        
        if not data_room_path.exists():
            logger.error(f"Data room path does not exist: {data_room_path}")
            return {'documents_count': 0, 'chunks_count': 0, 'has_embeddings': False}
        
        logger.info(f"Starting streamlined data room processing: {data_room_path}")
        
        # Clear existing documents
        self.documents = []
        documents_loaded = 0
        
        # Load documents by file type using DirectoryLoader with glob patterns
        supported_extensions = config.processing.supported_file_extensions
        
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
                
            except Exception as e:
                logger.error(f"Error loading {ext} files: {e}")
        
        scan_time = time.time() - start_time
        logger.info(f"Document loading completed in {scan_time:.2f} seconds")
        
        # Split documents into chunks using the text splitter
        chunk_start = time.time()
        if self.documents and self.text_splitter:
            self.documents = self.text_splitter.split_documents(self.documents)
            
            # Add chunk metadata and populate chunks for backward compatibility
            self.chunks = []
            for i, doc in enumerate(self.documents):
                doc.metadata['chunk_id'] = f"chunk_{i}"
                doc.metadata['processed_at'] = datetime.now().isoformat()
                
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
        
        chunk_time = time.time() - chunk_start
        logger.info(f"Text splitting completed in {chunk_time:.2f} seconds")
        
        # Create or update FAISS vector store
        embedding_time = 0
        if self.embeddings and self.documents:
            embedding_start = time.time()
            
            if self.vector_store is None:
                # Create new FAISS store
                self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
                logger.info(f"Created new FAISS store with {len(self.documents)} documents")
            else:
                # Add documents to existing store
                self.vector_store.add_documents(self.documents)
                logger.info(f"Added {len(self.documents)} documents to existing FAISS store")
            
            # Save the updated store
            self._save_store()
            
            embedding_time = time.time() - embedding_start
            logger.info(f"FAISS processing completed in {embedding_time:.2f} seconds")
        
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
        
        config = get_config()
        if threshold is None:
            threshold = config.processing.similarity_threshold
        
        try:
            # Perform similarity search with scores
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=top_k*2)
            
            results = []
            seen_texts = set()
            
            for doc, score in docs_and_scores:
                # Convert FAISS distance to similarity score (higher is better)
                similarity_score = 1.0 / (1.0 + score) if score >= 0 else 1.0
                
                if similarity_score < threshold:
                    continue
                
                # Avoid duplicates based on text content
                text_preview = doc.page_content[:100]
                if text_preview not in seen_texts:
                    seen_texts.add(text_preview)
                    
                    results.append({
                        'text': doc.page_content,
                        'source': doc.metadata.get('name', ''),
                        'path': doc.metadata.get('path', ''),
                        'full_path': doc.metadata.get('source', ''),
                        'citation': doc.metadata.get('citation', 'document'),
                        'score': float(similarity_score),
                        'metadata': doc.metadata
                    })
                    
                    if len(results) >= top_k:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search FAISS store: {e}")
            return []
    
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
    

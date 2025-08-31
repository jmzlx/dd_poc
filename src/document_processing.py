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


def extract_text_from_file(file_path: Path) -> Tuple[str, Dict]:
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
                        if st:
                            st.warning(f"Error reading page {page_num + 1} of {file_path.name}: {page_error}")
                        continue
                
                pdf_document.close()
                text_content = '\n'.join(texts)[:10000]
                metadata['type'] = 'pdf'
                
            except Exception as pdf_error:
                # Handle corrupted or unsupported PDF files
                error_msg = f"Error processing PDF {file_path.name}: {pdf_error}"
                if st:
                    st.error(error_msg)
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
        if st:  # Only use streamlit if available
            st.warning(error_msg)
        
    return text_content, metadata


def scan_data_room(data_room_path: str) -> Dict[str, Dict]:
    """
    Scan entire data room directory for documents
    
    Args:
        data_room_path: Path to the data room directory
        
    Returns:
        Dictionary mapping file paths to document information
    """
    documents = {}
    path = Path(data_room_path)
    
    if not path.exists():
        return documents
    
    # Scan all files recursively
    for file_path in path.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            if file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md']:
                text, metadata = extract_text_from_file(file_path)
                if text:
                    # Store relative path for display
                    rel_path = file_path.relative_to(path)
                    documents[str(file_path)] = {
                        'text': text,
                        'content': text,  # Alias for backward compatibility
                        'name': file_path.name,
                        'rel_path': str(rel_path),
                        'metadata': metadata
                    }
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


class DocumentProcessor:
    """
    Main document processing class that orchestrates document operations
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
    
    def load_data_room(self, data_room_path: str) -> Dict[str, any]:
        """
        Load and process an entire data room
        
        Args:
            data_room_path: Path to the data room directory
            
        Returns:
            Dictionary with processing results
        """
        # Scan documents
        self.documents = scan_data_room(data_room_path)
        
        # Create chunks
        self.chunks = create_chunks_with_metadata(self.documents)
        
        # Create embeddings if model is available
        if self.model and self.chunks:
            texts = [chunk['text'] for chunk in self.chunks]
            self.embeddings = create_embeddings_batch(texts, self.model)
        
        return {
            'documents_count': len(self.documents),
            'chunks_count': len(self.chunks),
            'has_embeddings': self.embeddings is not None
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
        """Get processing statistics"""
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'has_embeddings': self.embeddings is not None,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0
        }

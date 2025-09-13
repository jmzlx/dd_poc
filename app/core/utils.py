#!/usr/bin/env python3
"""
Utility Functions Module

Collection of utility functions used throughout the application.
This module contains helper functions for file operations, formatting,
and document processing utilities.
"""

from typing import List, Optional
from pathlib import Path


def get_mime_type(file_path: Path) -> str:
    """Get MIME type based on file extension"""
    file_extension = file_path.suffix.lower()
    if file_extension == '.pdf':
        return 'application/pdf'
    elif file_extension in ['.doc', '.docx']:
        return 'application/msword'
    elif file_extension == '.txt':
        return 'text/plain'
    elif file_extension == '.md':
        return 'text/markdown'
    else:
        return 'application/octet-stream'


def format_document_title(doc_name: str) -> str:
    """Format document name into a readable title"""
    if '.' in doc_name:
        doc_title = doc_name.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
    else:
        doc_title = doc_name.replace('_', ' ').replace('-', ' ').title()
    return doc_title


def count_documents_in_directory(directory: Path, supported_extensions: Optional[List[str]] = None) -> int:
    """Count supported documents in a directory recursively"""
    if supported_extensions is None:
        supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']

    return sum(1 for f in directory.rglob('*')
               if f.is_file() and f.suffix.lower() in supported_extensions)


def create_document_processor(store_name: Optional[str] = None) -> 'DocumentProcessor':
    """
    Create and initialize a DocumentProcessor.

    This utility function encapsulates the common pattern of creating a DocumentProcessor
    instance.

    Args:
        store_name: Optional name for the FAISS store (uses config default if None)

    Returns:
        Initialized DocumentProcessor instance
    """
    from app.core.document_processor import DocumentProcessor

    # Create document processor instance
    processor = DocumentProcessor(store_name=store_name)

    return processor

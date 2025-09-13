#!/usr/bin/env python3
"""
Unified Content Ingestion System

This module provides a unified processing pipeline with simple ingestion functions.
All content types (VDR documents, markdown files, etc.) go through the same processing pipeline
with different ingestion functions handling the content-specific parsing.
"""

# Standard library imports
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable

# Third-party imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

# Local imports
from app.core.config import get_config
from app.core.model_cache import get_cached_embeddings
from app.core.parsers import parse_checklist, parse_questions

logger = logging.getLogger(__name__)


def vdr_ingest(vdr_path: Path, store_name: str, llm=None) -> Tuple[List[Document], Dict[str, Any]]:
    """Ingest VDR documents using DocumentProcessor"""
    logger.info(f"Ingesting VDR documents from {vdr_path}")

    # Count total files for progress tracking
    total_files = sum(1 for f in vdr_path.rglob('*')
                     if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md'])

    # Initialize document processor
    from app.core.utils import create_document_processor
    processor = create_document_processor(store_name=store_name)

    # Process the data room with file-level progress
    with tqdm(total=total_files, desc=f"Files in {store_name}",
              unit="files", leave=False) as file_pbar:

        result = processor.load_data_room(str(vdr_path))

        # Update progress bar based on actual files processed
        if file_pbar and result.get('documents_count', 0) > 0:
            file_pbar.update(result['documents_count'])

    metadata = {
        'content_type': 'vdr',
        'source_path': str(vdr_path),
        'total_files': total_files,
        **result
    }

    return processor.documents, metadata


def classify_vdr_documents(documents: List[Document], store_name: str, classifier=None) -> Dict[str, str]:
    """Classify VDR documents using fast Haiku classifier"""
    if not classifier or not documents:
        return {}

    logger.info(f"🏷️ Classifying document types for {store_name}")

    # Extract only first chunks for classification efficiency
    first_chunks = []
    for doc in documents:
        if doc.metadata.get('is_first_chunk', False):
            first_chunks.append({
                'name': doc.metadata.get('name', ''),
                'path': doc.metadata.get('path', ''),
                'content': doc.page_content[:800]
            })

    if not first_chunks:
        logger.warning(f"⚠️ No first chunks found for classification in {store_name}")
        return {}

    try:
        from app.ai.document_classifier import batch_classify_document_types
        classified_docs = batch_classify_document_types(first_chunks, classifier)

        # Build classifications dictionary
        classifications = {}
        for doc in classified_docs:
            if 'document_type' in doc and doc['path']:
                classifications[doc['path']] = doc['document_type']

        logger.info(f"✅ Classified {len(classifications)} document types for {store_name}")
        return classifications

    except Exception as e:
        logger.error(f"⚠️ Failed to classify document types for {store_name}: {e}")
        return {}


def process_content(content_source: Any, content_type: str, store_name: str, classifier=None, llm=None) -> Dict[str, Any]:
    """Process content source into FAISS index"""
    start_time = time.time()

    try:
        # Get ingestion function
        ingest_func = get_ingestion_function(content_type)
        documents, ingestion_metadata = ingest_func(content_source, store_name, llm)

        if not documents:
            return {
                'success': False,
                'store_name': store_name,
                'error': 'No documents extracted'
            }

        # Classify VDR documents if classifier provided
        classifications = {}
        if classifier and content_type == 'vdr':
            classifications = classify_vdr_documents(documents, store_name, classifier)

        # Create FAISS index
        from app.core.model_cache import get_cached_embeddings
        from app.core.config import get_config
        config = get_config()
        embeddings = get_cached_embeddings(config.model['sentence_transformer_model'])
        vector_store = FAISS.from_documents(documents, embeddings)

        # Save index
        faiss_dir = config.paths['faiss_dir']
        faiss_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(faiss_dir), index_name=store_name)

        # Save classifications if available
        if classifications:
            classifications_file = faiss_dir / f"{store_name}_document_types.json"
            classifications_file.write_text(
                json.dumps(classifications, indent=2, ensure_ascii=False)
            )

        # Save enhanced checklists
        if 'enhanced_checklists' in ingestion_metadata:
            checklists_file = faiss_dir / "checklists.json"
            checklists_file.write_text(
                json.dumps(ingestion_metadata['enhanced_checklists'], indent=2, ensure_ascii=False)
            )

        processing_time = time.time() - start_time

        return {
            'success': True,
            'store_name': store_name,
            'processing_time': processing_time,
            'classifications_count': len(classifications),
            **ingestion_metadata
        }

    except Exception as e:
        return {
            'success': False,
            'store_name': store_name,
            'error': str(e),
            'processing_time': time.time() - start_time
        }


def checklist_ingest(content_dir: Path, store_name: str, llm=None) -> Tuple[List[Document], Dict[str, Any]]:
    """Ingest checklist markdown files"""
    logger.info(f"Ingesting checklist files from {content_dir}")

    if not content_dir.exists():
        raise FileNotFoundError(f"Checklist directory not found: {content_dir}")

    # Find all markdown files
    md_files = list(content_dir.glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown files found in {content_dir}")

    all_documents = []

    with tqdm(md_files, desc="Processing checklist files",
              unit="file", leave=False) as file_pbar:

        for md_file in file_pbar:
            file_pbar.set_description(f"Processing {md_file.name}")
            logger.info(f"Processing: {md_file.name}")

            content = md_file.read_text(encoding='utf-8')
            parsed_data = parse_checklist(content, llm)

            # Convert checklist items to documents
            for cat_key, category in parsed_data.items():
                for item in category.get('items', []):
                    doc = Document(
                        page_content=item['text'],
                        metadata={
                            'source': md_file.name,
                            'category': category['name'],
                            'type': 'checklist_item'
                        }
                    )
                    all_documents.append(doc)

    metadata = {
        'content_type': 'checklist',
        'source_path': str(content_dir),
        'md_files_count': len(md_files),
        'documents_count': len(all_documents)
    }

    return all_documents, metadata


def questions_ingest(content_dir: Path, store_name: str, llm=None) -> Tuple[List[Document], Dict[str, Any]]:
    """Ingest questions markdown files"""
    logger.info(f"Ingesting questions files from {content_dir}")

    if not content_dir.exists():
        raise FileNotFoundError(f"Questions directory not found: {content_dir}")

    # Find all markdown files
    md_files = list(content_dir.glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown files found in {content_dir}")

    all_documents = []

    with tqdm(md_files, desc="Processing questions files",
              unit="file", leave=False) as file_pbar:

        for md_file in file_pbar:
            file_pbar.set_description(f"Processing {md_file.name}")
            logger.info(f"Processing: {md_file.name}")

            content = md_file.read_text(encoding='utf-8')
            parsed_data = parse_questions(content, llm)

            # Convert questions to documents
            for question in parsed_data:
                doc = Document(
                    page_content=f"{question['category']}: {question['question']}",
                    metadata={
                        'source': md_file.name,
                        'category': question['category'],
                        'question_id': question['id'],
                        'type': 'question'
                    }
                )
                all_documents.append(doc)

    metadata = {
        'content_type': 'questions',
        'source_path': str(content_dir),
        'md_files_count': len(md_files),
        'documents_count': len(all_documents)
    }

    return all_documents, metadata


# Factory function for getting ingestion functions
def get_ingestion_function(content_type: str) -> Callable[..., Tuple[List[Document], Dict[str, Any]]]:
    """Factory function to get appropriate ingestion function"""
    functions = {
        'vdr': vdr_ingest,
        'checklist': checklist_ingest,
        'questions': questions_ingest
    }

    if content_type not in functions:
        raise ValueError(f"Unknown content type: {content_type}. Available: {list(functions.keys())}")

    return functions[content_type]


# Backward compatibility - create UnifiedContentProcessor class that uses process_content
class UnifiedContentProcessor:
    """Backward compatibility wrapper for process_content function"""

    def process_content_source(self, content_source: Any, content_type: str, store_name: str, classifier=None, progress_bar=None, llm=None):
        """Process content using the unified function"""
        return process_content(content_source, content_type, store_name, classifier, llm)

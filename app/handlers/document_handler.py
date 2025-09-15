#!/usr/bin/env python3
"""
Document Handler

Handles document processing operations and coordinates with the document processor.
"""

from pathlib import Path
from typing import Dict, List, Any

from app.ui.session_manager import SessionManager
from app.core.exceptions import ProcessingError
from app.ui.error_handler import ErrorHandler, handle_processing_errors
from app.core.exceptions import DocumentProcessingError, FileOperationError, create_processing_error
from app.core.logging import logger


class DocumentHandler:
    """
    Document handler that manages document processing operations.
    """

    def __init__(self, session: SessionManager):
        """Initialize handler with session manager"""
        self.session = session

    @handle_processing_errors("Data room processing", "Please check that the data room exists and contains documents")
    def process_data_room_fast(self, data_room_path: str):
        """
        Fast data room processing using pre-built FAISS indices.

        Args:
            data_room_path: Path to the data room directory

        Returns:
            Tuple of (documents_count, chunks_count) or None on error
        """
        # Extract company name from path
        company_name = Path(data_room_path).name.lower()

        # Initialize document processor with loaded FAISS store
        from app.core.utils import create_document_processor
        document_processor = create_document_processor(store_name=company_name)

        if not document_processor.vector_store:
            raise create_processing_error(
                f"No pre-built FAISS index found for '{company_name}'",
                recovery_hint="Please run scripts/build_indexes.py first to create the index"
            )

        # Quick document metadata scan
        documents_dict = self._quick_document_scan(data_room_path)

        # Get chunks from FAISS metadata
        chunks = self._extract_chunks_from_faiss(document_processor)

        # Store in session
        self.session.documents = documents_dict
        self.session.chunks = chunks
        self.session.embeddings = document_processor.embeddings
        self.session.vdr_store = company_name

        # Preload checklist embeddings into memory for fast search
        from app.core.search import preload_checklist_embeddings
        logger.info("Attempting to preload checklist embeddings...")
        try:
            preloaded_count = preload_checklist_embeddings()
            logger.info(f"✅ Successfully preloaded {preloaded_count} checklist embeddings for fast searching")
        except RuntimeError as e:
            logger.error(f"❌ Failed to preload checklist embeddings: {e}")
            logger.error("This will cause checklist matching to fail - embeddings must be available for search")
            # Don't fail the entire data room processing, but make it very clear this is a problem
            raise  # Re-raise to make this a hard failure

        # Load pre-built document type embeddings from disk
        from app.core.search import preload_document_type_embeddings
        logger.info(f"Loading pre-built document type embeddings for {company_name}...")
        try:
            type_embeddings = preload_document_type_embeddings(company_name)
            # Store in session for use during search
            self.session.document_type_embeddings = type_embeddings
            logger.info(f"✅ Loaded {len(type_embeddings)} pre-built document type embeddings")
            logger.info(f"Session ID: {id(self.session)}, Embeddings stored: {bool(self.session.document_type_embeddings)}")
        except RuntimeError as e:
            logger.error(f"❌ Failed to load pre-built document type embeddings: {e}")
            logger.error("This indicates the build process did not complete successfully.")
            logger.error("Please run 'uv run build-indexes' to generate required embeddings.")
            raise  # Fail fast - embeddings are required for checklist processing

        # Clear existing analysis
        self.session.reset()

        logger.info(f"Successfully processed {len(documents_dict)} documents and {len(chunks)} chunks")
        return len(documents_dict), len(chunks)

    def _quick_document_scan(self, data_room_path: str) -> Dict[str, Any]:
        """Quick scan of document files without loading content"""
        documents_dict = {}
        data_room_path_obj = Path(data_room_path)

        # Validate data room path exists
        if not data_room_path_obj.exists():
            raise create_processing_error(
                f"Data room path does not exist: {data_room_path}",
                recovery_hint="Please select a valid data room directory"
            )

        # Quick file system scan for supported extensions
        from app.core import get_config
        config = get_config()
        supported_extensions = config.get_supported_extensions()

        for ext in supported_extensions:
            for file_path in data_room_path_obj.rglob(f"*{ext}"):
                if file_path.is_file():
                    try:
                        rel_path = file_path.relative_to(data_room_path_obj)
                        documents_dict[str(file_path)] = {
                            'name': file_path.name,
                            'path': str(rel_path),
                            'content': f"[Indexed - {file_path.stat().st_size:,} bytes]",
                            'metadata': {
                                'source': str(file_path),
                                'name': file_path.name,
                                'path': str(rel_path)
                            }
                        }
                    except ValueError:
                        # Skip files outside data room path
                        continue

        if not documents_dict:
            raise create_processing_error(
                f"No supported documents found in {data_room_path}",
                recovery_hint="Please ensure the data room contains PDF, DOCX, or text files"
            )

        return documents_dict

    def _extract_chunks_from_faiss(self, document_processor) -> List[Dict]:
        """Extract chunk information from loaded FAISS store"""
        chunks = []

        if not document_processor.vector_store:
            logger.warning("No vector store available for chunk extraction")
            return chunks

        try:
            # Access the docstore to get document metadata
            docstore = document_processor.vector_store.docstore

            for doc_id in docstore._dict.keys():
                doc = docstore._dict[doc_id]
                chunk_text = doc.page_content
                if len(chunk_text) > 500:
                    chunk_text = chunk_text[:500] + "..."

                chunk_dict = {
                    'text': chunk_text,
                    'source': doc.metadata.get('name', ''),
                    'path': doc.metadata.get('path', ''),
                    'full_path': doc.metadata.get('source', ''),
                    'metadata': doc.metadata
                }
                chunks.append(chunk_dict)

        except (DocumentProcessingError, FileOperationError) as e:
            ErrorHandler.handle_error(
                e,
                "Failed to extract chunks from FAISS store",
                recovery_hint="The FAISS index may be corrupted"
            )
            # Fallback: create minimal chunks
            chunks = [{
                'text': '[Content available in search]',
                'source': 'indexed_content',
                'path': '',
                'full_path': '',
                'metadata': {}
            }]

        return chunks

    def get_document_processor(self, store_name: str = None):
        """
        Get a configured document processor.

        Args:
            store_name: Optional store name for the processor

        Returns:
            Configured DocumentProcessor instance
        """
        from app.core.utils import create_document_processor
        return create_document_processor(store_name=store_name)

    def validate_data_room(self, data_room_path: str) -> bool:
        """
        Validate that a data room path exists and contains documents.

        Args:
            data_room_path: Path to validate

        Returns:
            True if valid, False otherwise
        """
        path_obj = Path(data_room_path)
        if not path_obj.exists():
            return False

        return self._has_supported_files(path_obj)

    def _has_supported_files(self, path_obj: Path) -> bool:
        """
        Check if path contains files with supported extensions.

        Args:
            path_obj: Path object to check

        Returns:
            True if supported files are found
        """
        from app.core import get_config
        config = get_config()
        supported_extensions = config.get_supported_extensions()

        for ext in supported_extensions:
            if list(path_obj.rglob(f"*{ext}")):
                return True

        return False

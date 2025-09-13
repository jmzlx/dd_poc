#!/usr/bin/env python3
"""
Document Classification Module

This module contains functions for classifying document types and related utilities.
"""

# Standard library imports
import logging
from typing import List, Dict, Optional

# Third-party imports
from langchain_core.messages import HumanMessage
import httpx
import backoff

# Local imports
from app.ai.agent_utils import create_batch_processor
from app.ai.prompts import get_document_type_classification_prompt
from app.core.config import get_config
from app.core.constants import DEFAULT_BATCH_SIZE
from app.core.performance import get_performance_manager

logger = logging.getLogger(__name__)


@backoff.on_exception(
    backoff.expo,
    (Exception,),
    max_tries=3,
    jitter=backoff.random_jitter
)
def batch_classify_document_types(first_chunks: List[Dict], llm: "ChatAnthropic", batch_size: Optional[int] = None) -> List[Dict]:
    """
    Fast document type classification using first chunks only with Haiku model.
    Optimized for speed and cost with batched processing.

    Args:
        first_chunks: List of first chunk dictionaries to classify
        llm: ChatAnthropic instance (should be Haiku for speed/cost)
        batch_size: Number of documents to process in each batch (uses config default if None)

    Returns:
        List of documents with added document_type field
    """
    config = get_config()
    if batch_size is None:
        # Use optimized batch size for Haiku (faster model)
        batch_size = min(DEFAULT_BATCH_SIZE, 25)  # Increased to 25 docs per batch for better performance

    # Create batch processor with retry and fallback mechanisms
    batch_processor = create_batch_processor(llm, max_concurrency=5)  # Increased concurrency

    # Process documents in batches
    classified_docs = []
    total_docs = len(first_chunks)
    total_batches = (total_docs + batch_size - 1) // batch_size

    model_name = getattr(llm, 'model', 'unknown')
    logger.info(f"🏷️ Classifying {total_docs} document types using {model_name}")

    # Get performance manager for caching
    perf_manager = get_performance_manager()

    for batch_num, i in enumerate(range(0, total_docs, batch_size), 1):
        batch = first_chunks[i:i + batch_size]
        batch_end = min(i + batch_size, total_docs)

        # Check cache for existing classifications
        cached_batch = []
        uncached_batch = []
        uncached_indices = []

        for idx, doc in enumerate(batch):
            cache_key = f"classification:{doc.get('path', '')}"
            cached_result = perf_manager.doc_cache.get(cache_key)
            if cached_result:
                cached_batch.append(cached_result)
                logger.debug(f"Cache hit for document classification: {doc.get('name', '')}")
            else:
                uncached_batch.append(doc)
                uncached_indices.append(idx)

        logger.info(f"Processing classification batch {batch_num}/{total_batches} "
                   f"({len(uncached_batch)} new, {len(cached_batch)} cached documents)")

        # Only process uncached documents
        if uncached_batch:
            batch_inputs = []
            for doc in uncached_batch:
                template = get_document_type_classification_prompt()
                prompt = template.format(
                    doc_name=doc.get('name', 'Unknown'),
                    content_preview=doc.get('content', '')[:500]  # First 500 chars for classification
                )
                messages = [HumanMessage(content=prompt)]
                batch_inputs.append((messages, doc))

            # Process batch using LangChain's built-in mechanisms
            try:
                logger.info(f"Processing classification batch {batch_num}/{total_batches} with {len(uncached_batch)} new documents")
                batch_results = batch_processor.invoke(batch_inputs)

                # Process results with individual document error handling
                for idx, result in enumerate(batch_results):
                    doc = result['item_info'].copy()

                    if result['success'] and result['response']:
                        # Successfully classified document type
                        doc_type = result['response'].content.strip().lower()
                        # Remove any "the document type is" prefix if present (for backward compatibility)
                        if doc_type.startswith("the document type is "):
                            doc_type = doc_type[21:].strip()
                        doc['document_type'] = doc_type
                        logger.debug(f"Classified '{doc.get('name', 'Unknown')}' as: {doc_type}")

                        # Cache the result
                        cache_key = f"classification:{doc.get('path', '')}"
                        perf_manager.doc_cache.set(cache_key, doc, expire=86400 * 30)  # 30 days

                        classified_docs.append(doc)
                    else:
                        # Fail on classification error
                        error_msg = f"Failed to classify document '{doc.get('name', 'Unknown')}': {result.get('error', 'Unknown error')}"
                        logger.error(error_msg)
                        raise Exception(error_msg)

            except Exception as e:
                error_msg = f"Classification batch {batch_num} processing completely failed: {e}"
                logger.error(error_msg)
                raise Exception(error_msg)

        # Add cached results to the final list
        classified_docs.extend(cached_batch)

    successful_classifications = len([d for d in classified_docs if d.get('document_type') != 'unknown document'])
    success_rate = (successful_classifications / total_docs) * 100 if total_docs > 0 else 0
    logger.info(f"✅ Classified {successful_classifications}/{total_docs} documents ({success_rate:.1f}% success rate)")

    return classified_docs

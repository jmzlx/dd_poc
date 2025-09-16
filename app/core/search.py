#!/usr/bin/env python3
"""
Search and analysis functions for document retrieval and ranking.
"""

# Standard library imports
from typing import Dict, List, Tuple
from pathlib import Path

# Third-party imports for Unicode normalization
import unidecode

# Third-party imports
import numpy as np
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Local imports
from app.core.constants import (
    SIMILARITY_THRESHOLD, STATISTICAL_CANDIDATE_POOL_SIZE, 
    STATISTICAL_STD_MULTIPLIER, STATISTICAL_MIN_CANDIDATES,
    STATISTICAL_MIN_STD_DEV
)
from app.core.document_processor import DocumentProcessor
from app.core.logging import logger
from app.core.ranking import rerank_results
from app.core.sparse_index import load_sparse_index_for_store, BM25Index


def filter_statistically_significant_matches(matches: List[Dict], std_multiplier: float = STATISTICAL_STD_MULTIPLIER) -> Tuple[List[Dict], Dict]:
    """
    Filter matches using statistical significance instead of fixed thresholds.
    
    This approach analyzes the score distribution to identify documents that are 
    statistically significantly more relevant than the average, eliminating the
    need for arbitrary fixed thresholds.
    
    Optimized with vectorized numpy operations for better performance.
    
    Args:
        matches: List of match dictionaries with 'score' keys
        std_multiplier: Number of standard deviations above mean to use as threshold
                       (1.0=loose, 1.5=moderate, 2.0=strict)
    
    Returns:
        Tuple of (filtered_matches, statistics_dict)
    """
    if len(matches) < STATISTICAL_MIN_CANDIDATES:  # Need minimum samples for meaningful statistics
        return matches, {
            'method': 'insufficient_data',
            'total_candidates': len(matches),
            'significant_matches': len(matches),
            'note': 'Less than 5 candidates - returning all'
        }
    
    import numpy as np
    
    # VECTORIZED: Extract scores using numpy array operations instead of list comprehension
    scores_array = np.array([m['score'] for m in matches])
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    
    # Use minimum standard deviation when variance is too low to prevent tiny thresholds
    effective_std = max(std_score, STATISTICAL_MIN_STD_DEV)
    
    # Calculate adaptive threshold using effective standard deviation
    adaptive_threshold = mean_score + (std_multiplier * effective_std)
    
    # VECTORIZED: Use numpy boolean masking for efficient filtering
    significance_mask = scores_array >= adaptive_threshold
    significant_indices = np.where(significance_mask)[0]
    significant_matches = [matches[i] for i in significant_indices]
    
    # Return only statistically significant matches (no fallback)
    method = 'statistical_filtering' if significant_matches else 'no_significant_matches'
    
    # Generate statistics metadata - using vectorized operations
    score_min, score_max = np.min(scores_array), np.max(scores_array)
    
    stats = {
        'method': method,
        'mean': round(float(mean_score), 3),
        'std': round(float(std_score), 3),
        'effective_std': round(float(effective_std), 3),
        'adaptive_threshold': round(float(adaptive_threshold), 3),
        'std_multiplier': std_multiplier,
        'min_std_applied': effective_std > std_score,
        'total_candidates': len(matches),
        'significant_matches': len(significant_matches),
        'score_range': [round(float(score_min), 3), round(float(score_max), 3)]
    }
    
    return significant_matches, stats


def search_and_analyze(queries: List[Dict], vector_store: FAISS, llm=None, threshold: float = SIMILARITY_THRESHOLD, search_type: str = 'items', store_name: str = None, session=None) -> Dict:
    """Unified search function for both checklist items and questions using direct FAISS search for accurate scores"""

    # Create RAG chain if LLM is provided
    qa_chain = None
    if llm:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": threshold, "k": 5 if search_type == 'questions' else 10}
        )

        prompt_template = PromptTemplate(
            input_variables=["context", "input"],
            template="""Use the provided context to answer the question. Be concise and factual.

Context: {context}

Question: {input}

Answer:"""
        )
        # Create the document chain and then the retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        qa_chain = create_retrieval_chain(retriever, document_chain)

    if search_type == 'items':
        return _process_checklist_items(queries, vector_store, threshold, store_name, session)
    else:
        return _process_questions(queries, vector_store, threshold, qa_chain, llm)


def _process_checklist_items(checklist: Dict, vector_store: FAISS, threshold: float, store_name: str = None, session=None) -> Dict:
    """Compare checklist items directly against LLM-generated document type classifications using optimized batch processing"""
    import numpy as np

    # Ensure checklist embeddings are preloaded first
    if not hasattr(get_checklist_embedding, '_cache') or not get_checklist_embedding._cache:
        logger.error("CRITICAL: Checklist embeddings cache is empty during processing - this should have been preloaded!")
        logger.info("Attempting emergency preload of checklist embeddings...")
        try:
            from app.core.search import preload_checklist_embeddings
            count = preload_checklist_embeddings()
            logger.info(f"✅ Emergency preloaded {count} checklist embeddings for processing")
        except Exception as e:
            logger.error(f"❌ Failed to emergency preload checklist embeddings: {e}")
            logger.error("This indicates embeddings were not properly generated or saved during build process")
            raise RuntimeError(f"Checklist embeddings are required but not available: {e}")

    # Ensure document type embeddings are available
    if session:
        logger.debug(f"Checklist processing session ID: {id(session)}, has embeddings: {hasattr(session, 'document_type_embeddings')}")
        if hasattr(session, 'document_type_embeddings'):
            logger.debug(f"Embeddings count: {len(session.document_type_embeddings) if session.document_type_embeddings else 0}")

    # Check that document type embeddings are available in session
    embeddings_missing = not session or not hasattr(session, 'document_type_embeddings') or not session.document_type_embeddings

    if embeddings_missing:
        logger.error("Document type embeddings not available in session. Checklist processing requires pre-built embeddings.")
        logger.error("Make sure data room processing completed successfully during application startup.")
        logger.error("If embeddings are missing, run 'uv run build-indexes' to regenerate them.")
        return {}

    # OPTIMIZATION 1: Load document type classifications ONCE at the start
    doc_types = {}
    if store_name:
        doc_types = _load_document_types(vector_store, store_name)

    if not doc_types:
        logger.error(f"No document type classifications found for {store_name}")
        raise ValueError(f"No document type classifications available for {store_name}. This indicates the data room processing did not complete successfully or build indexes were not run.")

    # OPTIMIZATION 4: Pre-build matrices for batch similarity calculations
    logger.info(f"🚀 Preparing batch similarity computation for {len(doc_types)} documents...")
    
    # Filter out unclassified documents and prepare data structures
    valid_docs = []
    doc_embeddings = []
    
    for doc_path, doc_type in doc_types.items():
        if not doc_type or doc_type == 'not classified':
            continue
            
        doc_type_lower = doc_type.lower().strip()
        
        try:
            # Get document type embedding (from preloaded cache)
            doc_type_embedding = get_document_type_embedding(doc_type_lower, session)
            
            valid_docs.append({
                'path': doc_path,
                'type': doc_type,
                'name': _extract_doc_name_from_path(doc_path)
            })
            doc_embeddings.append(doc_type_embedding)
        except Exception as e:
            logger.error(f"Error loading embedding for {doc_path}: {e}")
            raise ValueError(f"Failed to load document type embedding for {doc_path}: {e}. This indicates missing or corrupted embeddings data.")
    
    if not valid_docs:
        logger.error("No valid documents with embeddings found")
        raise ValueError("No valid documents with embeddings available for checklist matching. This indicates document type embeddings were not properly generated during build process.")
        
    # Convert to numpy matrix for vectorized operations
    doc_embeddings_matrix = np.vstack(doc_embeddings)
    doc_norms = np.linalg.norm(doc_embeddings_matrix, axis=1)
    logger.info(f"✅ Built embedding matrix: {doc_embeddings_matrix.shape} for {len(valid_docs)} documents")

    results = {}
    for cat_letter, category in checklist.items():
        cat_results = {
            'name': category['name'],
            'items': [],
            'total_items': len(category['items']),
            'matched_items': 0
        }

        for item in category['items']:
            checklist_item_text = item['text'].lower().strip()
            
            
            try:
                # Get checklist embedding from memory cache
                checklist_embedding = get_checklist_embedding(checklist_item_text)
                checklist_norm = np.linalg.norm(checklist_embedding)
                
                # BATCH SIMILARITY CALCULATION: Compute all similarities at once using matrix operations
                # This replaces the O(n) inner loop with O(1) vectorized computation
                dot_products = np.dot(doc_embeddings_matrix, checklist_embedding)
                similarities = dot_products / (doc_norms * checklist_norm)
                
                # Build candidate matches from batch results
                candidate_matches = []
                for idx, similarity in enumerate(similarities):
                    doc_info = valid_docs[idx]
                    candidate_matches.append({
                        'name': doc_info['name'],
                        'path': doc_info['path'],
                        'full_path': doc_info['path'],  # For consistency
                        'score': round(float(similarity), 3),
                        'document_type': doc_info['type'],
                        'text': f"Document type: {doc_info['type']}"
                    })
                
                # Sort all candidates by score (highest first)
                candidate_matches.sort(key=lambda x: x['score'], reverse=True)

                # Take top candidates for statistical analysis 
                top_candidates = candidate_matches[:STATISTICAL_CANDIDATE_POOL_SIZE]

                # Apply statistical filtering instead of fixed threshold
                if top_candidates:
                    # Use configurable standard deviation multiplier
                    matches, stats = filter_statistically_significant_matches(top_candidates, STATISTICAL_STD_MULTIPLIER)
                    
                    # Reduced logging - only log summary info to improve performance
                    logger.debug(f"📊 '{checklist_item_text[:30]}...' -> {stats['significant_matches']} matches via {stats['method']}")
                    
                    # Only count as matched if there are actual matches after filtering
                    if matches:
                        cat_results['matched_items'] += 1
                else:
                    matches = []
                    stats = {'method': 'no_candidates', 'significant_matches': 0}

                cat_results['items'].append({
                    'text': item['text'],
                    'original': item['original'],
                    'matches': matches,
                    'statistics': stats  # Include statistical metadata for debugging/analysis
                })
                
            except Exception as e:
                logger.error(f"Failed to process checklist item '{checklist_item_text[:50]}...': {e}")
                raise ValueError(f"Checklist item processing failed: {e}. This indicates a critical failure in embedding comparison for item: '{checklist_item_text[:100]}...'")

        results[cat_letter] = cat_results

    logger.info(f"✅ Completed optimized batch processing for {len(checklist)} categories")
    return results


def _load_document_types(vector_store, store_name: str):
    """Load document type classifications for the given store"""
    try:
        from pathlib import Path
        from app.core.config import get_app_config
        config = get_app_config()
        doc_types_path = config.paths['faiss_dir'] / f"{store_name}_document_types.json"
        if doc_types_path.exists():
            import json
            with open(doc_types_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load document types for {store_name}: {e}")
        raise ValueError(f"Document type loading failed for {store_name}: {e}. This indicates the build process did not complete successfully.")


def _extract_doc_name_from_path(doc_path: str) -> str:
    """Extract document name from file path"""
    try:
        path_obj = Path(doc_path)
        return path_obj.name
    except Exception:
        # Fallback: extract name from path string
        return doc_path.split('/')[-1] if '/' in doc_path else doc_path.split('\\')[-1] if '\\' in doc_path else doc_path


def get_checklist_embedding(checklist_text: str):
    """
    Get cached embedding for checklist item from in-memory cache.

    This function only uses in-memory cache that should be preloaded during
    data room processing. It will fail if the embedding is not available.

    Args:
        checklist_text: The checklist item text to look up

    Returns:
        numpy array: The embedding vector for the checklist text

    Raises:
        RuntimeError: If embedding is not found in cache
    """
    # Initialize cache if not exists
    if not hasattr(get_checklist_embedding, '_cache'):
        get_checklist_embedding._cache = {}
        logger.error("❌ Checklist embedding cache was not initialized!")
        logger.error("This indicates embeddings were not preloaded during document processing.")
        logger.error("Make sure preload_checklist_embeddings() is called before any similarity calculations.")

    # Create cache key from checklist text with normalized Unicode
    cache_key = checklist_text.lower().strip()
    # Use unidecode for comprehensive Unicode to ASCII conversion
    cache_key = unidecode.unidecode(cache_key)
    # Additional normalization for common Unicode issues
    cache_key = cache_key.replace('–', '-').replace('—', '-')  # Normalize dashes
    cache_key = cache_key.replace(''', "'").replace(''', "'")  # Normalize quotes


    # Check in-memory cache only
    if cache_key in get_checklist_embedding._cache:
        return get_checklist_embedding._cache[cache_key]

    # No fallbacks - fail explicitly if embedding not found
    cache_size = len(get_checklist_embedding._cache)
    
    # Minimal debugging info
    logger.error(f"❌ Checklist embedding not found for: '{checklist_text[:50]}...'")
    logger.error(f"❌ Cache key: '{cache_key}'")
    logger.error(f"❌ Cache size: {cache_size}")
    
    if cache_size == 0:
        logger.error("❌ Cache is empty - embeddings were not preloaded!")
        logger.error("❌ Run data room processing first to generate embeddings")
    else:
        logger.error("❌ This indicates a mismatch between build-time and runtime parsing")
        logger.error("❌ The system should use pre-parsed structures, not runtime re-parsing")

    # Fail fast - no fallbacks, no workarounds
    raise RuntimeError(
        f"Checklist embedding not found. This should not happen with pre-parsed structures. "
        f"Cache key: '{cache_key}', Cache size: {cache_size}. "
        f"Rebuild search indexes or check checklist_structures.json exists."
    )


def get_document_type_embedding(doc_type: str, session=None):
    """
    Get cached embedding for document type from session cache.

    Args:
        doc_type: The document type text to get embedding for
        session: The session object containing preloaded embeddings

    Returns:
        numpy.ndarray: The embedding vector

    Raises:
        RuntimeError: If embedding is not found in cache
    """
    if not session or not hasattr(session, 'document_type_embeddings') or not session.document_type_embeddings:
        raise RuntimeError(f"Document type embedding not found for: '{doc_type[:50]}...'. Preloaded embeddings required.")

    # Create cache key with normalized Unicode
    cache_key = unidecode.unidecode(doc_type.lower().strip())

    # Get from session cache only
    if cache_key in session.document_type_embeddings:
        return session.document_type_embeddings[cache_key]

    raise RuntimeError(f"Document type embedding not found for: '{doc_type[:50]}...' (cache key: '{cache_key}')")


def generate_checklist_embeddings():
    """
    Generate embeddings for all checklist items and save to disk.

    This function should be called during the build process to pre-calculate
    embeddings for all checklist items from the available checklist files.
    Uses LLM parsing to ensure consistency with runtime parsing.

    Returns:
        int: Number of embeddings generated and saved
    """
    try:
        from app.core.config import get_config
        from app.core.model_cache import get_cached_embeddings
        from app.core.parsers import parse_checklist
        import json
        import numpy as np
        import unidecode

        config = get_config()
        embeddings_model = get_cached_embeddings()
        checklist_dir = config.paths['checklist_dir']

        logger.info("🔄 Generating checklist embeddings using LLM parsing...")

        # Get LLM instance for parsing - use same config as runtime for consistency
        try:
            from langchain_anthropic import ChatAnthropic
            import os
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
            
            # Use exact same configuration as runtime to ensure consistent parsing    
            model = os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514')
            temperature = float(os.getenv('CLAUDE_TEMPERATURE', '0.0'))
            max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', '16000'))
                
            llm = ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"Using LLM config: model={model}, temperature={temperature}, max_tokens={max_tokens}")
        except Exception as e:
            raise RuntimeError(f"Failed to create LLM instance: {e}")

        # Initialize embeddings cache and parsed structures storage
        embeddings_cache = {}
        all_parsed_checklists = {}

        # Process all checklist files
        checklist_files = list(checklist_dir.glob("*.md"))
        if not checklist_files:
            logger.warning(f"No checklist files found in {checklist_dir}")
            return 0

        for checklist_file in checklist_files:
            logger.info(f"Processing checklist: {checklist_file.name}")

            try:
                # Read checklist content
                content = checklist_file.read_text(encoding='utf-8')

                # Parse checklist using improved LLM parsing
                parsed_checklist = parse_checklist(content, llm)
                
                # Store parsed structure for runtime use
                all_parsed_checklists[checklist_file.name] = parsed_checklist
                
                # OPTIMIZATION: Collect all texts for batch embedding generation
                texts_to_embed = []
                text_to_cache_key = {}
                
                for category_key, category in parsed_checklist.items():
                    for item in category.get('items', []):
                        item_text = item['text']
                        
                        # Process cache key
                        cache_key = item_text.lower().strip()
                        cache_key = unidecode.unidecode(cache_key)
                        cache_key = cache_key.replace('–', '-').replace('—', '-')
                        cache_key = cache_key.replace(''', "'").replace(''', "'")

                        # Skip if already cached
                        if cache_key in embeddings_cache:
                            continue
                            
                        # Add to batch processing lists
                        texts_to_embed.append(item_text)
                        text_to_cache_key[item_text] = cache_key
                
                # BATCH EMBEDDING GENERATION: Process all texts at once using matrix operations
                if texts_to_embed:
                    logger.info(f"🚀 Batch generating embeddings for {len(texts_to_embed)} checklist items...")
                    from app.core.performance import optimize_embedding_batch
                    batch_embeddings = optimize_embedding_batch(texts_to_embed, embeddings_model)
                    
                    # Store batch results in cache
                    for item_text, embedding in zip(texts_to_embed, batch_embeddings):
                        cache_key = text_to_cache_key[item_text]
                        if hasattr(embedding, 'tolist'):
                            embeddings_cache[cache_key] = embedding.tolist()
                        elif isinstance(embedding, list):
                            embeddings_cache[cache_key] = embedding
                        else:
                            embeddings_cache[cache_key] = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                        logger.debug(f"✅ Batch embedded: {item_text[:50]}...")
                        
                    logger.info(f"✅ Successfully batch generated {len(batch_embeddings)} checklist embeddings")

            except Exception as e:
                logger.error(f"Failed to process checklist file {checklist_file}: {e}")
                # Continue processing other files even if one fails
                continue

        # Save embeddings to disk
        cache_file = config.paths['faiss_dir'] / "checklist_embeddings.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_cache, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Saved {len(embeddings_cache)} checklist embeddings to {cache_file}")
        
        # Save parsed checklist structures to eliminate runtime re-parsing
        structures_file = config.paths['faiss_dir'] / "checklist_structures.json"
        with open(structures_file, 'w', encoding='utf-8') as f:
            json.dump(all_parsed_checklists, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(all_parsed_checklists)} parsed checklist structures to {structures_file}")
        
        if not embeddings_cache:
            raise RuntimeError("No checklist embeddings were successfully generated. All checklist files failed to process.")
        
        return len(embeddings_cache)

    except Exception as e:
        error_msg = f"Failed to generate checklist embeddings: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def _parse_checklist_items_from_markdown(content: str) -> list:
    """
    Parse checklist items from markdown content.

    Args:
        content: Markdown content containing checklist items

    Returns:
        list: List of checklist item texts
    """
    import re

    items = []

    # Find numbered items like "1. Item text" or "- Item text"
    # Look for patterns after category headers
    lines = content.split('\n')

    for line in lines:
        line = line.strip()

        # Skip empty lines and headers
        if not line or line.startswith('#') or line.startswith('⸻'):
            continue

        # Look for numbered items: "1. ", "2. ", etc. or bullet points
        if re.match(r'^\d+\.\s+', line) or line.startswith('- '):
            # Clean up the item text
            if line.startswith('- '):
                item_text = line[2:].strip()
            else:
                # Remove the number prefix
                item_text = re.sub(r'^\d+\.\s+', '', line).strip()

            # Skip if too short or looks like a header
            if len(item_text) > 10 and not item_text.isupper():
                items.append(item_text)

    logger.info(f"Parsed {len(items)} checklist items from markdown")
    return items


def load_prebuilt_checklist(checklist_filename: str) -> dict:
    """
    Load pre-parsed checklist structure from build artifacts.
    
    This function loads checklist structures that were parsed during build time,
    eliminating the need for runtime LLM parsing and ensuring consistency.
    
    Args:
        checklist_filename: Name of the checklist file (e.g., 'bloomberg.md')
        
    Returns:
        Dictionary containing parsed checklist structure
        
    Raises:
        RuntimeError: If structures file doesn't exist or checklist not found
    """
    from app.core.config import get_config
    import json
    
    config = get_config()
    structures_file = config.paths['faiss_dir'] / "checklist_structures.json"
    
    if not structures_file.exists():
        raise RuntimeError(
            f"Checklist structures file not found: {structures_file}. "
            f"Run build process first to generate pre-parsed structures."
        )
    
    try:
        with open(structures_file, 'r', encoding='utf-8') as f:
            all_structures = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load checklist structures: {e}")
    
    if checklist_filename not in all_structures:
        available_files = list(all_structures.keys())
        raise RuntimeError(
            f"Checklist '{checklist_filename}' not found in pre-built structures. "
            f"Available: {available_files}. Rebuild search indexes."
        )
    
    logger.info(f"✅ Loaded pre-parsed checklist structure for: {checklist_filename}")
    return all_structures[checklist_filename]


def preload_checklist_embeddings():
    """
    Preload all checklist embeddings into memory during data room processing.

    This function loads pre-calculated embeddings from disk into the in-memory cache.
    It should be called once during data room processing to prepare for fast searches.

    Returns:
        int: Number of embeddings successfully preloaded

    Raises:
        RuntimeError: If embeddings file doesn't exist or can't be loaded
    """
    try:
        from app.core.config import get_config
        import json
        import numpy as np

        config = get_config()
        cache_file = config.paths['faiss_dir'] / "checklist_embeddings.json"

        if not cache_file.exists():
            raise RuntimeError(
                f"Checklist embeddings file not found: {cache_file}. "
                f"Run build process first: python scripts/build_indexes.py"
            )

        # Initialize cache
        if not hasattr(get_checklist_embedding, '_cache'):
            get_checklist_embedding._cache = {}

        # Load all embeddings from disk
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # Convert and cache all embeddings in memory
        preloaded_count = 0
        for cache_key, embedding_list in cache_data.items():
            # Normalize Unicode in cache key to match search normalization
            normalized_key = unidecode.unidecode(cache_key)
            # Additional normalization for common Unicode issues
            normalized_key = normalized_key.replace('–', '-').replace('—', '-')  # Normalize dashes
            normalized_key = normalized_key.replace(''', "'").replace(''', "'")  # Normalize quotes
            embedding_array = np.array(embedding_list, dtype=np.float32)
            get_checklist_embedding._cache[normalized_key] = embedding_array
            preloaded_count += 1

        logger.info(f"✅ Preloaded {preloaded_count} checklist embeddings into memory")
        return preloaded_count

    except Exception as e:
        error_msg = f"Failed to preload checklist embeddings: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def preload_document_type_embeddings(store_name: str):
    """
    Load pre-built document type embeddings from disk.
    
    This function loads document type embeddings that were generated and saved
    during the build process. It will fail if embeddings are not available.

    Args:
        store_name: Name of the document store
        
    Returns:
        dict: Dictionary mapping normalized document types to their embeddings

    Raises:
        RuntimeError: If pre-built embeddings can't be loaded
    """
    try:
        from app.core.config import get_app_config
        import pickle
        
        config = get_app_config()
        embeddings_file = config.paths['faiss_dir'] / f"{store_name}_document_type_embeddings.pkl"
        
        if not embeddings_file.exists():
            raise RuntimeError(
                f"Pre-built document type embeddings not found: {embeddings_file}\n"
                f"Run 'uv run build-indexes' to generate embeddings during build process"
            )
        
        logger.info(f"📥 Loading pre-built document type embeddings from {embeddings_file.name}...")
        
        with open(embeddings_file, 'rb') as f:
            type_embeddings = pickle.load(f)
        
        if not type_embeddings:
            raise RuntimeError(f"Empty embeddings file: {embeddings_file}")
            
        logger.info(f"✅ Loaded {len(type_embeddings)} pre-built document type embeddings")
        return type_embeddings

    except Exception as e:
        error_msg = f"Failed to load pre-built document type embeddings for {store_name}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)




def _process_questions(queries: List[Dict], vector_store: FAISS, threshold: float, qa_chain=None, llm=None) -> Dict:
    """Process questions using batch processing for parallel LLM calls"""
    if not queries:
        return {'questions': []}
    
    if qa_chain and llm:
        return _process_questions_with_rag_batch(queries, vector_store, threshold, llm)
    elif qa_chain:
        raise ValueError("LLM required for RAG processing but not provided")
    else:
        return _process_questions_simple_search(queries, vector_store, threshold)


def _process_questions_with_rag_batch(queries: List[Dict], vector_store: FAISS, threshold: float, llm) -> Dict:
    """Process questions using batch processing - fail fast, no fallbacks"""
    from app.ai.agent_utils import create_batch_processor
    from langchain_core.messages import HumanMessage
    
    # Create batch processor
    batch_processor = create_batch_processor(llm, max_concurrency=5)
    
    logger.info(f"Processing {len(queries)} questions using batch processing")
    
    # Prepare all batch inputs
    batch_inputs = []
    question_contexts = []
    
    for query in queries:
        question = query['question']
        
        # Retrieve documents for this question
        docs_with_scores = vector_store.similarity_search_with_score(question, k=5)
        relevant_docs = [doc for doc, score in docs_with_scores if (1.0 - (score / 2.0) if score <= 2.0 else 0.0) >= threshold]
        
        # Create context and sources with full document content (chunks already properly sized)
        if relevant_docs:
            context = "\n\n".join([f"- {doc.metadata.get('name', 'Unknown')}: {doc.page_content}" 
                                 for doc in relevant_docs[:5]])
            # Deduplicate sources by document name, keeping highest score
            sources_dict = {}
            for doc, score in docs_with_scores[:5]:
                similarity_score = round(1.0 - (score / 2.0) if score <= 2.0 else 0.0, 3)
                if similarity_score >= threshold:
                    doc_name = doc.metadata.get('name', '')
                    doc_path = doc.metadata.get('path', '')
                    # Keep the highest score for each unique document
                    if doc_name not in sources_dict or similarity_score > sources_dict[doc_name]['score']:
                        sources_dict[doc_name] = {
                            'name': doc_name,
                            'path': doc_path,
                            'score': similarity_score
                        }
            sources = list(sources_dict.values())
        else:
            context = ""
            sources = []
        
        question_contexts.append(sources)
        
        # Create prompt
        prompt_content = f"""Use the provided context to answer the question. Be concise and factual.

Context: {context}

Question: {question}

Answer:"""
        
        messages = [HumanMessage(content=prompt_content)]
        batch_inputs.append((messages, query))
    
    # Process batch - fail if anything goes wrong
    batch_results = batch_processor.invoke(batch_inputs)
    
    # Build results
    results = []
    for idx, result in enumerate(batch_results):
        if not result['success'] or not result['response']:
            raise RuntimeError(f"Failed to process question: {result['item_info']['question']}")
        
        query = result['item_info']
        answer = result['response'].content.strip()
        sources = question_contexts[idx]
        
        results.append({
            'question': query['question'],
            'category': query.get('category', ''),
            'answer': answer,
            'sources': sources,
            'method': 'rag_batch',
            'has_answer': bool(answer and answer.strip())
        })
    
    return {'questions': results}




def _process_questions_simple_search(queries: List[Dict], vector_store: FAISS, threshold: float) -> Dict:
    """Process questions using simple search without RAG (already fast, no batch needed)"""
    results = []
    
    for query in queries:
        question = query['question']
        category = query.get('category', '')
        
        # Simple search without RAG
        docs_with_scores = vector_store.similarity_search_with_score(question, k=5)
        # Deduplicate sources by document name, keeping highest score
        sources_dict = {}
        for doc, score in docs_with_scores:
            if score >= threshold:
                doc_name = doc.metadata.get('name', '')
                doc_path = doc.metadata.get('path', '')
                score_rounded = round(score, 3)
                # Keep the highest score for each unique document
                if doc_name not in sources_dict or score_rounded > sources_dict[doc_name]['score']:
                    sources_dict[doc_name] = {
                        'name': doc_name,
                        'path': doc_path,
                        'score': score_rounded
                    }
        sources = list(sources_dict.values())

        answer = f"Based on the following documents: {', '.join([s['name'] for s in sources])}" if sources else "No relevant documents found"
        results.append({
            'question': question,
            'category': category,
            'answer': answer,
            'sources': sources,
            'method': 'search',
            'has_answer': bool(sources)
        })

    return {'questions': results}


def search_documents(query: str, document_processor: DocumentProcessor, top_k: int = 5, threshold: float = None):
    """Search documents using the document processor"""
    if not document_processor:
        return []

    return document_processor.search(query, top_k=top_k, threshold=threshold)


def hybrid_search(query: str, vector_store: FAISS, store_name: str,
                 top_k: int = 10, sparse_weight: float = 0.3,
                 dense_weight: float = 0.7, threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
    """
    Hybrid search combining sparse (BM25) and dense retrieval.

    Args:
        query: Search query
        vector_store: FAISS vector store for dense retrieval
        store_name: Name of the document store
        top_k: Number of top results to return
        sparse_weight: Weight for sparse scores (0-1)
        dense_weight: Weight for dense scores (0-1)
        threshold: Minimum similarity threshold for dense retrieval

    Returns:
        Combined search results sorted by hybrid score
    """
    logger.info(f"Performing hybrid search for query: {query[:50]}...")

    # Get sparse results
    sparse_results = []
    bm25_index = load_sparse_index_for_store(store_name)

    if bm25_index:
        sparse_results = bm25_index.search(query, top_k=top_k*2)
        logger.info(f"Sparse search returned {len(sparse_results)} results")
    else:
        logger.warning(f"No sparse index found for {store_name}, falling back to dense only")

    # Get dense results
    dense_docs = vector_store.similarity_search_with_score(query, k=top_k*2)
    dense_results = []

    for doc, score in dense_docs:
        if score >= threshold:
            dense_results.append({
                'doc_id': doc.metadata.get('source', ''),
                'document': doc.page_content,
                'score': float(score),
                'metadata': doc.metadata
            })

    logger.info(f"Dense search returned {len(dense_results)} results")

    # Combine results using reciprocal rank fusion or weighted scoring
    combined_scores = {}

    # Process sparse results
    for result in sparse_results:
        doc_id = result['doc_id']
        combined_scores[doc_id] = {
            'sparse_score': result['score'] * sparse_weight,
            'dense_score': 0.0,
            'result': result
        }

    # Process dense results
    for result in dense_results:
        doc_id = result['doc_id']
        if doc_id in combined_scores:
            combined_scores[doc_id]['dense_score'] = result['score'] * dense_weight
        else:
            combined_scores[doc_id] = {
                'sparse_score': 0.0,
                'dense_score': result['score'] * dense_weight,
                'result': result
            }

    # Calculate final hybrid scores
    final_results = []
    for doc_id, scores in combined_scores.items():
        hybrid_score = scores['sparse_score'] + scores['dense_score']

        # Create unified result format
        result = scores['result'].copy()
        result.update({
            'hybrid_score': hybrid_score,
            'sparse_score': scores['sparse_score'] / sparse_weight if sparse_weight > 0 else 0,
            'dense_score': scores['dense_score'] / dense_weight if dense_weight > 0 else 0,
            'score': hybrid_score  # For backward compatibility
        })
        final_results.append(result)

    # Sort by hybrid score
    final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

    # Return top_k results
    top_results = final_results[:top_k]
    logger.info(f"Hybrid search returned {len(top_results)} final results")

    return top_results


def generate_questions_embeddings():
    """
    Generate embeddings for all questions and save to disk.

    This function should be called during the build process to pre-calculate
    embeddings for all questions from the available question files.
    Uses LLM parsing to ensure consistency with runtime parsing.

    Returns:
        int: Number of embeddings generated and saved
    """
    try:
        from app.core.config import get_config
        from app.core.model_cache import get_cached_embeddings
        from app.core.parsers import parse_questions
        import json
        import numpy as np
        import unidecode

        config = get_config()
        embeddings_model = get_cached_embeddings()
        questions_dir = config.paths['questions_dir']

        logger.info("🔄 Generating questions embeddings using LLM parsing...")

        # Get LLM instance for parsing - use same config as runtime for consistency
        try:
            from langchain_anthropic import ChatAnthropic
            import os
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
            
            # Use exact same configuration as runtime to ensure consistent parsing    
            model = os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514')
            temperature = float(os.getenv('CLAUDE_TEMPERATURE', '0.0'))
            max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', '16000'))
                
            llm = ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"Using LLM config: model={model}, temperature={temperature}, max_tokens={max_tokens}")
        except Exception as e:
            raise RuntimeError(f"Failed to create LLM instance: {e}")

        # Initialize embeddings cache and parsed structures storage
        embeddings_cache = {}
        all_parsed_questions = {}

        # Process all question files
        question_files = list(questions_dir.glob("*.md"))
        if not question_files:
            logger.warning(f"No question files found in {questions_dir}")
            return 0

        for question_file in question_files:
            logger.info(f"Processing questions file: {question_file.name}")

            try:
                # Read question content
                content = question_file.read_text(encoding='utf-8')

                # Parse questions using improved LLM parsing
                parsed_questions = parse_questions(content, llm)
                
                # Store parsed structure for runtime use
                all_parsed_questions[question_file.name] = parsed_questions
                
                # OPTIMIZATION: Collect all question texts for batch embedding generation
                texts_to_embed = []
                text_to_cache_key = {}
                
                for question_data in parsed_questions:
                    question_text = question_data['question']
                    
                    # Process cache key
                    cache_key = question_text.lower().strip()
                    cache_key = unidecode.unidecode(cache_key)
                    cache_key = cache_key.replace('–', '-').replace('—', '-')
                    cache_key = cache_key.replace(''', "'").replace(''', "'")

                    # Skip if already cached
                    if cache_key in embeddings_cache:
                        continue
                        
                    # Add to batch processing lists
                    texts_to_embed.append(question_text)
                    text_to_cache_key[question_text] = cache_key
                
                # BATCH EMBEDDING GENERATION: Process all question texts at once using matrix operations
                if texts_to_embed:
                    logger.info(f"🚀 Batch generating embeddings for {len(texts_to_embed)} questions...")
                    from app.core.performance import optimize_embedding_batch
                    batch_embeddings = optimize_embedding_batch(texts_to_embed, embeddings_model)
                    
                    # Store batch results in cache
                    for question_text, embedding in zip(texts_to_embed, batch_embeddings):
                        cache_key = text_to_cache_key[question_text]
                        if hasattr(embedding, 'tolist'):
                            embeddings_cache[cache_key] = embedding.tolist()
                        elif isinstance(embedding, list):
                            embeddings_cache[cache_key] = embedding
                        else:
                            embeddings_cache[cache_key] = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                        logger.debug(f"✅ Batch embedded: {question_text[:50]}...")
                        
                    logger.info(f"✅ Successfully batch generated {len(batch_embeddings)} question embeddings")

            except Exception as e:
                logger.error(f"Failed to process question file {question_file}: {e}")
                # Continue processing other files even if one fails
                continue

        # Save embeddings to disk
        cache_file = config.paths['faiss_dir'] / "questions_embeddings.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_cache, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Saved {len(embeddings_cache)} questions embeddings to {cache_file}")
        
        # Save parsed question structures to eliminate runtime re-parsing
        structures_file = config.paths['faiss_dir'] / "questions_structures.json"
        with open(structures_file, 'w', encoding='utf-8') as f:
            json.dump(all_parsed_questions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(all_parsed_questions)} parsed question structures to {structures_file}")
        
        if not embeddings_cache:
            raise RuntimeError("No questions embeddings were successfully generated. All question files failed to process.")
        
        return len(embeddings_cache)

    except Exception as e:
        error_msg = f"Failed to generate questions embeddings: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def load_prebuilt_questions(questions_filename: str) -> list:
    """
    Load pre-parsed questions structure from build artifacts.
    
    This function loads question structures that were parsed during build time,
    eliminating the need for runtime LLM parsing and ensuring consistency.
    
    Args:
        questions_filename: Name of the questions file (e.g., 'due diligence.md')
        
    Returns:
        List containing parsed question structures
        
    Raises:
        RuntimeError: If structures file doesn't exist or questions not found
    """
    from app.core.config import get_config
    import json
    
    config = get_config()
    structures_file = config.paths['faiss_dir'] / "questions_structures.json"
    
    if not structures_file.exists():
        raise RuntimeError(
            f"Questions structures file not found: {structures_file}. "
            f"Run build process first to generate pre-parsed structures."
        )
    
    try:
        with open(structures_file, 'r', encoding='utf-8') as f:
            all_structures = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load questions structures: {e}")
    
    if questions_filename not in all_structures:
        available_files = list(all_structures.keys())
        raise RuntimeError(
            f"Questions file '{questions_filename}' not found in pre-built structures. "
            f"Available: {available_files}. Rebuild search indexes."
        )
    
    logger.info(f"✅ Loaded pre-parsed questions structure for: {questions_filename}")
    return all_structures[questions_filename]



#!/usr/bin/env python3
"""
Search and analysis functions for document retrieval and ranking.
"""

# Standard library imports
from typing import Dict, List
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
from app.core.constants import SIMILARITY_THRESHOLD
from app.core.document_processor import DocumentProcessor
from app.core.logging import logger
from app.core.ranking import rerank_results
from app.core.sparse_index import load_sparse_index_for_store, BM25Index


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
    """Compare checklist items directly against LLM-generated document type classifications"""

    # Ensure checklist embeddings are preloaded first
    if not hasattr(get_checklist_embedding, '_cache') or not get_checklist_embedding._cache:
        logger.info("Checklist embeddings cache is empty, preloading...")
        try:
            from app.core.search import preload_checklist_embeddings
            count = preload_checklist_embeddings()
            logger.info(f"✅ Preloaded {count} checklist embeddings for processing")
        except Exception as e:
            logger.error(f"Failed to preload checklist embeddings: {e}")
            return {}

    # Ensure document type embeddings are available
    if session:
        logger.debug(f"Checklist processing session ID: {id(session)}, has embeddings: {hasattr(session, 'document_type_embeddings')}")
        if hasattr(session, 'document_type_embeddings'):
            logger.debug(f"Embeddings count: {len(session.document_type_embeddings) if session.document_type_embeddings else 0}")

    # Try to auto-preload embeddings if missing
    embeddings_missing = not session or not hasattr(session, 'document_type_embeddings') or not session.document_type_embeddings

    if embeddings_missing and store_name:
        logger.info(f"Document type embeddings missing, attempting auto-preload for {store_name}...")
        try:
            from app.core.search import preload_document_type_embeddings
            type_embeddings = preload_document_type_embeddings(store_name)
            if not hasattr(session, 'document_type_embeddings') or session.document_type_embeddings is None:
                session.document_type_embeddings = {}
            session.document_type_embeddings.update(type_embeddings)
            logger.info(f"✅ Auto-preloaded {len(type_embeddings)} document type embeddings")
            embeddings_missing = False
        except Exception as e:
            logger.warning(f"Failed to auto-preload document type embeddings: {e}")

    if embeddings_missing:
        logger.error("Document type embeddings not available. Checklist processing requires preloaded embeddings.")
        logger.error("Make sure data room processing completed successfully or embeddings can be auto-loaded.")
        return {}

    # Load document type classifications - these are our primary comparison targets
    doc_types = {}
    if store_name:
        doc_types = _load_document_types(vector_store, store_name)

    if not doc_types:
        logger.warning(f"No document type classifications found for {store_name}")
        return {}

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
            matches = []

            # Compare checklist item against each document's type classification
            for doc_path, doc_type in doc_types.items():
                if not doc_type or doc_type == 'not classified':
                    continue

                doc_type_lower = doc_type.lower().strip()

                # Calculate semantic similarity between checklist item and document type
                try:
                    # Get checklist embedding from memory cache (preloaded during data room processing)
                    checklist_embedding = get_checklist_embedding(checklist_item_text)

                    # Get document type embedding (from preloaded cache)
                    doc_type_embedding = get_document_type_embedding(doc_type_lower, session)

                    # Calculate cosine similarity
                    import numpy as np
                    similarity = np.dot(checklist_embedding, doc_type_embedding) / (
                        np.linalg.norm(checklist_embedding) * np.linalg.norm(doc_type_embedding)
                    )

                    # Only include matches above threshold
                    if similarity >= threshold:
                        # Find the document metadata from the vector store
                        # We need to get the document name and other metadata
                        doc_name = _extract_doc_name_from_path(doc_path)

                        matches.append({
                            'name': doc_name,
                            'path': doc_path,
                            'full_path': doc_path,  # For consistency
                            'score': round(float(similarity), 3),
                            'document_type': doc_type,
                            'text': f"Document type: {doc_type}"  # Include document type as text
                        })

                except Exception as e:
                    logger.warning(f"Error calculating similarity for {doc_path}: {e}")
                    continue

            # Sort matches by score (highest first)
            matches.sort(key=lambda x: x['score'], reverse=True)

            # Limit to top matches for performance
            matches = matches[:10]

            if matches:
                cat_results['matched_items'] += 1
                logger.info(f"✅ Found {len(matches)} matches for checklist item: '{checklist_item_text[:50]}...'")

            cat_results['items'].append({
                'text': item['text'],
                'original': item['original'],
                'matches': matches
            })

        results[cat_letter] = cat_results

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
        logger.warning(f"Failed to load document types for {store_name}: {e}")
    return {}


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
        logger.warning("Checklist embedding cache was not initialized - this should not happen!")

    # Create cache key from checklist text with normalized Unicode
    cache_key = checklist_text.lower().strip()
    # Use unidecode for comprehensive Unicode to ASCII conversion
    cache_key = unidecode.unidecode(cache_key)

    # Check in-memory cache only
    if cache_key in get_checklist_embedding._cache:
        return get_checklist_embedding._cache[cache_key]

    # Enhanced debugging for troubleshooting
    cache_size = len(get_checklist_embedding._cache)
    logger.warning(f"Checklist embedding not found: '{checklist_text[:50]}...'")
    logger.warning(f"Cache key generated: '{cache_key}'")
    logger.warning(f"Cache has {cache_size} items total")

    if cache_size > 0:
        # Look for similar keys to help debug
        similar_keys = []
        search_terms = checklist_text.lower().split()
        for key in get_checklist_embedding._cache.keys():
            if any(term in key for term in search_terms if len(term) > 3):
                similar_keys.append(key)

        if similar_keys:
            logger.warning(f"Similar keys found: {similar_keys[:3]}")
        else:
            logger.warning("No similar keys found in cache")

        # Show a few sample keys
        sample_keys = list(get_checklist_embedding._cache.keys())[:5]
        logger.warning(f"Sample cache keys: {sample_keys}")
    else:
        logger.error("Cache is completely empty - embeddings were not preloaded!")

    # Fail if not found - no fallbacks
    raise RuntimeError(
        f"Checklist embedding not found for: '{checklist_text[:50]}...' (cache key: '{cache_key}'). "
        f"Cache has {cache_size} items. "
        "Make sure embeddings were preloaded during data room processing."
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

    Returns:
        int: Number of embeddings generated and saved
    """
    try:
        from app.core.config import get_config
        from app.core.model_cache import get_cached_embeddings
        import json
        import numpy as np

        config = get_config()
        embeddings_model = get_cached_embeddings()
        checklist_dir = config.paths['checklist_dir']

        logger.info("🔄 Generating checklist embeddings...")

        # Initialize embeddings cache
        embeddings_cache = {}

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

                # Parse checklist items from markdown
                checklist_items = _parse_checklist_items_from_markdown(content)

                # Generate embeddings for each item
                for item_text in checklist_items:
                    # Normalize Unicode in cache key
                    cache_key = item_text.lower().strip()
                    cache_key = unidecode.unidecode(cache_key)

                    # Skip if already processed
                    if cache_key in embeddings_cache:
                        continue

                    try:
                        # Generate embedding
                        embedding = embeddings_model.embed_query(item_text)

                        # Handle both list and numpy array cases
                        if hasattr(embedding, 'tolist'):
                            embeddings_cache[cache_key] = embedding.tolist()
                        else:
                            # Already a list
                            embeddings_cache[cache_key] = embedding

                        logger.debug(f"✅ Embedded: {item_text[:50]}...")

                    except Exception as e:
                        logger.warning(f"Failed to embed checklist item '{item_text[:50]}...': {e}")
                        continue

            except Exception as e:
                logger.error(f"Failed to process checklist file {checklist_file}: {e}")
                continue

        # Save to disk
        cache_file = config.paths['faiss_dir'] / "checklist_embeddings.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_cache, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Saved {len(embeddings_cache)} checklist embeddings to {cache_file}")
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
            logger.warning(f"Checklist embeddings file not found: {cache_file}")
            logger.info("Generating checklist embeddings now...")

            # Try to generate embeddings on-the-fly
            try:
                generated_count = generate_checklist_embeddings()
                if generated_count > 0:
                    logger.info(f"✅ Generated {generated_count} embeddings, now preloading...")
                else:
                    raise RuntimeError("No checklist items found to embed")
            except Exception as gen_error:
                raise RuntimeError(
                    f"Could not generate checklist embeddings: {gen_error}. "
                    "Make sure checklist files exist and are properly formatted."
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
    Preload all document type embeddings into memory during data room processing.

    This function loads document type classifications and computes their embeddings
    once during data room processing to avoid runtime computation.

    Returns:
        dict: Dictionary mapping normalized document types to their embeddings

    Raises:
        RuntimeError: If document types can't be loaded or embeddings can't be computed
    """
    try:
        from app.core.model_cache import get_cached_embeddings
        import numpy as np

        # Load document type classifications
        doc_types = _load_document_types(None, store_name)
        if not doc_types:
            raise RuntimeError(f"No document type classifications found for {store_name}")

        # Get embeddings model
        embeddings = get_cached_embeddings()

        # Precompute embeddings for all unique document types
        type_embeddings = {}
        unique_types = set()

        # Collect all unique document types
        for doc_path, doc_type in doc_types.items():
            if doc_type and doc_type != 'not classified':
                normalized_type = unidecode.unidecode(doc_type.lower().strip())
                unique_types.add(normalized_type)

        # Precompute embeddings for each unique type
        for doc_type in unique_types:
            try:
                embedding = embeddings.embed_query(doc_type)
                # Ensure it's a numpy array
                if hasattr(embedding, 'tolist'):
                    embedding_array = np.array(embedding, dtype=np.float32)
                else:
                    embedding_array = np.array(embedding, dtype=np.float32)
                type_embeddings[doc_type] = embedding_array
            except Exception as e:
                logger.warning(f"Failed to compute embedding for document type '{doc_type}': {e}")
                continue

        logger.info(f"✅ Precomputed {len(type_embeddings)} document type embeddings")
        return type_embeddings

    except Exception as e:
        error_msg = f"Failed to preload document type embeddings: {e}"
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
        
        # Create context and sources
        if relevant_docs:
            context = "\n".join([f"- {doc.metadata.get('name', 'Unknown')}: {doc.page_content[:200]}..." 
                               for doc in relevant_docs[:5]])
            sources = [{'name': doc.metadata.get('name', ''), 
                       'path': doc.metadata.get('path', ''),
                       'score': round(1.0 - (score / 2.0) if score <= 2.0 else 0.0, 3)}
                      for doc, score in docs_with_scores[:5] if (1.0 - (score / 2.0) if score <= 2.0 else 0.0) >= threshold]
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
        sources = []
        for doc, score in docs_with_scores:
            if score >= threshold:
                sources.append({
                    'name': doc.metadata.get('name', ''),
                    'path': doc.metadata.get('path', ''),
                    'score': round(score, 3)
                })

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



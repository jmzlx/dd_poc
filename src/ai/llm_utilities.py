#!/usr/bin/env python3
"""
LLM Utilities Module

This module contains utility functions for batch processing, document
summarization, embeddings, and checklist matching operations.
"""

import time
import random
from typing import Dict, List, Any, Optional

try:
    import streamlit as st
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    import numpy as np
    import faiss
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    st = None
    ChatAnthropic = object
    HumanMessage = object

from ..config import get_config
from .prompts import get_description_generation_prompt, get_document_summarization_prompt


def exponential_backoff_retry(func, max_retries: Optional[int] = None, base_delay: Optional[float] = None):
    """
    Execute function with exponential backoff retry logic for rate limiting.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retries (uses config default if None)
        base_delay: Base delay in seconds (uses config default if None)
        
    Returns:
        Result of the function call
    """
    config = get_config()
    if max_retries is None:
        max_retries = config.api.max_retries
    if base_delay is None:
        base_delay = config.api.base_delay
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a rate limiting error
            if any(keyword in error_str for keyword in ['rate', 'limit', 'quota', 'throttl', '429', 'too many']):
                if attempt < max_retries - 1:
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Rate limit exceeded after {max_retries} attempts")
                    raise e
            else:
                # Non-rate limit error, don't retry
                raise e
    return None


def generate_checklist_descriptions(checklist: Dict, llm: ChatAnthropic, batch_size: Optional[int] = None) -> Dict:
    """
    Generate detailed descriptions for each checklist item explaining what documents should satisfy it.
    Returns checklist with added 'description' field for each item.
    
    Args:
        checklist: Checklist dictionary to enhance
        llm: ChatAnthropic instance for generating descriptions
        batch_size: Number of items to process in each batch (uses config default if None)
        
    Returns:
        Enhanced checklist with descriptions
    """
    if not DEPENDENCIES_AVAILABLE:
        return checklist
    
    config = get_config()
    if batch_size is None:
        batch_size = config.processing.description_batch_size
    

    
    # Process all checklist items
    enhanced_checklist = {}
    all_items_to_process = []
    
    # Collect all items with their context
    for cat_letter, category in checklist.items():
        cat_name = category.get('name', '')
        enhanced_checklist[cat_letter] = {
            'name': cat_name,
            'letter': cat_letter,
            'items': []
        }
        
        for item in category.get('items', []):
            item_data = {
                'category_letter': cat_letter,
                'category_name': cat_name,
                'item_text': item.get('text', ''),
                'original_item': item,
                'prompt': get_description_generation_prompt(cat_name, item.get('text', ''))
            }
            all_items_to_process.append(item_data)
    
    # Process items in batches
    total_items = len(all_items_to_process)
    total_batches = (total_items + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, total_items, batch_size), 1):
        batch = all_items_to_process[i:i + batch_size]
        batch_end = min(i + batch_size, total_items)
        
        # Update progress if available
        if st and hasattr(st, 'progress') and 'description_progress' in st.session_state:
            progress = i / total_items
            st.session_state.description_progress.progress(
                progress, 
                text=f"📝 Generating descriptions batch {batch_num}/{total_batches} (items {i+1}-{batch_end} of {total_items})"
            )
        
        # Create prompts for batch processing
        prompts = [item_data['prompt'] for item_data in batch]
        messages_batch = [[HumanMessage(content=prompt)] for prompt in prompts]
        
        # Use exponential backoff for batch processing
        def process_descriptions_batch():
            # Use higher concurrency for descriptions since they're short
            max_concurrent = min(batch_size * 2, config.api.max_concurrent_requests)
            return llm.batch(
                messages_batch, 
                config={"max_concurrency": max_concurrent}
            )
        
        try:
            responses = exponential_backoff_retry(
                process_descriptions_batch, 
                max_retries=config.api.max_retries, 
                base_delay=config.api.batch_base_delay
            )
            
            # Extract descriptions from responses
            batch_descriptions = [response.content.strip() if response else f"Documents related to {item_data['item_text']}" 
                                for response, item_data in zip(responses, batch)]
        except Exception as e:
            # Fallback to sequential processing with individual retries if batch fails
            print(f"Batch {batch_num} description generation failed: {e}. Falling back to sequential with retries.")
            batch_descriptions = []
            for item_data in batch:
                def single_description_process():
                    return llm.invoke([HumanMessage(content=item_data['prompt'])])
                
                try:
                    response = exponential_backoff_retry(
                        single_description_process, 
                        max_retries=config.api.batch_retry_attempts, 
                        base_delay=config.api.single_retry_base_delay
                    )
                    batch_descriptions.append(response.content.strip())
                except Exception as inner_e:
                    print(f"Failed to generate description for {item_data['item_text']}: {inner_e}")
                    batch_descriptions.append(f"Documents related to {item_data['item_text']}")
        
        # Add descriptions to items
        for item_data, description in zip(batch, batch_descriptions):
            enhanced_item = item_data['original_item'].copy()
            enhanced_item['description'] = description
            enhanced_checklist[item_data['category_letter']]['items'].append(enhanced_item)
        
        # No delay between batches - using rate limiting with exponential backoff instead
    
    return enhanced_checklist


def batch_summarize_documents(documents: List[Dict], llm: ChatAnthropic, batch_size: Optional[int] = None) -> List[Dict]:
    """
    Summarize documents using LangChain's built-in batch processing for true parallelization.
    Optimized with larger batches, higher concurrency, and exponential backoff rate limiting.
    Returns documents with added 'summary' field.
    
    Args:
        documents: List of document dictionaries to summarize
        llm: ChatAnthropic instance for generating summaries
        batch_size: Number of documents to process in each batch (uses config default if None)
        
    Returns:
        List of documents with added summary field
    """
    if not DEPENDENCIES_AVAILABLE:
        return documents
    
    config = get_config()
    if batch_size is None:
        batch_size = config.processing.batch_size
    
    # Process documents in batches
    summarized_docs = []
    total_docs = len(documents)
    total_batches = (total_docs + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, total_docs, batch_size), 1):
        batch = documents[i:i + batch_size]
        batch_end = min(i + batch_size, total_docs)
        
        # Update progress with batch info
        if st and hasattr(st, 'progress') and 'summary_progress' in st.session_state:
            progress = i / total_docs
            st.session_state.summary_progress.progress(
                progress, 
                text=f"📝 Processing batch {batch_num}/{total_batches} (docs {i+1}-{batch_end} of {total_docs})"
            )
        
        # Create prompts for all documents in the batch
        prompts = [get_document_summarization_prompt(doc) for doc in batch]
        
        # Convert prompts to HumanMessage format for batch processing
        messages_batch = [[HumanMessage(content=prompt)] for prompt in prompts]
        
        # Use exponential backoff for batch processing
        def process_batch():
            max_concurrent = min(batch_size, config.api.max_concurrent_requests)
            return llm.batch(
                messages_batch, 
                config={"max_concurrency": max_concurrent}
            )
        
        try:
            responses = exponential_backoff_retry(
                process_batch, 
                max_retries=config.api.max_retries, 
                base_delay=config.api.batch_base_delay
            )
            
            # Extract summaries from responses
            batch_summaries = [response.content.strip() if response else f"Document: {doc.get('name', 'Unknown')}" 
                              for response, doc in zip(responses, batch)]
        except Exception as e:
            # Fallback to sequential processing with individual retries if batch fails
            print(f"Batch {batch_num} processing failed: {e}. Falling back to sequential with retries.")
            batch_summaries = []
            for doc_idx, doc in enumerate(batch):
                prompt = get_document_summarization_prompt(doc)
                
                def single_doc_process():
                    return llm.invoke([HumanMessage(content=prompt)])
                
                try:
                    response = exponential_backoff_retry(
                        single_doc_process, 
                        max_retries=config.api.batch_retry_attempts, 
                        base_delay=config.api.single_retry_base_delay
                    )
                    batch_summaries.append(response.content.strip())
                except Exception as inner_e:
                    print(f"Failed to summarize {doc.get('name', 'Unknown')}: {inner_e}")
                    batch_summaries.append(f"Document: {doc.get('name', 'Unknown')}")
                
                # Update progress within fallback
                if st and hasattr(st, 'progress') and 'summary_progress' in st.session_state:
                    sub_progress = (i + doc_idx + 1) / total_docs
                    st.session_state.summary_progress.progress(
                        sub_progress,
                        text=f"📝 Sequential fallback: {i + doc_idx + 1}/{total_docs}"
                    )
        
        # Add summaries to documents
        for doc, summary in zip(batch, batch_summaries):
            doc['summary'] = summary
            summarized_docs.append(doc)
        
        # No delay between batches - using rate limiting with exponential backoff instead
    
    return summarized_docs


def create_document_embeddings_with_summaries(documents: List[Dict], model) -> Dict[str, Any]:
    """
    Create embeddings for documents using their LLM-generated summaries.
    
    Args:
        documents: List of documents with summaries
        model: SentenceTransformer model for embeddings
        
    Returns:
        Dictionary with document info and embeddings
    """
    doc_embeddings = []
    doc_info = []
    
    for doc in documents:
        # Combine filename, path context, and LLM summary for rich embedding
        doc_name = doc.get('name', 'Unknown')
        doc_path = doc.get('path', '')
        summary = doc.get('summary', '')
        
        # Create rich text representation
        embedding_text = f"{doc_name}\n{doc_path}\n{summary}"
        
        # Generate embedding
        embedding = model.encode(embedding_text)
        
        doc_embeddings.append(embedding)
        doc_info.append({
            'name': doc_name,
            'path': doc_path,
            'full_path': doc.get('full_path', doc_path),
            'summary': summary,
            'embedding_text': embedding_text,
            'original_doc': doc
        })
    
    return {
        'embeddings': doc_embeddings,
        'documents': doc_info
    }


def match_checklist_with_summaries(
    checklist: Dict, 
    doc_embeddings_data: Dict,
    model,
    threshold: Optional[float] = None
) -> Dict:
    """
    Match checklist items against document summaries using FAISS for 10x faster similarity search.
    Enhanced to use LLM-generated descriptions for better semantic matching.
    
    Args:
        checklist: Checklist dictionary with items and descriptions
        doc_embeddings_data: Dictionary containing document embeddings and info
        model: SentenceTransformer model for embeddings
        threshold: Similarity threshold for matching (uses config default if None)
        
    Returns:
        Dictionary with matching results
    """
    if not DEPENDENCIES_AVAILABLE:
        return {}
    
    config = get_config()
    if threshold is None:
        threshold = config.processing.similarity_threshold
    
    doc_embeddings = np.array(doc_embeddings_data['embeddings'], dtype='float32')
    doc_info = doc_embeddings_data['documents']
    
    # Build FAISS index for fast similarity search
    faiss.normalize_L2(doc_embeddings)  # Normalize for cosine similarity
    dimension = doc_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(doc_embeddings)
    
    results = {}
    
    for cat_letter, category in checklist.items():
        cat_name = category.get('name', '')
        cat_results = {
            'name': cat_name,
            'letter': cat_letter,
            'total_items': len(category.get('items', [])),
            'matched_items': 0,
            'items': []
        }
        
        for item in category.get('items', []):
            item_text = item.get('text', '')
            item_description = item.get('description', '')
            
            # Create enhanced embedding text using both item text and generated description
            if item_description:
                # Use the LLM-generated description for richer semantic matching
                checklist_embedding_text = f"{cat_name}: {item_text}\n{item_description}"
            else:
                # Fallback to original method if no description available
                checklist_embedding_text = f"{cat_name}: {item_text}"
            
            # Create and normalize item embedding
            item_embedding = model.encode(checklist_embedding_text).astype('float32').reshape(1, -1)
            faiss.normalize_L2(item_embedding)
            
            # Use FAISS for fast similarity search
            scores, indices = faiss_index.search(item_embedding, len(doc_info))
            
            # Find matching documents above threshold
            matches = []
            min_display_threshold = config.processing.min_display_threshold
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # No more results
                    break
                if score < min_display_threshold:  # Skip very low scoring documents
                    break  # Scores are sorted, so we can stop here
                
                match_data = {
                    'name': doc_info[idx]['name'],
                    'path': doc_info[idx]['path'],
                    'full_path': doc_info[idx].get('full_path', doc_info[idx]['path']),
                    'summary': doc_info[idx]['summary'],
                    'score': float(score),
                    'metadata': doc_info[idx].get('original_doc', {}).get('metadata', {})
                }
                
                matches.append(match_data)
            
            # Keep top 5 matches for display
            display_matches = matches[:5]
            
            item_result = {
                'text': item_text,
                'original': item.get('original', item_text),
                'description': item_description,  # Include the generated description
                'matches': display_matches
            }
            
            # Count items with ANY matches (both green and yellow) toward category total
            if display_matches:
                cat_results['matched_items'] += 1
            
            cat_results['items'].append(item_result)
        
        results[cat_letter] = cat_results
    
    return results

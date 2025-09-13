#!/usr/bin/env python3
"""
Ranking utilities for search results reranking.

This module provides functions for reranking search results using cross-encoder models
to improve relevance scoring. Separated from search.py to avoid circular imports.
"""

from typing import Dict, List
from app.core.logging import logger
from app.core.model_cache import get_cached_cross_encoder


def rerank_results(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    Rerank search results using cross-encoder model for improved relevance

    Args:
        query: The search query
        candidates: List of candidate documents with 'text', 'score', etc.

    Returns:
        Reranked list of candidates with updated scores
    """
    if not candidates:
        return candidates

    try:
        # Get cached cross-encoder model
        cross_encoder = get_cached_cross_encoder()

        # Prepare input pairs for cross-encoder
        query_doc_pairs = [(query, candidate['text']) for candidate in candidates]

        # Get cross-encoder scores
        ce_scores = cross_encoder.predict(query_doc_pairs)

        # Update candidates with reranked scores
        for i, candidate in enumerate(candidates):
            candidate['reranked_score'] = float(ce_scores[i])
            candidate['score'] = float(ce_scores[i])  # Update main score for consistency

        # Sort by reranked score (higher is better for cross-encoder)
        candidates.sort(key=lambda x: x['reranked_score'], reverse=True)

        logger.info(f"Reranked {len(candidates)} results using cross-encoder")
        return candidates

    except Exception as e:
        logger.warning(f"Cross-encoder reranking failed: {e}. Using original scores.")
        return candidates

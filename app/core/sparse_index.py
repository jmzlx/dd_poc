#!/usr/bin/env python3
"""
BM25 Sparse Index Implementation for Due Diligence Documents

This module provides BM25-based sparse retrieval that complements the existing
dense retrieval system. The index is pre-calculated locally and persisted
to disk for fast loading on Streamlit Cloud.
"""

import pickle
import os
import re
from typing import List, Dict, Optional, Callable, Tuple
from pathlib import Path

from rank_bm25 import BM25Okapi
from app.core.logging import logger


class BM25Index:
    """
    BM25-based sparse index for document retrieval.

    This class provides:
    - Pre-calculated BM25 index persistence
    - Custom tokenization for legal/financial documents
    - Efficient search with relevance scoring
    - Integration with existing document processing pipeline
    """

    def __init__(self, index_path: str):
        """
        Initialize BM25 index.

        Args:
            index_path: Path to save/load the index file
        """
        self.index_path = Path(index_path)
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.metadata: Dict = {}

    def custom_tokenizer(self, text: str) -> List[str]:
        """
        Custom tokenization optimized for legal and financial documents.

        Handles:
        - Legal abbreviations (LLC, Inc., Corp.)
        - Financial terms (IPO, GAAP, EBITDA)
        - Contract terminology (force majeure, indemnification)
        - Proper names and entities
        """
        if not text:
            return []

        # Convert to lowercase
        text = text.lower()

        # Preserve important legal/financial abbreviations
        legal_abbrevs = [
            'llc', 'inc', 'corp', 'ltd', 'co', 'lp', 'llp',
            'ipo', 'gaap', 'sec', 'fdic', 'irs', 'sox', 'gdpr',
            'nda', 'mou', 'spa', 'joa', 'ipa', 'dpa'
        ]

        # Replace common terms to avoid splitting
        for abbrev in legal_abbrevs:
            text = text.replace(abbrev, abbrev.replace(' ', '_'))

        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)

        # Restore underscores to spaces for abbreviations
        tokens = [token.replace('_', '') for token in tokens]

        # Filter out very short tokens (likely noise)
        tokens = [token for token in tokens if len(token) > 1]

        return tokens

    def build_index(self, documents: List[Dict[str, str]], custom_tokenizer: Optional[Callable] = None):
        """
        Build BM25 index from documents.

        Args:
            documents: List of dicts with 'id' and 'content' keys
            custom_tokenizer: Optional custom tokenization function
        """
        logger.info(f"Building BM25 index from {len(documents)} documents")

        # Extract content and IDs
        self.documents = [doc['content'] for doc in documents]
        self.doc_ids = [doc['id'] for doc in documents]

        # Tokenize documents
        tokenizer = custom_tokenizer or self.custom_tokenizer
        self.tokenized_docs = [tokenizer(doc) for doc in self.documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Store metadata
        self.metadata = {
            'total_documents': len(self.documents),
            'total_tokens': sum(len(tokens) for tokens in self.tokenized_docs),
            'avg_tokens_per_doc': sum(len(tokens) for tokens in self.tokenized_docs) / len(self.documents) if self.documents else 0
        }

        # Save to disk
        self._save_index()

        logger.info(f"✅ BM25 index built and saved: {self.metadata}")

    def _save_index(self):
        """Save index to pickle file"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        index_data = {
            'bm25': self.bm25,
            'documents': self.documents,
            'doc_ids': self.doc_ids,
            'tokenized_docs': self.tokenized_docs,
            'metadata': self.metadata
        }

        with open(self.index_path, 'wb') as f:
            pickle.dump(index_data, f)

        logger.info(f"💾 BM25 index saved to {self.index_path}")

    def load_index(self) -> bool:
        """
        Load index from disk.

        Returns:
            True if index loaded successfully, False otherwise
        """
        if self.index_path.exists():
            try:
                with open(self.index_path, 'rb') as f:
                    index_data = pickle.load(f)

                self.bm25 = index_data['bm25']
                self.documents = index_data['documents']
                self.doc_ids = index_data['doc_ids']
                self.tokenized_docs = index_data['tokenized_docs']
                self.metadata = index_data.get('metadata', {})

                logger.info(f"📂 BM25 index loaded: {len(self.documents)} documents")
                return True

            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                return False
        else:
            logger.warning(f"BM25 index not found: {self.index_path}")
            return False

    def search(self, query: str, top_k: int = 10, custom_tokenizer: Optional[Callable] = None) -> List[Dict]:
        """
        Search the BM25 index.

        Args:
            query: Search query
            top_k: Number of top results to return
            custom_tokenizer: Optional custom tokenization function

        Returns:
            List of search results with scores
        """
        if not self.bm25:
            logger.warning("BM25 index not loaded")
            return []

        # Tokenize query
        tokenizer = custom_tokenizer or self.custom_tokenizer
        tokenized_query = tokenizer(query)

        if not tokenized_query:
            logger.warning("Query produced no tokens")
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top results
        if len(scores) == 0:
            return []

        # Get indices of top scores (handling edge case of fewer results than requested)
        num_results = min(top_k, len(scores))
        top_indices = scores.argsort()[-num_results:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return relevant results
                results.append({
                    'doc_id': self.doc_ids[idx],
                    'document': self.documents[idx],
                    'score': float(scores[idx]),
                    'rank': len(results) + 1
                })

        logger.debug(f"BM25 search returned {len(results)} results for query: {query[:50]}...")
        return results

    def get_stats(self) -> Dict:
        """Get index statistics"""
        if not self.index_path.exists():
            return {'status': 'index_not_found'}

        stats = {
            'index_path': str(self.index_path),
            'index_exists': self.index_path.exists(),
            'is_loaded': self.bm25 is not None,
            'index_size_mb': self.index_path.stat().st_size / (1024 * 1024) if self.index_path.exists() else 0
        }

        if self.metadata:
            stats.update(self.metadata)

        return stats


def build_sparse_index_for_store(store_name: str, documents: List[Dict[str, str]],
                                index_dir: str = "data/search_indexes") -> BM25Index:
    """
    Convenience function to build sparse index for a document store.

    Args:
        store_name: Name of the document store (e.g., 'summit-digital-solutions-inc')
        documents: List of documents with 'id' and 'content' keys
        index_dir: Directory to store the index

    Returns:
        BM25Index instance
    """
    index_path = f"{index_dir}/{store_name}_bm25.pkl"
    bm25_index = BM25Index(index_path)
    bm25_index.build_index(documents)
    return bm25_index


def load_sparse_index_for_store(store_name: str, index_dir: str = "data/search_indexes") -> Optional[BM25Index]:
    """
    Convenience function to load sparse index for a document store.

    Args:
        store_name: Name of the document store
        index_dir: Directory containing the index

    Returns:
        BM25Index instance if found, None otherwise
    """
    index_path = f"{index_dir}/{store_name}_bm25.pkl"
    bm25_index = BM25Index(index_path)

    if bm25_index.load_index():
        return bm25_index

    return None

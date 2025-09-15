#!/usr/bin/env python3
"""
Entity Resolution Module

This module provides embedding-based entity resolution for knowledge graphs,
using semantic similarity to identify and merge duplicate entities.

Key features:
- Leverages existing sentence transformer models
- Contextual entity matching using document context
- Configurable similarity thresholds per entity type
- Preserves provenance and merge history
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from app.core.logging import logger
from app.core.config import get_config


class EntityResolver:
    """
    Resolves duplicate entities using semantic embeddings and clustering.
    
    This class identifies and merges similar entities based on their semantic
    similarity, using pre-trained sentence transformers and contextual information.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the entity resolver.
        
        Args:
            model_path: Path to sentence transformer model. If None, uses default from config.
        """
        self.config = get_config()
        
        # Use existing model from project
        if model_path is None:
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "models" / "sentence_transformers" / "all-mpnet-base-v2"
        
        self.model_path = Path(model_path)
        self.model: Optional[SentenceTransformer] = None
        
        # Entity-specific similarity thresholds (higher = more strict)
        self.similarity_thresholds = {
            'people': 0.85,      # Strict for people (names are distinctive)
            'companies': 0.80,   # Moderate for companies (more variation)
            'financial_metrics': 0.90,  # Very strict (numbers should be exact)
            'documents': 0.75,   # Looser for documents (filename variations)
            'legal_keywords': 0.95  # Very strict for legal keywords (exact matches only)
        }
        
        # Context weights for different entity types
        self.context_weights = {
            'people': 0.7,       # Names + context both important
            'companies': 0.6,    # Names more important than context
            'financial_metrics': 0.9,  # Numbers are most important
            'documents': 0.5,    # Context less important for docs
            'legal_keywords': 0.8  # Context important for legal keywords
        }
        
    def _load_model(self):
        """Load the sentence transformer model lazily"""
        if self.model is None:
            logger.info(f"Loading sentence transformer model from {self.model_path}")
            try:
                self.model = SentenceTransformer(str(self.model_path))
                logger.info("✅ Entity resolution model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Could not load sentence transformer model: {e}")
    
    def _create_entity_text(self, entity: Dict[str, Any], entity_type: str) -> str:
        """
        Create rich text representation for an entity.
        
        Args:
            entity: Entity dictionary with name, context, etc.
            entity_type: Type of entity (people, companies, etc.)
            
        Returns:
            String representation combining name and context
        """
        name = entity.get('name', '').strip()
        context = entity.get('context', '').strip()
        
        # Weight name vs context based on entity type
        context_weight = self.context_weights.get(entity_type, 0.6)
        
        if context and context_weight > 0.5:
            # For entities where context matters more, include more context
            context_snippet = context[:150] if len(context) > 150 else context
            return f"{name} {context_snippet}"
        else:
            # For entities where name matters most, include minimal context
            context_snippet = context[:50] if len(context) > 50 else context
            return f"{name} {context_snippet}".strip()
    
    def _normalize_entity_name(self, name: str, entity_type: str) -> str:
        """
        Apply basic normalization rules to entity names.
        
        Args:
            name: Raw entity name
            entity_type: Type of entity
            
        Returns:
            Normalized entity name
        """
        import re
        
        # Basic cleanup
        name = name.strip()
        
        if entity_type == 'companies':
            # Remove common company suffixes for better matching
            name = re.sub(r',?\s*(Inc\.?|LLC|Corp\.?|Corporation|Ltd\.?|Limited)\.?$', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s+', ' ', name).strip()
        
        elif entity_type == 'people':
            # Normalize titles and degrees
            name = re.sub(r'^(Dr\.?|Mr\.?|Ms\.?|Mrs\.?)\s+', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s+\([^)]+\)$', '', name)  # Remove trailing (Title) 
            name = re.sub(r'\s+', ' ', name).strip()
        
        elif entity_type == 'financial_metrics':
            # Normalize financial formatting
            name = re.sub(r'[\s,]', '', name)  # Remove spaces and commas from numbers
            name = name.upper()  # Standardize currency symbols
        
        return name
    
    def _cluster_entities(self, embeddings: np.ndarray, entity_type: str) -> np.ndarray:
        """
        Cluster entities based on their embeddings.
        
        Args:
            embeddings: Entity embeddings matrix
            entity_type: Type of entities being clustered
            
        Returns:
            Cluster labels array
        """
        if len(embeddings) < 2:
            return np.array([0] * len(embeddings))
        
        # Get similarity threshold for this entity type
        similarity_threshold = self.similarity_thresholds.get(entity_type, 0.8)
        distance_threshold = 1.0 - similarity_threshold
        
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage='average',
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(embeddings)
            return cluster_labels
            
        except Exception as e:
            logger.warning(f"Clustering failed for {entity_type}: {e}. Using no clustering.")
            return np.arange(len(embeddings))  # Each entity in its own cluster
    
    def _select_canonical_entity(self, entity_cluster: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Select the best representative entity from a cluster.
        
        Args:
            entity_cluster: List of (index, entity) tuples in the cluster
            
        Returns:
            Canonical entity with merged information
        """
        if len(entity_cluster) == 1:
            return entity_cluster[0][1]
        
        # Score entities by quality metrics
        scored_entities = []
        for idx, entity in entity_cluster:
            score = 0.0
            
            # Prefer higher confidence
            confidence = entity.get('confidence', 0.0)
            score += confidence * 0.4
            
            # Prefer longer, more informative contexts
            context_length = len(entity.get('context', ''))
            score += min(context_length / 200.0, 1.0) * 0.3
            
            # Prefer entities from transformer extraction (usually higher quality)
            if entity.get('extraction_method') == 'transformer':
                score += 0.2
            elif entity.get('extraction_method') == 'document_metadata':
                score += 0.1
            
            # Prefer entities with cleaner names (fewer special characters)
            name_quality = 1.0 - (len([c for c in entity.get('name', '') if not c.isalnum() and c != ' ']) / max(len(entity.get('name', '')), 1))
            score += name_quality * 0.1
            
            scored_entities.append((score, idx, entity))
        
        # Select highest scoring entity as canonical
        best_score, best_idx, canonical_entity = max(scored_entities)
        
        # Enhance canonical entity with merged information
        all_sources = set()
        all_contexts = []
        confidence_scores = []
        
        for _, entity in entity_cluster:
            if entity.get('source'):
                all_sources.add(entity['source'])
            if entity.get('context'):
                all_contexts.append(entity['context'])
            if entity.get('confidence'):
                confidence_scores.append(entity['confidence'])
        
        # Update canonical entity with merged data
        canonical_entity = canonical_entity.copy()
        canonical_entity['sources'] = list(all_sources)
        canonical_entity['merged_contexts'] = all_contexts[:3]  # Keep top 3 contexts
        canonical_entity['cluster_size'] = len(entity_cluster)
        canonical_entity['merged_confidence'] = np.mean(confidence_scores) if confidence_scores else canonical_entity.get('confidence', 0.0)
        canonical_entity['resolution_method'] = 'embedding_clustering'
        
        return canonical_entity
    
    def resolve_entities(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Resolve duplicate entities using semantic similarity.
        
        Args:
            entities: Dictionary mapping entity types to lists of entities
            
        Returns:
            Dictionary with resolved entities (duplicates merged)
        """
        self._load_model()
        
        resolved_entities = {}
        total_before = 0
        total_after = 0
        
        logger.info("🔍 Starting entity resolution using semantic embeddings...")
        
        for entity_type, entity_list in entities.items():
            total_before += len(entity_list)
            
            if len(entity_list) < 2:
                # No duplicates possible
                resolved_entities[entity_type] = entity_list
                total_after += len(entity_list)
                continue
            
            logger.info(f"Resolving {len(entity_list)} {entity_type} entities...")
            
            try:
                # Create text representations for embeddings
                entity_texts = []
                for entity in entity_list:
                    text = self._create_entity_text(entity, entity_type)
                    entity_texts.append(text)
                
                # Generate embeddings
                embeddings = self.model.encode(entity_texts, show_progress_bar=False)
                
                # Cluster similar entities
                cluster_labels = self._cluster_entities(embeddings, entity_type)
                
                # Group entities by cluster
                clusters = defaultdict(list)
                for idx, label in enumerate(cluster_labels):
                    clusters[label].append((idx, entity_list[idx]))
                
                # Select canonical entity from each cluster
                canonical_entities = []
                duplicates_removed = 0
                
                for cluster_entities in clusters.values():
                    canonical_entity = self._select_canonical_entity(cluster_entities)
                    canonical_entities.append(canonical_entity)
                    
                    if len(cluster_entities) > 1:
                        duplicates_removed += len(cluster_entities) - 1
                
                resolved_entities[entity_type] = canonical_entities
                total_after += len(canonical_entities)
                
                logger.info(f"✅ {entity_type}: {len(entity_list)} → {len(canonical_entities)} entities "
                          f"({duplicates_removed} duplicates removed)")
                
            except Exception as e:
                logger.error(f"Failed to resolve {entity_type} entities: {e}")
                # Fall back to original entities if resolution fails
                resolved_entities[entity_type] = entity_list
                total_after += len(entity_list)
        
        reduction_pct = ((total_before - total_after) / total_before * 100) if total_before > 0 else 0
        logger.info(f"🎯 Entity resolution complete: {total_before} → {total_after} entities "
                   f"({reduction_pct:.1f}% reduction)")
        
        return resolved_entities
    
    def get_resolution_stats(self, original_entities: Dict[str, List[Dict]], 
                           resolved_entities: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Generate statistics about the resolution process.
        
        Args:
            original_entities: Original entities before resolution
            resolved_entities: Entities after resolution
            
        Returns:
            Dictionary with resolution statistics
        """
        stats = {
            'total_before': sum(len(entities) for entities in original_entities.values()),
            'total_after': sum(len(entities) for entities in resolved_entities.values()),
            'by_type': {}
        }
        
        for entity_type in original_entities.keys():
            before = len(original_entities.get(entity_type, []))
            after = len(resolved_entities.get(entity_type, []))
            reduction = before - after
            reduction_pct = (reduction / before * 100) if before > 0 else 0
            
            stats['by_type'][entity_type] = {
                'before': before,
                'after': after,
                'duplicates_removed': reduction,
                'reduction_percentage': reduction_pct
            }
        
        stats['overall_reduction'] = stats['total_before'] - stats['total_after']
        stats['overall_reduction_percentage'] = (stats['overall_reduction'] / stats['total_before'] * 100) if stats['total_before'] > 0 else 0
        
        return stats


def resolve_knowledge_graph_entities(entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to resolve entities using default settings.
    
    Args:
        entities: Dictionary mapping entity types to lists of entities
        
    Returns:
        Dictionary with resolved entities
    """
    resolver = EntityResolver()
    return resolver.resolve_entities(entities)

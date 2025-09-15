#!/usr/bin/env python3
"""
Legal Coreference Resolution Module

This module handles legal document cross-references by:
1. Extracting legal keyword definitions from documents
2. Creating keyword nodes in the knowledge graph
3. Preprocessing text for better entity embedding
4. Establishing keyword-entity relationships

Supports both preprocessing enhancement and graph-based keyword representation.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

from app.core.logging import logger


class LegalCoreferenceResolver:
    """
    Resolves legal document cross-references and keyword mappings.
    
    Implements hybrid approach:
    - Strategy 1: Preprocessing for better embeddings
    - Strategy 2: Graph nodes for legal keyword relationships
    """
    
    def __init__(self):
        """Initialize the legal coreference resolver"""
        
        # Comprehensive legal keyword patterns
        self.legal_patterns = [
            # GROUP 1: Standard parenthetical references
            # Entity Name ("KEYWORD") or Entity Name (the "KEYWORD")
            r'([^"(]+?)\s*\("([^"]+)"\)',
            r'([^"(]+?)\s*\(the\s+"([^"]+)"\)',
            
            # GROUP 2: Formal quoted definitions  
            # "Term" shall mean... or "Term" means...
            r'"([^"]+)"\s+(?:shall\s+)?(?:mean|means|refer|refers|include|includes)\s+(.{1,100}?)(?:\.|;|,)',
            
            # GROUP 3: Unquoted definition patterns
            # Term shall mean... or Term means... (capitalize first word)
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:shall\s+)?(?:mean|means)\s+(.{1,100}?)(?:\.|;|,)',
            
            # Term includes... or Term refers to...
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:includes?|refers?\s+to)\s+(.{1,100}?)(?:\.|;|,)',
            
            # GROUP 4: Contextual definition patterns
            # As used herein, Term means... or For purposes of this Agreement, Term means...
            r'(?:As\s+used\s+herein|For\s+purposes?\s+of\s+this\s+\w+),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:means?|refers?\s+to)\s+(.{1,100}?)(?:\.|;|,)',
            
            # GROUP 5: Corporate structure patterns
            # Entity, a Delaware corporation
            r'([^,]+),\s*a\s+([A-Z][a-z]+\s+(?:corporation|company|LLC|partnership))',
            
            # GROUP 6: Agreement/document references
            # THIS AGREEMENT ("Agreement")
            r'THIS\s+([A-Z\s]+)\s*\((?:the\s+)?"([^"]+)"\)',
            
            # GROUP 7: Party relationship patterns
            # between Company and Client
            r'between\s+([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)',
            
            # GROUP 8: Section reference definitions
            # Term (as defined in Section X.Y)
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(as\s+defined\s+in\s+Section\s+[\d.]+\)',
            
            # GROUP 9: Capitalized term patterns (common in legal docs)
            # When capitalized terms are used consistently
            r'the\s+([A-Z][A-Z\s]{2,})\s+(?:means?|refers?\s+to|includes?)\s+(.{1,100}?)(?:\.|;|,)',
        ]
        
        # Keywords that commonly refer to entities
        self.entity_keywords = {
            # Core business entities
            'company', 'corporation', 'employer', 'client', 'customer', 
            'vendor', 'supplier', 'contractor', 'provider', 'licensee', 
            'licensor', 'buyer', 'seller', 'borrower', 'lender',
            
            # Organizational entities  
            'subsidiary', 'affiliate', 'parent', 'holding company',
            'joint venture', 'partnership', 'entity', 'organization',
            
            # People/roles
            'employee', 'team member', 'staff', 'personnel', 'worker',
            'officer', 'director', 'manager', 'executive', 'representative',
            'agent', 'consultant', 'advisor', 'member',
            
            # Legal parties
            'party', 'parties', 'counterparty', 'participant', 'stakeholder',
            'beneficiary', 'trustee', 'assignee', 'successor'
        }
        
        # Keywords that refer to documents/agreements
        self.document_keywords = {
            'agreement', 'contract', 'terms', 'conditions', 'policy', 
            'procedure', 'guidelines', 'manual', 'document', 'exhibit',
            'schedule', 'attachment', 'addendum', 'amendment'
        }
        
    def extract_legal_definitions(self, text: str, document_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract legal keyword definitions from document text using comprehensive patterns.
        
        Args:
            text: Full document text
            document_name: Name of the document
            
        Returns:
            Dictionary mapping keywords to their definitions and metadata
        """
        definitions = {}
        
        # Extract using each pattern with enhanced logic
        for pattern_idx, pattern in enumerate(self.legal_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    # Different patterns have different group structures
                    keyword, canonical_name = self._extract_keyword_and_canonical(match, pattern_idx)
                    
                    if not keyword or not canonical_name:
                        continue
                    
                    # Clean up extracted values
                    keyword = keyword.strip().lower()
                    canonical_name = re.sub(r'\s+', ' ', canonical_name).strip()
                    canonical_name = canonical_name.rstrip('.,;:')
                    
                    # Skip if too short or generic
                    if len(canonical_name) < 3 or len(keyword) < 2:
                        continue
                    
                    # Skip common noise words
                    if keyword in {'the', 'this', 'that', 'such', 'any', 'all', 'each'}:
                        continue
                    
                    # Determine keyword type
                    keyword_type = self._classify_keyword(keyword)
                    
                    # Calculate confidence based on pattern type and context
                    confidence = self._calculate_definition_confidence(match.group(0), pattern_idx)
                    
                    # Store definition (prefer higher confidence if duplicate)
                    if keyword not in definitions or definitions[keyword]['confidence'] < confidence:
                        definitions[keyword] = {
                            'canonical_name': canonical_name,
                            'keyword_type': keyword_type,
                            'document': document_name,
                            'context': match.group(0),
                            'confidence': confidence,
                            'pattern_type': self._get_pattern_description(pattern_idx)
                        }
        
        return definitions
    
    def _extract_keyword_and_canonical(self, match, pattern_idx: int) -> tuple:
        """
        Extract keyword and canonical name based on pattern type.
        Different patterns have different group arrangements.
        """
        groups = match.groups()
        
        # GROUP 1-2: Standard parenthetical and quoted definitions
        if pattern_idx in [0, 1, 2]:  # Parenthetical and quoted patterns
            if len(groups) >= 2:
                return groups[1], groups[0]  # keyword, canonical_name
        
        # GROUP 3-4: Unquoted definition patterns  
        elif pattern_idx in [3, 4, 5]:  # "Term means...", "Term includes..."
            if len(groups) >= 2:
                return groups[0], groups[1]  # keyword, canonical_name
        
        # GROUP 5: Corporate patterns
        elif pattern_idx == 6:  # "Entity, a Delaware corporation"
            if len(groups) >= 2:
                return groups[1].lower(), groups[0]  # "corporation", "Entity"
        
        # GROUP 6: Agreement patterns  
        elif pattern_idx == 7:  # "THIS AGREEMENT (Agreement)"
            if len(groups) >= 2:
                return groups[1], groups[0]  # "agreement", "THIS AGREEMENT"
        
        # GROUP 7: Party patterns
        elif pattern_idx == 8:  # "between Company and Client"
            if len(groups) >= 2:
                # Create two definitions
                return groups[0].lower(), groups[0]  # First party
                # Note: This pattern needs special handling for multiple parties
        
        # GROUP 8: Section reference patterns
        elif pattern_idx == 9:  # "Term (as defined in Section X.Y)"
            if len(groups) >= 1:
                return groups[0].lower(), groups[0]  # Self-reference
        
        # GROUP 9: Capitalized term patterns
        elif pattern_idx == 10:  # "the TERM means..."
            if len(groups) >= 2:
                return groups[0].lower(), groups[1]  # keyword, definition
        
        return None, None
    
    def _get_pattern_description(self, pattern_idx: int) -> str:
        """Get human-readable description of pattern type"""
        descriptions = [
            "parenthetical_reference",      # 0-1
            "parenthetical_reference",      
            "quoted_definition",            # 2
            "unquoted_definition",          # 3-4
            "unquoted_definition",
            "contextual_definition",        # 5
            "corporate_structure",          # 6
            "document_reference",           # 7
            "party_reference",              # 8
            "section_reference",            # 9
            "capitalized_term"              # 10
        ]
        return descriptions[min(pattern_idx, len(descriptions) - 1)]
    
    def _classify_keyword(self, keyword: str) -> str:
        """Classify keyword as entity, document, or other"""
        keyword_lower = keyword.lower()
        
        if keyword_lower in self.entity_keywords:
            return 'entity'
        elif keyword_lower in self.document_keywords:
            return 'document'
        elif keyword_lower in {'party', 'parties'}:
            return 'entity'
        else:
            return 'other'
    
    def _calculate_definition_confidence(self, context: str, pattern_idx: int = 0) -> float:
        """Calculate confidence score for a legal definition based on pattern type and context"""
        
        # Base confidence by pattern type (more specific patterns = higher confidence)
        pattern_confidence = {
            0: 0.95,  # parenthetical_reference - very reliable
            1: 0.95,  # parenthetical_reference  
            2: 0.90,  # quoted_definition - formal legal language
            3: 0.80,  # unquoted_definition - less formal but still clear
            4: 0.80,  # unquoted_definition
            5: 0.85,  # contextual_definition - explicit context
            6: 0.85,  # corporate_structure - standard legal pattern
            7: 0.90,  # document_reference - formal document pattern
            8: 0.75,  # party_reference - can be ambiguous
            9: 0.70,  # section_reference - cross-reference, less direct
            10: 0.75, # capitalized_term - formatting convention
        }
        
        confidence = pattern_confidence.get(pattern_idx, 0.70)
        
        # Boost confidence for specific formal legal patterns
        context_lower = context.lower()
        
        if re.search(r'shall\s+mean', context_lower):
            confidence += 0.10
        if re.search(r'for\s+purposes?\s+of\s+this', context_lower):
            confidence += 0.08
        if re.search(r'as\s+used\s+herein', context_lower):
            confidence += 0.08
        if re.search(r'this\s+\w+\s*\(', context_lower):
            confidence += 0.05
        if re.search(r'a\s+\w+\s+corporation', context_lower):
            confidence += 0.05
        
        # Reduce confidence for potential noise patterns
        if len(context) > 200:  # Very long matches might be noisy
            confidence -= 0.05
        if re.search(r'\b(?:and|or|but|however|therefore)\b', context_lower):
            confidence -= 0.02  # Complex sentences might be less precise
        
        return min(confidence, 1.0)
    
    def preprocess_text_with_replacements(self, text: str, definitions: Dict[str, Dict]) -> str:
        """
        Strategy 1: Replace keywords with canonical names for better embeddings.
        
        Args:
            text: Original text
            definitions: Keyword definitions from extract_legal_definitions
            
        Returns:
            Text with keywords replaced by canonical names
        """
        processed_text = text
        
        # Sort by keyword length (longest first) to avoid partial replacements
        sorted_keywords = sorted(definitions.keys(), key=len, reverse=True)
        
        for keyword in sorted_keywords:
            definition = definitions[keyword]
            canonical_name = definition['canonical_name']
            
            # Only replace entity keywords to avoid over-replacement
            if definition['keyword_type'] == 'entity':
                # Create regex pattern for whole word matching
                pattern = rf'\b{re.escape(keyword)}\b'
                processed_text = re.sub(pattern, canonical_name, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def create_keyword_entities(self, definitions: Dict[str, Dict], document_name: str) -> List[Dict[str, Any]]:
        """
        Strategy 2: Create keyword entities for the knowledge graph.
        
        Args:
            definitions: Keyword definitions
            document_name: Source document name
            
        Returns:
            List of keyword entities to add to the graph
        """
        keyword_entities = []
        
        for keyword, definition in definitions.items():
            # Create keyword node
            keyword_entity = {
                'name': keyword.upper(),  # Use uppercase for legal keywords
                'type': 'legal_keyword',
                'keyword_type': definition['keyword_type'],
                'canonical_reference': definition['canonical_name'],
                'source': document_name,
                'context': definition['context'],
                'confidence': definition['confidence'],
                'extraction_method': 'legal_coreference'
            }
            
            keyword_entities.append(keyword_entity)
        
        return keyword_entities
    
    def create_keyword_relationships(self, definitions: Dict[str, Dict], document_name: str) -> List[Dict[str, Any]]:
        """
        Create relationships between keywords and their canonical entities.
        
        Args:
            definitions: Keyword definitions
            document_name: Source document name
            
        Returns:
            List of relationships to add to the graph
        """
        relationships = []
        
        for keyword, definition in definitions.items():
            # Keyword -> Document relationship
            relationships.append({
                'source_entity': keyword.upper(),
                'target_entity': document_name,
                'relationship_type': 'defined_in',
                'source_document': document_name,
                'context': f'Keyword "{keyword}" defined in {document_name}',
                'confidence': definition['confidence']
            })
            
            # Keyword -> Canonical Entity relationship
            if definition['keyword_type'] == 'entity':
                relationships.append({
                    'source_entity': keyword.upper(),
                    'target_entity': definition['canonical_name'],
                    'relationship_type': 'refers_to',
                    'source_document': document_name,
                    'context': definition['context'],
                    'confidence': definition['confidence']
                })
        
        return relationships
    
    def process_document_chunks(self, chunks: List[Dict[str, Any]], use_preprocessing: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Process document chunks with legal coreference resolution.
        
        Args:
            chunks: Document chunks to process
            use_preprocessing: Whether to apply Strategy 1 (text replacement)
            
        Returns:
            Tuple of (processed_chunks, all_definitions)
        """
        processed_chunks = []
        all_definitions = {}
        
        # Group chunks by document
        chunks_by_doc = defaultdict(list)
        for chunk in chunks:
            doc_name = chunk.get('source', 'unknown')
            chunks_by_doc[doc_name].append(chunk)
        
        # Process each document
        for doc_name, doc_chunks in chunks_by_doc.items():
            logger.info(f"Processing legal coreferences for {doc_name}")
            
            # Combine all chunks for definition extraction
            full_text = ' '.join([chunk.get('text', '') for chunk in doc_chunks])
            
            # Extract legal definitions
            definitions = self.extract_legal_definitions(full_text, doc_name)
            all_definitions[doc_name] = definitions
            
            if definitions:
                logger.info(f"Found {len(definitions)} legal definitions in {doc_name}: {list(definitions.keys())}")
            
            # Process chunks
            for chunk in doc_chunks:
                processed_chunk = chunk.copy()
                
                if use_preprocessing and definitions:
                    # Strategy 1: Replace keywords in chunk text
                    original_text = chunk.get('text', '')
                    processed_text = self.preprocess_text_with_replacements(original_text, definitions)
                    processed_chunk['text'] = processed_text
                    processed_chunk['legal_preprocessing_applied'] = True
                
                processed_chunks.append(processed_chunk)
        
        return processed_chunks, all_definitions
    
    def enhance_entities_with_keywords(self, entities: Dict[str, List[Dict]], all_definitions: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """
        Add keyword entities to the entity collection.
        
        Args:
            entities: Existing entities
            all_definitions: Legal definitions by document
            
        Returns:
            Enhanced entities including keyword entities
        """
        enhanced_entities = entities.copy()
        
        # Add legal_keywords as a new entity type
        enhanced_entities['legal_keywords'] = []
        
        for doc_name, definitions in all_definitions.items():
            keyword_entities = self.create_keyword_entities(definitions, doc_name)
            enhanced_entities['legal_keywords'].extend(keyword_entities)
        
        logger.info(f"Added {len(enhanced_entities['legal_keywords'])} legal keyword entities")
        
        return enhanced_entities
    
    def create_all_keyword_relationships(self, all_definitions: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Create all keyword relationships from definitions.
        
        Args:
            all_definitions: Legal definitions by document
            
        Returns:
            List of all keyword relationships
        """
        all_relationships = []
        
        for doc_name, definitions in all_definitions.items():
            relationships = self.create_keyword_relationships(definitions, doc_name)
            all_relationships.extend(relationships)
        
        logger.info(f"Created {len(all_relationships)} keyword relationships")
        
        return all_relationships


def enhance_chunks_with_legal_coreference(chunks: List[Dict[str, Any]], 
                                        use_preprocessing: bool = True) -> Tuple[List[Dict], Dict]:
    """
    Convenience function to enhance chunks with legal coreference resolution.
    
    Args:
        chunks: Document chunks
        use_preprocessing: Whether to apply text preprocessing
        
    Returns:
        Tuple of (enhanced_chunks, legal_definitions)
    """
    resolver = LegalCoreferenceResolver()
    return resolver.process_document_chunks(chunks, use_preprocessing)


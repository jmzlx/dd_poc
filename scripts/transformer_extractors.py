#!/usr/bin/env python3
"""
Transformer-based Entity and Relationship Extraction

Simplified, clean implementation using Hugging Face transformers
for entity and relationship extraction.
"""

import re
import warnings
from typing import Dict, List, Any, Optional, Set
from tqdm import tqdm

# Suppress tokenizer warnings
warnings.filterwarnings("ignore", message=".*token_type_ids.*")
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*")

from transformers import pipeline
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

from app.core.logging import logger


class TransformerEntityExtractor:
    """Clean transformer-based entity extraction"""
    
    def __init__(self):
        self.models_loaded = False
        self.ner_pipeline = None
        self._load_models()
        
        # Simple financial patterns (only what transformers can't handle)
        self.financial_patterns = [
            r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|M|B|K))?',
            r'(?:revenue|profit|loss|EBITDA|earnings)\s*of\s*\$[\d,]+'
        ]
    
    def _load_models(self):
        """Load transformer models"""
        logger.info("Loading transformer models for entity extraction...")
        self.ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple",
            device=-1
        )
        self.models_loaded = True
        logger.info("✅ Transformer models loaded successfully")
    
    def extract_entities(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities from document chunks"""
        entities = {
            'companies': [],
            'people': [],
            'financial_metrics': [],
            'documents': []
        }
        
        if not self.models_loaded:
            raise RuntimeError("Transformer models failed to load")
        
        logger.info(f"Extracting entities using transformers from {len(chunks)} chunks")
        
        # Track unique documents
        seen_documents = set()
        
        for chunk in tqdm(chunks, desc="Transformer entity extraction"):
            text = chunk.get('text', '')
            source = chunk.get('source', 'unknown')
            metadata = chunk.get('metadata', {})
            
            # Create document entity (one per unique document)
            if source not in seen_documents and source != 'unknown':
                seen_documents.add(source)
                doc_name = source.split('/')[-1].replace('.pdf', '').replace('_', ' ')
                entities['documents'].append({
                    'name': doc_name,
                    'source': source,
                    'context': text[:200],
                    'confidence': 1.0,
                    'extraction_method': 'document_metadata'
                })
            
            if len(text.strip()) < 10:
                continue
            
            # Truncate very long text
            if len(text) > 2000:
                text = text[:2000]
            
            # Extract entities using NER
            ner_results = self.ner_pipeline(text)
            
            for entity in ner_results:
                entity_text = entity['word'].strip()
                entity_type = entity['entity_group']
                confidence = float(entity['score'])
                
                if confidence < 0.7:
                    continue
                
                entity_data = {
                    'name': entity_text,
                    'source': source,
                    'context': self._get_context(text, entity_text),
                    'confidence': confidence,
                    'extraction_method': 'transformer'
                }
                
                # Categorize entities with simple validation
                if entity_type == 'ORG' and self._is_valid_company(entity_text):
                    entities['companies'].append(entity_data)
                elif entity_type == 'PER' and self._is_valid_person(entity_text):
                    entities['people'].append(entity_data)
            
            # Extract financial metrics using simple regex
            for pattern in self.financial_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities['financial_metrics'].append({
                        'name': match.group(0),
                        'source': source,
                        'context': self._get_context(text, match.group(0)),
                        'confidence': 0.9,
                        'extraction_method': 'regex'
                    })
        
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        logger.info(f"Extracted {total_entities} entities using transformers")
        
        return entities
    
    def _get_context(self, text: str, entity_text: str, context_size: int = 50) -> str:
        """Get context around entity"""
        start_idx = text.find(entity_text)
        if start_idx == -1:
            return text[:100]
        context_start = max(0, start_idx - context_size)
        context_end = min(len(text), start_idx + len(entity_text) + context_size)
        return text[context_start:context_end]
    
    def _is_valid_company(self, name: str) -> bool:
        """Simple company name validation"""
        name = name.strip()
        if len(name) < 3 or len(name) > 100:
            return False
        if name.isupper() and len(name) > 30:
            return False
        return any(c.isalpha() for c in name)
    
    def _is_valid_person(self, name: str) -> bool:
        """Simple person name validation"""
        name = name.strip()
        if len(name) < 3 or len(name) > 50:
            return False
        parts = name.split()
        return len(parts) >= 2 and all(part[0].isupper() for part in parts)


class TransformerRelationshipExtractor:
    """Simple relationship extraction without complex matching"""
    
    def __init__(self):
        # Simple relationship patterns
        self.relationship_patterns = [
            # Corporate relationships
            (r'(\w+(?:\s+\w+)*)\s+(?:acquired|purchased|bought)\s+(\w+(?:\s+\w+)*)', 'ACQUIRED'),
            (r'(\w+(?:\s+\w+)*)\s+(?:partnered with|partnership with)\s+(\w+(?:\s+\w+)*)', 'PARTNERSHIP'),
            (r'(\w+(?:\s+\w+)*)\s+(?:invested in)\s+(\w+(?:\s+\w+)*)', 'INVESTED_IN'),
            
            # Executive relationships  
            (r'(\w+(?:\s+\w+)*)\s+(?:is the |is |serves as )?(?:CEO|CFO|CTO|President|Director)\s+(?:of |at )?(\w+(?:\s+\w+)*)', 'EXECUTIVE_OF'),
            (r'(\w+(?:\s+\w+)*)\s+(?:founded|established|created)\s+(\w+(?:\s+\w+)*)', 'FOUNDED'),
            
            # Ownership relationships
            (r'(\w+(?:\s+\w+)*)\s+(?:owns|controls)\s+(\w+(?:\s+\w+)*)', 'OWNS'),
            (r'(\w+(?:\s+\w+)*)\s+(?:subsidiary of|owned by)\s+(\w+(?:\s+\w+)*)', 'SUBSIDIARY_OF'),
        ]
    
    def extract_relationships(self, entities: Dict[str, List[Dict]], chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Extract relationships using simple pattern matching only"""
        relationships = []
        
        logger.info(f"Extracting relationships using simple pattern matching from {len(chunks)} chunks")
        
        # Process only a sample of chunks to avoid memory issues
        sample_size = min(500, len(chunks))  # Process max 500 chunks
        sample_chunks = chunks[:sample_size]
        
        for chunk in tqdm(sample_chunks, desc="Extracting relationships"):
            text = chunk.get('text', '')
            source = chunk.get('source', 'unknown')
            
            if len(text.strip()) < 50:
                continue
            
            # Apply simple relationship patterns
            for pattern, relationship_type in self.relationship_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        entity1 = match.group(1).strip()
                        entity2 = match.group(2).strip()
                        
                        # Clean entity names
                        entity1 = self._clean_entity_name(entity1)
                        entity2 = self._clean_entity_name(entity2)
                        
                        if (entity1 and entity2 and entity1 != entity2 and 
                            len(entity1) > 2 and len(entity2) > 2):
                            
                            relationships.append({
                                'source_entity': entity1,
                                'target_entity': entity2,
                                'relationship_type': relationship_type,
                                'source_document': source,
                                'context': text[max(0, match.start()-50):match.end()+50],
                                'confidence': 0.7,
                                'extraction_method': 'pattern_matching'
                            })
                    except (IndexError, AttributeError):
                        continue
        
        # Removed: Basic co-occurrence relationships
        # These created noise with low confidence (0.5) and no semantic value
        
        # Remove duplicates
        relationships = self._deduplicate_relationships(relationships)
        
        logger.info(f"Extracted {len(relationships)} relationships")
        return relationships
    
    
    def _clean_entity_name(self, name: str) -> str:
        """Clean entity names"""
        if not name:
            return ""
        
        name = name.strip()
        
        # Remove common prefixes
        for prefix in ['the ', 'a ', 'an ', 'by ']:
            if name.lower().startswith(prefix):
                name = name[len(prefix):]
                break
        
        # Truncate at common endings
        for ending in [' and ', ' or ', ',', ';']:
            if ending in name.lower():
                name = name[:name.lower().find(ending)]
                break
        
        return name.strip()
    
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships"""
        seen = set()
        deduplicated = []
        
        for rel in relationships:
            key = (
                rel['source_entity'].lower(),
                rel['target_entity'].lower(),
                rel['relationship_type']
            )
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(rel)
        
        return deduplicated

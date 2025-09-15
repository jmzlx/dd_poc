#!/usr/bin/env python3
"""
Enhanced Entity Extractor for Multi-Column Splink Normalization

This module extracts rich, multi-attribute entity data that leverages
Splink's multi-column comparison capabilities for superior entity resolution.

For each entity type, we extract multiple independent attributes:
- Companies: name, industry, revenue, location, employees, legal_form
- People: first_name, last_name, title, department, company, email_domain
- Financial: amount, currency, metric_type, period, context_type
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from app.core.logging import logger


@dataclass
class RichEntity:
    """Rich entity with multiple attributes for Splink matching"""
    entity_type: str
    primary_name: str
    attributes: Dict[str, Any]
    source: str
    context: str
    confidence: float
    extraction_method: str


class EnhancedEntityExtractor:
    """
    Extract rich, multi-column entity data optimized for Splink
    """
    
    def __init__(self):
        # Patterns for extracting additional attributes
        self.company_patterns = {
            'industry': [
                r'(?:industry|sector|business):\s*([^.\n]+)',
                r'(?:specializes? in|focuses on)\s+([^.\n]+)',
                r'(?:provider of|leader in)\s+([^.\n]+)'
            ],
            'revenue': [
                r'(?:revenue|sales|income).*?\$([0-9.,]+(?:\s*(?:million|billion|M|B))?)',
                r'\$([0-9.,]+(?:\s*(?:million|billion|M|B))?).*?(?:revenue|annual|yearly)'
            ],
            'employees': [
                r'(?:employees?|staff|workforce).*?([0-9,]+(?:-[0-9,]+)?)',
                r'([0-9,]+(?:-[0-9,]+)?)\s+(?:employees?|staff|people)'
            ],
            'location': [
                r'(?:headquartered|located|based)\s+in\s+([^.\n,]+)',
                r'(?:state|jurisdiction):\s*([A-Z][a-z]+)',
                r'([A-Z][a-z]+)\s+(?:corporation|corp|inc)'
            ],
            'legal_form': [
                r'\b(Inc\.?|Corporation|Corp\.?|LLC|Ltd\.?|Limited)\b',
                r'\b(Delaware|Nevada|California)\s+(corporation|corp)\b'
            ]
        }
        
        self.person_patterns = {
            'title': [
                r'\b(CEO|CTO|CFO|COO|President|Director|Manager|VP|Vice President)\b',
                r'\b(Chief\s+\w+\s+Officer)\b',
                r'\b(Senior|Principal|Lead)\s+\w+'
            ],
            'department': [
                r'\b(Human Resources?|HR|Engineering|Finance|Legal|Marketing|Sales|Operations)\b',
                r'\b(IT|Information Technology|Security|Compliance)\b'
            ],
            'email_domain': [
                r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                r'([a-zA-Z0-9.-]+\.com|\.org|\.net)'
            ]
        }
        
        self.financial_patterns = {
            'currency': [r'\$', r'USD', r'EUR', r'GBP'],
            'metric_type': [
                r'\b(revenue|profit|loss|EBITDA|earnings|income|sales)\b',
                r'\b(assets|liabilities|equity|debt)\b'
            ],
            'period': [
                r'\b(annual|yearly|quarterly|monthly|FY\d{4}|Q[1-4])\b',
                r'\b(2024|2023|2022|2021|2020)\b'
            ]
        }
    
    def extract_rich_entities(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract rich, multi-column entities optimized for Splink
        
        Args:
            chunks: Document chunks with text, source, metadata
            
        Returns:
            Dictionary of entity types to rich entity lists
        """
        logger.info("Extracting rich multi-column entities for Splink...")
        
        rich_entities = {
            'companies': [],
            'people': [],
            'financial_metrics': []
        }
        
        for chunk in chunks:
            text = chunk.get('text', '')
            source = chunk.get('source', 'unknown')
            
            if len(text.strip()) < 20:
                continue
            
            # Extract rich company entities
            company_entities = self._extract_rich_companies(text, source)
            rich_entities['companies'].extend(company_entities)
            
            # Extract rich person entities
            person_entities = self._extract_rich_people(text, source)
            rich_entities['people'].extend(person_entities)
            
            # Extract rich financial entities
            financial_entities = self._extract_rich_financials(text, source)
            rich_entities['financial_metrics'].extend(financial_entities)
        
        # Log extraction results
        for entity_type, entity_list in rich_entities.items():
            logger.info(f"Extracted {len(entity_list)} rich {entity_type} entities")
        
        return rich_entities
    
    def _extract_rich_companies(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Extract companies with multiple attributes"""
        
        companies = []
        
        # Find company name mentions
        company_patterns = [
            r'\b([A-Z][a-zA-Z\s&]+(?:Inc\.?|Corp\.?|LLC|Ltd\.?|Corporation|Company|Co\.?))\b',
            r'\b([A-Z][a-zA-Z\s&]+(?:Systems?|Solutions?|Services?|Technologies?))\b'
        ]
        
        for pattern in company_patterns:
            for match in re.finditer(pattern, text):
                company_name = match.group(1).strip()
                
                if len(company_name) < 5 or len(company_name) > 80:
                    continue
                
                # Extract additional attributes from surrounding context
                context_window = text[max(0, match.start()-200):match.end()+200]
                
                attributes = {
                    'name': company_name,
                    'industry': self._extract_attribute(context_window, self.company_patterns['industry']),
                    'revenue': self._extract_attribute(context_window, self.company_patterns['revenue']),
                    'employees': self._extract_attribute(context_window, self.company_patterns['employees']),
                    'location': self._extract_attribute(context_window, self.company_patterns['location']),
                    'legal_form': self._extract_attribute(context_window, self.company_patterns['legal_form']),
                    'source_document': source.split('/')[-1],
                    'context_length': len(context_window),
                    'mention_position': match.start() / len(text)  # Relative position in document
                }
                
                companies.append({
                    'name': company_name,
                    'source': source,
                    'context': context_window[:200],
                    'confidence': 0.9,
                    'extraction_method': 'enhanced_regex',
                    'rich_attributes': attributes
                })
        
        return companies
    
    def _extract_rich_people(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Extract people with multiple attributes"""
        
        people = []
        
        # Find person name patterns
        person_patterns = [
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',  # John Smith, Mary Jane Doe
            r'\b(?:Dr\.?|Mr\.?|Ms\.?|Mrs\.?)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b'  # Dr. John Smith
        ]
        
        for pattern in person_patterns:
            for match in re.finditer(pattern, text):
                person_name = match.group(1).strip()
                
                if len(person_name.split()) < 2:  # Need at least first + last name
                    continue
                
                # Extract additional attributes
                context_window = text[max(0, match.start()-200):match.end()+200]
                name_parts = person_name.split()
                
                attributes = {
                    'full_name': person_name,
                    'first_name': name_parts[0],
                    'last_name': name_parts[-1],
                    'middle_name': ' '.join(name_parts[1:-1]) if len(name_parts) > 2 else '',
                    'title': self._extract_attribute(context_window, self.person_patterns['title']),
                    'department': self._extract_attribute(context_window, self.person_patterns['department']),
                    'email_domain': self._extract_attribute(context_window, self.person_patterns['email_domain']),
                    'source_document': source.split('/')[-1],
                    'context_length': len(context_window),
                    'name_length': len(person_name)
                }
                
                people.append({
                    'name': person_name,
                    'source': source,
                    'context': context_window[:200],
                    'confidence': 0.85,
                    'extraction_method': 'enhanced_regex',
                    'rich_attributes': attributes
                })
        
        return people
    
    def _extract_rich_financials(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Extract financial metrics with multiple attributes"""
        
        financials = []
        
        # Financial patterns
        financial_patterns = [
            r'\$([0-9,]+(?:\.[0-9]+)?(?:\s*(?:million|billion|thousand|M|B|K))?)',
            r'([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand|M|B|K)?\s*(?:dollars?|USD|\$)'
        ]
        
        for pattern in financial_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                amount_text = match.group(1) if match.group(1) else match.group(0)
                
                # Extract additional attributes
                context_window = text[max(0, match.start()-200):match.end()+200]
                
                attributes = {
                    'amount_text': amount_text,
                    'normalized_amount': self._normalize_amount(amount_text),
                    'currency': self._extract_attribute(context_window, self.financial_patterns['currency']) or 'USD',
                    'metric_type': self._extract_attribute(context_window, self.financial_patterns['metric_type']) or 'unknown',
                    'period': self._extract_attribute(context_window, self.financial_patterns['period']) or 'unknown',
                    'source_document': source.split('/')[-1],
                    'context_length': len(context_window),
                    'position_in_doc': match.start() / len(text)
                }
                
                financials.append({
                    'name': amount_text,
                    'source': source,
                    'context': context_window[:200],
                    'confidence': 0.9,
                    'extraction_method': 'enhanced_regex',
                    'rich_attributes': attributes
                })
        
        return financials
    
    def _extract_attribute(self, text: str, patterns: List[str]) -> Optional[str]:
        """Extract attribute value using regex patterns"""
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip() if match.groups() else match.group(0).strip()
        
        return None
    
    def _normalize_amount(self, amount_text: str) -> float:
        """Convert amount text to normalized float value"""
        
        # Remove commas and extract number
        amount_str = re.sub(r'[,$]', '', amount_text)
        
        # Handle multipliers
        multiplier = 1
        if re.search(r'\b(?:billion|B)\b', amount_text, re.IGNORECASE):
            multiplier = 1_000_000_000
        elif re.search(r'\b(?:million|M)\b', amount_text, re.IGNORECASE):
            multiplier = 1_000_000
        elif re.search(r'\b(?:thousand|K)\b', amount_text, re.IGNORECASE):
            multiplier = 1_000
        
        # Extract numeric value
        number_match = re.search(r'([0-9]+(?:\.[0-9]+)?)', amount_str)
        if number_match:
            return float(number_match.group(1)) * multiplier
        
        return 0.0


def convert_to_splink_format(rich_entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert rich entities to Splink-optimized multi-column format
    
    Args:
        rich_entities: Entities with rich_attributes
        
    Returns:
        Entities in multi-column format for Splink
    """
    splink_entities = {}
    
    for entity_type, entity_list in rich_entities.items():
        splink_list = []
        
        for entity in entity_list:
            rich_attrs = entity.get('rich_attributes', {})
            
            if entity_type == 'companies':
                splink_entity = {
                    # Core identification columns
                    'name': rich_attrs.get('name', entity.get('name', '')),
                    'industry': rich_attrs.get('industry', ''),
                    'legal_form': rich_attrs.get('legal_form', ''),
                    'location': rich_attrs.get('location', ''),
                    
                    # Numeric attributes
                    'revenue_text': rich_attrs.get('revenue', ''),
                    'employees_text': rich_attrs.get('employees', ''),
                    
                    # Document context
                    'source_document': rich_attrs.get('source_document', ''),
                    'context_length': rich_attrs.get('context_length', 0),
                    'mention_position': rich_attrs.get('mention_position', 0.0),
                    
                    # Original metadata
                    'source': entity.get('source', ''),
                    'context': entity.get('context', ''),
                    'confidence': entity.get('confidence', 0.0),
                    'extraction_method': entity.get('extraction_method', '')
                }
                
            elif entity_type == 'people':
                splink_entity = {
                    # Core identification columns
                    'full_name': rich_attrs.get('full_name', entity.get('name', '')),
                    'first_name': rich_attrs.get('first_name', ''),
                    'last_name': rich_attrs.get('last_name', ''),
                    'middle_name': rich_attrs.get('middle_name', ''),
                    
                    # Professional attributes
                    'title': rich_attrs.get('title', ''),
                    'department': rich_attrs.get('department', ''),
                    'email_domain': rich_attrs.get('email_domain', ''),
                    
                    # Document context
                    'source_document': rich_attrs.get('source_document', ''),
                    'name_length': rich_attrs.get('name_length', 0),
                    
                    # Original metadata
                    'source': entity.get('source', ''),
                    'context': entity.get('context', ''),
                    'confidence': entity.get('confidence', 0.0),
                    'extraction_method': entity.get('extraction_method', '')
                }
                
            elif entity_type == 'financial_metrics':
                splink_entity = {
                    # Core identification columns
                    'amount_text': rich_attrs.get('amount_text', entity.get('name', '')),
                    'normalized_amount': rich_attrs.get('normalized_amount', 0.0),
                    'currency': rich_attrs.get('currency', 'USD'),
                    'metric_type': rich_attrs.get('metric_type', 'unknown'),
                    'period': rich_attrs.get('period', 'unknown'),
                    
                    # Document context
                    'source_document': rich_attrs.get('source_document', ''),
                    'position_in_doc': rich_attrs.get('position_in_doc', 0.0),
                    
                    # Original metadata
                    'source': entity.get('source', ''),
                    'context': entity.get('context', ''),
                    'confidence': entity.get('confidence', 0.0),
                    'extraction_method': entity.get('extraction_method', '')
                }
            
            else:
                # Fallback for other entity types
                splink_entity = entity.copy()
            
            splink_list.append(splink_entity)
        
        splink_entities[entity_type] = splink_list
    
    return splink_entities


def enhance_existing_entities(entities: Dict[str, List[Dict[str, Any]]], chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Enhance existing entities with additional attributes by re-analyzing their source contexts
    
    Args:
        entities: Existing entities from transformer extraction
        chunks: Original document chunks
        
    Returns:
        Enhanced entities with rich attributes
    """
    logger.info("Enhancing existing entities with additional attributes...")
    
    # Create context lookup by source
    source_contexts = {}
    for chunk in chunks:
        source = chunk.get('source', 'unknown')
        if source not in source_contexts:
            source_contexts[source] = []
        source_contexts[source].append(chunk.get('text', ''))
    
    enhancer = EnhancedEntityExtractor()
    enhanced_entities = {}
    
    for entity_type, entity_list in entities.items():
        enhanced_list = []
        
        for entity in entity_list:
            # Get all text from the entity's source document
            source = entity.get('source', '')
            source_texts = source_contexts.get(source, [''])
            full_context = ' '.join(source_texts)
            
            # Extract additional attributes based on entity type
            if entity_type == 'companies':
                rich_attrs = enhancer._extract_company_attributes(entity.get('name', ''), full_context)
            elif entity_type == 'people':
                rich_attrs = enhancer._extract_person_attributes(entity.get('name', ''), full_context)
            elif entity_type == 'financial_metrics':
                rich_attrs = enhancer._extract_financial_attributes(entity.get('name', ''), full_context)
            else:
                rich_attrs = {}
            
            # Add rich attributes to entity
            enhanced_entity = entity.copy()
            enhanced_entity['rich_attributes'] = rich_attrs
            enhanced_list.append(enhanced_entity)
        
        enhanced_entities[entity_type] = enhanced_list
    
    return enhanced_entities
    
    def _extract_company_attributes(self, company_name: str, context: str) -> Dict[str, Any]:
        """Extract additional company attributes from context"""
        
        attributes = {'name': company_name}
        
        for attr_name, patterns in self.company_patterns.items():
            value = self._extract_attribute(context, patterns)
            attributes[attr_name] = value or ''
        
        # Add derived attributes
        attributes['source_document'] = ''  # Will be filled by caller
        attributes['context_length'] = len(context)
        
        return attributes
    
    def _extract_person_attributes(self, person_name: str, context: str) -> Dict[str, Any]:
        """Extract additional person attributes from context"""
        
        name_parts = person_name.split()
        attributes = {
            'full_name': person_name,
            'first_name': name_parts[0] if name_parts else '',
            'last_name': name_parts[-1] if len(name_parts) > 1 else '',
            'middle_name': ' '.join(name_parts[1:-1]) if len(name_parts) > 2 else ''
        }
        
        for attr_name, patterns in self.person_patterns.items():
            value = self._extract_attribute(context, patterns)
            attributes[attr_name] = value or ''
        
        attributes['name_length'] = len(person_name)
        
        return attributes
    
    def _extract_financial_attributes(self, amount_text: str, context: str) -> Dict[str, Any]:
        """Extract additional financial attributes from context"""
        
        attributes = {
            'amount_text': amount_text,
            'normalized_amount': self._normalize_amount(amount_text)
        }
        
        for attr_name, patterns in self.financial_patterns.items():
            value = self._extract_attribute(context, patterns)
            attributes[attr_name] = value or ''
        
        return attributes

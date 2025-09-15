#!/usr/bin/env python3
"""
Unit tests for transformer-based entity extraction

Tests the transformer extractors with sample text to validate functionality.
"""

import sys
from pathlib import Path

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.transformer_extractors import TransformerEntityExtractor, TransformerRelationshipExtractor


def test_entity_extraction():
    """Test entity extraction with sample business text"""
    
    # Sample business text with document signatures and parties
    sample_texts = [
        {
            'text': "ACQUISITION AGREEMENT\n\nThis Agreement is entered into between Microsoft Corporation and OpenAI LLC for the acquisition amount of $10 billion. The deal was announced by CEO Satya Nadella and will be completed by December 2024.\n\nSigned by: Satya Nadella, CEO Microsoft Corporation\nSigned by: Sam Altman, CEO OpenAI LLC",
            'source': 'acquisition_agreement_microsoft_openai.pdf',
            'metadata': {'chunk_id': 'test_chunk_1', 'document_type': 'acquisition'}
        },
        {
            'text': "PARTNERSHIP AGREEMENT\n\nParties: TechCorp Inc. and DataSolutions Ltd.\nJohn Smith, CEO of TechCorp Inc., announced a partnership with DataSolutions Ltd. The agreement includes a $50 million investment.\n\nExecuted by: John Smith, TechCorp Inc.\nWitnessed by: Legal Counsel",
            'source': 'partnership_agreement_techcorp.pdf',
            'metadata': {'chunk_id': 'test_chunk_2', 'document_type': 'partnership'}
        },
        {
            'text': "FINANCIAL STATEMENT Q3 2024\n\nDeepShield Systems, Inc. reported revenue of $25.5 million for Q3 2024. Sarah Martinez, the Chief Financial Officer, will present the results.\n\nPrepared by: Sarah Martinez, CFO\nReviewed by: Board of Directors",
            'source': 'financial_statement_q3_2024.pdf',
            'metadata': {'chunk_id': 'test_chunk_3', 'document_type': 'financial'}
        }
    ]
    
    # Test entity extraction
    extractor = TransformerEntityExtractor()
    entities = extractor.extract_entities(sample_texts)
    
    # Assertions for pytest
    assert len(entities) > 0, "Should extract some entity types"
    assert any(entities.values()), "Should have entities in at least one category"


def test_relationship_extraction():
    """Test relationship extraction with sample entities and text"""
    
    # Sample entities (would come from entity extraction)
    sample_entities = {
        'companies': [
            {'name': 'Microsoft Corporation'},
            {'name': 'OpenAI LLC'},
            {'name': 'TechCorp Inc.'},
            {'name': 'DataSolutions Ltd.'},
            {'name': 'DeepShield Systems, Inc.'}
        ],
        'people': [
            {'name': 'Satya Nadella'},
            {'name': 'John Smith'},
            {'name': 'Sarah Martinez'},
            {'name': 'Sam Altman'}
        ],
        'financial_metrics': [
            {'name': '$10 billion'},
            {'name': '$50 million'},
            {'name': '$25.5 million'}
        ]
    }
    
    # Sample text chunks with document relationships
    sample_chunks = [
        {
            'text': "ACQUISITION AGREEMENT\n\nThis Agreement is entered into between Microsoft Corporation and OpenAI LLC for the acquisition amount of $10 billion. The deal was announced by CEO Satya Nadella.\n\nSigned by: Satya Nadella, CEO Microsoft Corporation\nSigned by: Sam Altman, CEO OpenAI LLC",
            'source': 'acquisition_agreement_microsoft_openai.pdf'
        },
        {
            'text': "PARTNERSHIP AGREEMENT\n\nParties: TechCorp Inc. and DataSolutions Ltd.\nJohn Smith, CEO of TechCorp Inc., announced a partnership with DataSolutions Ltd.\n\nExecuted by: John Smith, TechCorp Inc.",
            'source': 'partnership_agreement_techcorp.pdf'
        },
        {
            'text': "Sarah Martinez serves as Chief Financial Officer of DeepShield Systems, Inc. This document was prepared by Sarah Martinez.",
            'source': 'financial_statement_q3_2024.pdf'
        }
    ]
    
    # Test relationship extraction
    extractor = TransformerRelationshipExtractor()
    relationships = extractor.extract_relationships(sample_entities, sample_chunks)
    
    # Assertions for pytest
    assert isinstance(relationships, list), "Should return a list of relationships"


def test_all_extraction():
    """Run all extraction tests"""
    # Run individual tests
    test_entity_extraction()
    test_relationship_extraction()
    
    # Should complete without errors
    assert True


if __name__ == "__main__":
    test_all_extraction()

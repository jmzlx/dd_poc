#!/usr/bin/env python3
"""
Behavior-focused tests for entity resolution module

Tests focus on expected outcomes and public API behavior rather than
internal implementation details.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.entity_resolution import EntityResolver


class TestEntityResolverBehavior:
    """Behavior-focused tests for EntityResolver"""

    @pytest.fixture
    def mock_model(self):
        """Mock sentence transformer model"""
        model = MagicMock()
        # Mock simple embeddings for predictable clustering behavior
        model.encode.return_value = [
            [0.1, 0.2, 0.3],      # Entity 1
            [0.11, 0.21, 0.31],   # Similar to entity 1 
            [0.9, 0.8, 0.7],      # Different entity
        ]
        return model

    @pytest.fixture
    @patch('app.core.entity_resolution.SentenceTransformer')
    def resolver(self, mock_transformer_class, mock_model):
        """Create EntityResolver instance with mocked dependencies"""
        mock_transformer_class.return_value = mock_model
        return EntityResolver()

    @pytest.fixture
    def sample_entities_with_duplicates(self):
        """Sample entities that contain obvious duplicates"""
        return {
            'companies': [
                {
                    'name': 'Microsoft Corporation',
                    'source': 'doc1.pdf',
                    'context': 'Microsoft Corporation announced earnings',
                    'confidence': 0.95
                },
                {
                    'name': 'Microsoft Corp',  # Similar to above
                    'source': 'doc2.pdf', 
                    'context': 'Microsoft Corp stock price',
                    'confidence': 0.90
                },
                {
                    'name': 'Apple Inc',  # Clearly different
                    'source': 'doc3.pdf',
                    'context': 'Apple Inc released new products',
                    'confidence': 0.88
                }
            ]
        }

    def test_resolution_produces_valid_output_structure(self, resolver, sample_entities_with_duplicates):
        """Test that resolution returns properly structured data"""
        result = resolver.resolve_entities(sample_entities_with_duplicates)
        
        # Should return dictionary with same entity types
        assert isinstance(result, dict)
        assert 'companies' in result
        
        # Each entity type should map to a list
        assert isinstance(result['companies'], list)
        
        # Each resolved entity should be a dictionary
        for entity in result['companies']:
            assert isinstance(entity, dict)

    def test_resolution_reduces_or_maintains_entity_count(self, resolver, sample_entities_with_duplicates):
        """Test that resolution doesn't increase entity count (merges duplicates)"""
        original_count = len(sample_entities_with_duplicates['companies'])
        
        result = resolver.resolve_entities(sample_entities_with_duplicates)
        resolved_count = len(result['companies'])
        
        # Should not increase entity count (may merge duplicates)
        assert resolved_count <= original_count

    def test_resolution_preserves_essential_entity_information(self, resolver, sample_entities_with_duplicates):
        """Test that essential entity information is preserved after resolution"""
        result = resolver.resolve_entities(sample_entities_with_duplicates)
        
        # Each resolved entity should retain essential fields
        for entity in result['companies']:
            # Should have identification
            assert 'name' in entity
            assert isinstance(entity['name'], str)
            assert len(entity['name']) > 0
            
            # Should have source tracking
            assert 'source' in entity
            
            # Should have context
            assert 'context' in entity

    def test_handles_empty_entity_input(self, resolver):
        """Test that empty input is handled gracefully"""
        empty_entities = {'companies': [], 'people': []}
        
        result = resolver.resolve_entities(empty_entities)
        
        # Should return same structure with empty lists
        assert result == empty_entities

    def test_handles_single_entity_per_type(self, resolver):
        """Test handling when no duplicates exist"""
        single_entities = {
            'companies': [
                {
                    'name': 'Unique Company',
                    'source': 'doc.pdf',
                    'context': 'Only company mentioned',
                    'confidence': 0.9
                }
            ]
        }
        
        result = resolver.resolve_entities(single_entities)
        
        # Should return the single entity unchanged
        assert len(result['companies']) == 1
        assert result['companies'][0]['name'] == 'Unique Company'

    def test_handles_multiple_entity_types(self, resolver):
        """Test resolution across multiple entity types"""
        multi_type_entities = {
            'companies': [
                {'name': 'TechCorp', 'source': 'doc1.pdf', 'context': 'TechCorp info', 'confidence': 0.9}
            ],
            'people': [
                {'name': 'John Doe', 'source': 'doc1.pdf', 'context': 'John Doe mentioned', 'confidence': 0.8}
            ]
        }
        
        result = resolver.resolve_entities(multi_type_entities)
        
        # Should handle both entity types
        assert 'companies' in result
        assert 'people' in result
        assert len(result['companies']) == 1
        assert len(result['people']) == 1
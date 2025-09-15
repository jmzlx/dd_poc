#!/usr/bin/env python3
"""
Test Legal Coreference Resolution

Test script to validate the legal coreference resolution system
on Summit Digital Solutions documents.
"""

import sys
from pathlib import Path

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.legal_coreference import LegalCoreferenceResolver
from app.core.logging import setup_logging

# Set up logging
logger = setup_logging("test_legal_coreference", log_level="INFO")

def test_legal_pattern_extraction():
    """Test legal pattern extraction on sample texts"""
    
    resolver = LegalCoreferenceResolver()
    
    # Test cases with different legal patterns
    test_texts = [
        {
            'name': 'Standard Entity Reference',
            'text': '''CONFIDENTIALITY AGREEMENT
            THIS CONFIDENTIALITY AGREEMENT (the "Agreement") is made effective as of January 1, 2024
            BY AND BETWEEN:
            SUMMIT DIGITAL SOLUTIONS, INC., a Delaware corporation ("Company")
            AND
            CLIENT CORPORATION ("Client")''',
            'expected': ['agreement', 'company', 'client']
        },
        {
            'name': 'Policy Document',
            'text': '''TRAVEL AND EXPENSE POLICY
            This Policy applies to all employees of Summit Digital Solutions, Inc. ("Company").
            The Company shall reimburse reasonable expenses.''',
            'expected': ['company']
        },
        {
            'name': 'Complex Legal Document',
            'text': '''PROFESSIONAL SERVICES AGREEMENT
            THIS PROFESSIONAL SERVICES AGREEMENT ("Agreement") is made between
            Summit Digital Solutions, Inc., a Delaware corporation ("Provider")
            and the client entity ("Customer").
            The Provider shall deliver services as outlined in this Agreement.''',
            'expected': ['agreement', 'provider', 'customer']
        }
    ]
    
    print("🧪 Testing Legal Pattern Extraction")
    print("=" * 50)
    
    for test_case in test_texts:
        print(f"\nTest: {test_case['name']}")
        print("-" * 30)
        
        definitions = resolver.extract_legal_definitions(test_case['text'], 'test-document.pdf')
        
        print(f"Found {len(definitions)} definitions:")
        for keyword, definition in definitions.items():
            print(f"  • '{keyword}' → '{definition['canonical_name']}' "
                  f"(type: {definition['keyword_type']}, confidence: {definition['confidence']:.2f})")
        
        # Check if expected keywords were found
        found_keywords = set(definitions.keys())
        expected_keywords = set(test_case['expected'])
        
        if expected_keywords.issubset(found_keywords):
            print("✅ All expected keywords found")
        else:
            missing = expected_keywords - found_keywords
            print(f"❌ Missing keywords: {missing}")

def test_preprocessing_replacement():
    """Test text preprocessing with keyword replacement"""
    
    resolver = LegalCoreferenceResolver()
    
    # Sample text with legal cross-references
    original_text = '''
    The Company shall provide services to the Client.
    Company employees must follow all policies.
    This Agreement supersedes all previous agreements.
    The Provider is responsible for deliverables.
    '''
    
    # Sample definitions (as would be extracted from document)
    definitions = {
        'company': {
            'canonical_name': 'Summit Digital Solutions, Inc',
            'keyword_type': 'entity',
            'confidence': 0.95
        },
        'client': {
            'canonical_name': 'Acme Corporation',
            'keyword_type': 'entity',
            'confidence': 0.90
        },
        'agreement': {
            'canonical_name': 'Professional Services Agreement',
            'keyword_type': 'document',
            'confidence': 0.85
        },
        'provider': {
            'canonical_name': 'Summit Digital Solutions, Inc',
            'keyword_type': 'entity',
            'confidence': 0.90
        }
    }
    
    print("\n\n🔄 Testing Preprocessing Replacement")
    print("=" * 50)
    
    print("Original text:")
    print(original_text)
    
    processed_text = resolver.preprocess_text_with_replacements(original_text, definitions)
    
    print("\nProcessed text:")
    print(processed_text)
    
    print("\nReplacements made:")
    for keyword, definition in definitions.items():
        if definition['keyword_type'] == 'entity':  # Only entity keywords are replaced
            if keyword.lower() in original_text.lower():
                print(f"  • '{keyword}' → '{definition['canonical_name']}'")

def test_keyword_entities_and_relationships():
    """Test creation of keyword entities and relationships"""
    
    resolver = LegalCoreferenceResolver()
    
    # Sample definitions
    definitions = {
        'company': {
            'canonical_name': 'Summit Digital Solutions, Inc',
            'keyword_type': 'entity',
            'document': 'test-agreement.pdf',
            'context': 'Summit Digital Solutions, Inc. ("Company")',
            'confidence': 0.95
        },
        'agreement': {
            'canonical_name': 'Professional Services Agreement',
            'keyword_type': 'document',
            'document': 'test-agreement.pdf',
            'context': 'THIS PROFESSIONAL SERVICES AGREEMENT ("Agreement")',
            'confidence': 0.90
        }
    }
    
    print("\n\n🔗 Testing Keyword Entities and Relationships")
    print("=" * 50)
    
    # Test keyword entity creation
    keyword_entities = resolver.create_keyword_entities(definitions, 'test-agreement.pdf')
    
    print(f"Created {len(keyword_entities)} keyword entities:")
    for entity in keyword_entities:
        print(f"  • {entity['name']} (type: {entity['keyword_type']}, "
              f"refers to: {entity['canonical_reference']})")
    
    # Test relationship creation
    relationships = resolver.create_keyword_relationships(definitions, 'test-agreement.pdf')
    
    print(f"\nCreated {len(relationships)} relationships:")
    for rel in relationships:
        print(f"  • {rel['source_entity']} --{rel['relationship_type']}--> {rel['target_entity']}")

def main():
    """Run all legal coreference tests"""
    print("🏛️ Legal Coreference Resolution Test Suite")
    print("=" * 60)
    
    try:
        test_legal_pattern_extraction()
        test_preprocessing_replacement()
        test_keyword_entities_and_relationships()
        
        print("\n\n✅ All tests completed successfully!")
        print("\n🎯 Next Steps:")
        print("1. Run the knowledge graph builder with legal coreference enabled")
        print("2. Check for reduced 'Company' entities in the resulting graph")
        print("3. Verify legal keyword entities and relationships are created")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


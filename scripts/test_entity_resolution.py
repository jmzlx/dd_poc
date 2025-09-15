#!/usr/bin/env python3
"""
Test Entity Resolution

Quick test script to validate the entity resolution system on existing
Summit Digital Solutions data before rebuilding the full knowledge graph.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.entity_resolution import EntityResolver
from app.core.logging import setup_logging

# Set up logging
logger = setup_logging("test_entity_resolution", log_level="INFO")

def load_existing_entities(store_name: str = "summit-digital-solutions-inc") -> Dict[str, List[Dict]]:
    """Load existing entities from the knowledge graph"""
    entities_file = Path(__file__).parent.parent / "data" / "search_indexes" / "knowledge_graphs" / f"{store_name}_entities.json"
    
    if not entities_file.exists():
        raise FileNotFoundError(f"Entities file not found: {entities_file}")
    
    with open(entities_file, 'r') as f:
        data = json.load(f)
    
    return {
        'companies': data.get('companies', []),
        'people': data.get('people', []),
        'financial_metrics': data.get('financial_metrics', []),
        'documents': data.get('documents', [])
    }

def analyze_sample_entities(entities: Dict[str, List[Dict]], sample_size: int = 20):
    """Analyze a sample of entities to understand potential duplicates"""
    print("\n🔍 Sample Entity Analysis:")
    print("=" * 50)
    
    for entity_type, entity_list in entities.items():
        if not entity_list:
            continue
            
        print(f"\n{entity_type.upper()} (showing first {sample_size}):")
        print("-" * 30)
        
        # Show sample entities with their key attributes
        sample_entities = entity_list[:sample_size]
        for i, entity in enumerate(sample_entities, 1):
            name = entity.get('name', 'N/A')
            confidence = entity.get('confidence', 0.0)
            source = entity.get('source', 'N/A')
            context = entity.get('context', '')[:100] + "..." if len(entity.get('context', '')) > 100 else entity.get('context', '')
            
            print(f"{i:2d}. {name}")
            print(f"    Confidence: {confidence:.3f}")
            print(f"    Source: {source}")
            print(f"    Context: {context}")
            print()

def find_potential_duplicates(entities: Dict[str, List[Dict]]) -> Dict[str, List[List[str]]]:
    """Find potential duplicates using simple string matching"""
    potential_duplicates = {}
    
    for entity_type, entity_list in entities.items():
        if len(entity_list) < 2:
            continue
            
        # Group by normalized names
        name_groups = {}
        for entity in entity_list:
            name = entity.get('name', '').strip().lower()
            # Simple normalization
            name = name.replace(',', '').replace('.', '').replace('inc', '').replace('corp', '').strip()
            
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(entity.get('name', ''))
        
        # Find groups with multiple entities
        duplicates = []
        for normalized_name, original_names in name_groups.items():
            if len(original_names) > 1:
                duplicates.append(original_names)
        
        if duplicates:
            potential_duplicates[entity_type] = duplicates
    
    return potential_duplicates

def test_entity_resolution():
    """Test the entity resolution system"""
    print("🧪 Testing Entity Resolution System")
    print("=" * 40)
    
    try:
        # Load existing entities
        print("📥 Loading existing entities...")
        entities = load_existing_entities()
        
        # Show original counts
        print("\n📊 Original Entity Counts:")
        total_original = 0
        for entity_type, entity_list in entities.items():
            count = len(entity_list)
            total_original += count
            print(f"  {entity_type}: {count}")
        print(f"  TOTAL: {total_original}")
        
        # Analyze sample entities
        analyze_sample_entities(entities)
        
        # Find potential duplicates using simple string matching
        print("\n🔍 Potential Duplicates (simple string matching):")
        potential_duplicates = find_potential_duplicates(entities)
        for entity_type, duplicate_groups in potential_duplicates.items():
            print(f"\n{entity_type}:")
            for i, group in enumerate(duplicate_groups[:5], 1):  # Show first 5 groups
                print(f"  {i}. {group}")
        
        # Test entity resolution with a smaller sample first
        print("\n🔬 Testing Entity Resolution (sample):")
        sample_entities = {}
        for entity_type, entity_list in entities.items():
            # Take first 10 entities of each type for testing (smaller sample for speed)
            sample_entities[entity_type] = entity_list[:10]
        
        # Initialize resolver and test
        resolver = EntityResolver()
        
        print("🚀 Running entity resolution...")
        resolved_entities = resolver.resolve_entities(sample_entities)
        
        # Show results
        print("\n📈 Resolution Results (sample):")
        stats = resolver.get_resolution_stats(sample_entities, resolved_entities)
        
        print(f"Overall: {stats['total_before']} → {stats['total_after']} entities "
              f"({stats['overall_reduction_percentage']:.1f}% reduction)")
        
        for entity_type, type_stats in stats['by_type'].items():
            if type_stats['duplicates_removed'] > 0:
                print(f"  {entity_type}: {type_stats['before']} → {type_stats['after']} "
                      f"({type_stats['duplicates_removed']} duplicates, "
                      f"{type_stats['reduction_percentage']:.1f}% reduction)")
        
        # Show some examples of resolved entities
        print("\n✨ Example Resolved Entities:")
        for entity_type, entity_list in resolved_entities.items():
            merged_entities = [e for e in entity_list if e.get('cluster_size', 1) > 1]
            if merged_entities:
                print(f"\n{entity_type} (showing merged entities):")
                for entity in merged_entities[:3]:  # Show first 3 merged entities
                    print(f"  • {entity['name']} (merged {entity['cluster_size']} entities)")
                    if entity.get('sources'):
                        print(f"    Sources: {len(entity['sources'])} documents")
                    if entity.get('merged_confidence'):
                        print(f"    Avg confidence: {entity['merged_confidence']:.3f}")
        
        print("\n✅ Entity resolution test completed successfully!")
        
    except Exception as e:
        logger.error(f"Entity resolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_entity_resolution()
    sys.exit(0 if success else 1)

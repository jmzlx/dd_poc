#!/usr/bin/env python3
"""
Knowledge Graph Builder Script

This script extends the FAISS index building process to create knowledge graphs
from document analysis. It extracts entities and relationships from processed
documents and builds NetworkX graphs that can be stored in the repo and loaded
efficiently in Streamlit Cloud.

The graph building process:
1. Load existing FAISS indices and document chunks
2. Extract entities (companies, people, contracts, etc.) using NER
3. Identify relationships between entities using pattern matching and AI
4. Build NetworkX graphs with rich metadata
5. Serialize graphs to files for fast loading in Streamlit

Run this after build_indexes.py to generate knowledge graphs.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime

# Progress indicators
from tqdm import tqdm

# NetworkX for graph operations
import networkx as nx

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_config
from app.core.logging import setup_logging
from app.core.utils import create_document_processor
from app.core.entity_resolution import EntityResolver
from app.core.legal_coreference import LegalCoreferenceResolver
from scripts.transformer_extractors import TransformerEntityExtractor

# Set up logging
logger = setup_logging("build_knowledge_graphs", log_level="INFO")

# Old regex-based extractors have been removed
# Now using transformer-based extractors from scripts.transformer_extractors

class KnowledgeGraphBuilder:
    """Build NetworkX knowledge graphs from extracted entities and relationships"""
    
    def __init__(self, store_name: str):
        self.store_name = store_name
        self.graph = nx.MultiDiGraph()  # Allow multiple edges between nodes
        
    def build_graph(self, entities: Dict[str, List[Dict]], relationships: List[Dict]) -> nx.MultiDiGraph:
        """Build knowledge graph from entities and relationships"""
        
        # Add entity nodes
        print(f"{BLUE}Adding entity nodes...{NC}")
        for entity_type, entity_list in entities.items():
            for entity in tqdm(entity_list, desc=f"Adding {entity_type}"):
                # Skip entities without names
                if 'name' not in entity or not entity['name']:
                    continue
                    
                node_id = f"{entity_type}:{entity['name']}"
                
                # Add node with rich metadata
                self.graph.add_node(node_id, 
                    name=entity['name'],
                    type=entity_type,
                    sources=entity.get('source', ''),
                    document_type=entity.get('document_type', 'unknown'),
                    context_samples=[entity.get('context', '')],
                    first_seen=datetime.now().isoformat()
                )
        
        # Add relationship edges
        print(f"{BLUE}Adding relationship edges...{NC}")
        for rel in tqdm(relationships, desc="Adding relationships"):
            # Find matching nodes
            source_nodes = [n for n in self.graph.nodes() if rel['source_entity'].lower() in n.lower()]
            target_nodes = [n for n in self.graph.nodes() if rel['target_entity'].lower() in n.lower()]
            
            for source_node in source_nodes:
                for target_node in target_nodes:
                    if source_node != target_node:
                        self.graph.add_edge(
                            source_node, 
                            target_node,
                            relationship=rel['relationship_type'],
                            source_document=rel['source_document'],
                            context=rel['context'],
                            confidence=rel['confidence']
                        )
        
        # Add graph metadata
        self.graph.graph.update({
            'store_name': self.store_name,
            'created_at': datetime.now().isoformat(),
            'num_entities': len(self.graph.nodes()),
            'num_relationships': len(self.graph.edges()),
            'entity_types': list(entities.keys())
        })
        
        return self.graph
    
    def compute_graph_metrics(self) -> Dict[str, Any]:
        """Compute useful graph metrics for analysis"""
        metrics = {
            'num_nodes': len(self.graph.nodes()),
            'num_edges': len(self.graph.edges()),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
        }
        
        # Node centrality measures
        if len(self.graph.nodes()) > 1:
            try:
                centrality = nx.degree_centrality(self.graph)
                metrics['top_central_entities'] = sorted(
                    [(node, score) for node, score in centrality.items()],
                    key=lambda x: x[1], reverse=True
                )[:10]
            except:
                metrics['top_central_entities'] = []
        
        # Entity type distribution
        entity_types = defaultdict(int)
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            entity_types[node_type] += 1
        metrics['entity_distribution'] = dict(entity_types)
        
        return metrics

def process_company_knowledge_graph(store_name: str, config) -> Optional[Dict[str, Any]]:
    """Process a single company's knowledge graph"""
    # Determine what type of data store this is
    store_type = "unknown"
    if "summit-digital-solutions" in store_name or "deepshield-systems" in store_name:
        store_type = "company data room"
    elif "questions" in store_name:
        store_type = "due diligence questions"
    elif "checklist" in store_name:
        store_type = "due diligence checklist"
    
    print(f"\n{GREEN}Processing knowledge graph for: {store_name} ({store_type}){NC}")
    
    try:
        # Load existing FAISS index and document processor
        document_processor = create_document_processor(store_name=store_name)
        
        if not document_processor.vector_store:
            print(f"{YELLOW}⚠️ No FAISS index found for {store_name}, skipping...{NC}")
            return None
        
        # Extract chunks from FAISS metadata
        chunks = []
        if hasattr(document_processor, 'chunks') and document_processor.chunks:
            chunks = document_processor.chunks
        else:
            # Fallback: extract from FAISS docstore
            for i, doc in enumerate(document_processor.vector_store.docstore._dict.values()):
                chunks.append({
                    'text': doc.page_content,
                    'source': doc.metadata.get('name', f'doc_{i}'),
                    'metadata': doc.metadata
                })
        
        if not chunks:
            print(f"{YELLOW}⚠️ No chunks found for {store_name}, skipping...{NC}")
            return None
        
        print(f"📄 Processing {len(chunks)} document chunks")
        
        # Apply legal coreference resolution (hybrid approach)
        print(f"{BLUE}Applying legal coreference resolution...{NC}")
        coreference_resolver = LegalCoreferenceResolver()
        processed_chunks, legal_definitions = coreference_resolver.process_document_chunks(
            chunks, use_preprocessing=True
        )
        
        total_definitions = sum(len(defs) for defs in legal_definitions.values())
        if total_definitions > 0:
            print(f"📋 Found {total_definitions} legal keyword definitions across {len(legal_definitions)} documents")
        
        # Extract entities using transformer-based extraction (on processed chunks)
        print(f"{BLUE}Initializing transformer-based entity extraction...{NC}")
        entity_extractor = TransformerEntityExtractor()
        raw_entities = entity_extractor.extract_entities(processed_chunks)
        
        total_raw_entities = sum(len(entity_list) for entity_list in raw_entities.values())
        print(f"🏷️ Extracted {total_raw_entities} raw entities")
        
        # Add legal keyword entities to the collection (Strategy 2)
        print(f"{BLUE}Adding legal keyword entities to knowledge graph...{NC}")
        entities_with_keywords = coreference_resolver.enhance_entities_with_keywords(raw_entities, legal_definitions)
        
        # Resolve duplicate entities using semantic embeddings
        print(f"{BLUE}Resolving duplicate entities using semantic embeddings...{NC}")
        entity_resolver = EntityResolver()
        entities = entity_resolver.resolve_entities(entities_with_keywords)
        
        # Get resolution statistics
        resolution_stats = entity_resolver.get_resolution_stats(raw_entities, entities)
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        print(f"✨ Entity resolution complete: {total_raw_entities} → {total_entities} entities "
              f"({resolution_stats['overall_reduction_percentage']:.1f}% reduction)")
        
        # Print per-type statistics
        for entity_type, stats in resolution_stats['by_type'].items():
            if stats['duplicates_removed'] > 0:
                print(f"   • {entity_type}: {stats['before']} → {stats['after']} "
                      f"({stats['duplicates_removed']} duplicates removed)")
        
        # Extract high-quality legal keyword relationships only
        print(f"{BLUE}Extracting legal keyword relationships...{NC}")
        relationships = coreference_resolver.create_all_keyword_relationships(legal_definitions)
        
        print(f"🔗 Extracted {len(relationships)} high-quality legal relationships")
        
        # Removed: Base transformer relationship extraction (low yield: 59 relationships from 3,091 chunks)
        # Legal keyword relationships provide 98% of the value with much higher precision
        
        # Build knowledge graph
        graph_builder = KnowledgeGraphBuilder(store_name)
        knowledge_graph = graph_builder.build_graph(entities, relationships)
        
        # Compute metrics
        metrics = graph_builder.compute_graph_metrics()
        print(f"📊 Graph metrics: {metrics['num_nodes']} nodes, {metrics['num_edges']} edges")
        
        # Save knowledge graph files
        graphs_dir = config.paths['faiss_dir'] / 'knowledge_graphs'
        graphs_dir.mkdir(exist_ok=True)
        
        # Save NetworkX graph (pickle format for fast loading)
        graph_file = graphs_dir / f"{store_name}_knowledge_graph.pkl"
        with open(graph_file, 'wb') as f:
            pickle.dump(knowledge_graph, f)
        
        # Save graph metadata and metrics (JSON for inspection)
        metadata_file = graphs_dir / f"{store_name}_graph_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'store_name': store_name,
                'metrics': metrics,
                'entities': {k: len(v) for k, v in entities.items()},
                'relationships_count': len(relationships),
                'created_at': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save entities for inspection (JSON)
        entities_file = graphs_dir / f"{store_name}_entities.json"
        with open(entities_file, 'w') as f:
            json.dump(entities, f, indent=2)
        
        print(f"✅ Knowledge graph saved to {graph_file}")
        
        return {
            'store_name': store_name,
            'success': True,
            'metrics': metrics,
            'files_created': [str(graph_file), str(metadata_file), str(entities_file)]
        }
        
    except Exception as e:
        print(f"{RED}❌ Error processing {store_name}: {str(e)}{NC}")
        logger.error(f"Knowledge graph processing failed for {store_name}: {e}")
        return {
            'store_name': store_name,
            'success': False,
            'error': str(e)
        }

def main():
    """Main function to build knowledge graphs for all companies"""
    print(f"{GREEN}🧠 Building Knowledge Graphs for Due Diligence Analysis{NC}")
    print(f"{GREEN}Using transformer-based entity and relationship extraction{NC}")
    print("=" * 60)
    
    # Load configuration
    config = get_config()
    
    # Find all existing FAISS indices
    faiss_dir = config.paths['faiss_dir']
    if not faiss_dir.exists():
        print(f"{RED}❌ FAISS directory not found: {faiss_dir}{NC}")
        print("Please run scripts/build_indexes.py first")
        return
    
    # Find all .faiss files (these are the indices)
    faiss_files = list(faiss_dir.glob("*.faiss"))
    if not faiss_files:
        print(f"{RED}❌ No FAISS indices found in {faiss_dir}{NC}")
        print("Please run scripts/build_indexes.py first")
        return
    
    # Extract store names from FAISS files
    store_names = [f.stem for f in faiss_files]
    print(f"Found {len(store_names)} FAISS indices: {', '.join(store_names)}")
    
    # Process each company's knowledge graph
    results = []
    for store_name in store_names:
        result = process_company_knowledge_graph(store_name, config)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{GREEN}📋 Knowledge Graph Building Summary{NC}")
    print("=" * 40)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"✅ Successfully processed: {len(successful)} data stores")
    for result in successful:
        metrics = result.get('metrics', {})
        store_name = result['store_name']
        
        # Determine store type for clearer output
        if "summit-digital-solutions" in store_name or "deepshield-systems" in store_name:
            store_type = "company"
        elif "questions" in store_name:
            store_type = "questions"
        elif "checklist" in store_name:
            store_type = "checklist"
        else:
            store_type = "unknown"
            
        print(f"   • {store_name} ({store_type}): {metrics.get('num_nodes', 0)} entities, {metrics.get('num_edges', 0)} relationships")
    
    if failed:
        print(f"❌ Failed to process: {len(failed)} data stores")
        for result in failed:
            print(f"   • {result['store_name']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n🎉 Knowledge graph building complete!")
    print(f"📁 Files saved in: {config.paths['faiss_dir'] / 'knowledge_graphs'}")

if __name__ == "__main__":
    main()

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
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
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

# Set up logging
logger = setup_logging("build_knowledge_graphs", log_level="INFO")

class EntityExtractor:
    """Extract entities from document chunks using pattern matching and NER"""
    
    def __init__(self):
        # Common business entity patterns
        self.company_patterns = [
            r'\b([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Corporation|Company|Co|Ltd|Limited|Group|Holdings|Ventures|Partners|Associates|Solutions|Systems|Technologies|Services|Enterprises)\.?)\b',
            r'\b([A-Z][a-zA-Z\s&]+(?:AG|GmbH|SA|SAS|PLC|Pty|AB|AS))\b',
        ]
        
        self.person_patterns = [
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b(?=\s+(?:CEO|CFO|CTO|President|Director|Manager|VP|Vice President|Chairman|Founder))',
            r'(?:CEO|CFO|CTO|President|Director|Manager|VP|Vice President|Chairman|Founder)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]
        
        self.financial_patterns = [
            r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|M|B|K))?',
            r'(?:revenue|profit|loss|EBITDA|earnings)\s*of\s*\$[\d,]+',
            r'(?:valuation|market cap)\s*[:=]\s*\$[\d,]+',
        ]
        
        self.contract_patterns = [
            r'(?:contract|agreement|deal|acquisition|merger|partnership|joint venture|MOU|LOI)',
            r'(?:signed|executed|entered into|agreed to)\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]

    def extract_entities(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities from document chunks"""
        entities = {
            'companies': [],
            'people': [],
            'financial_metrics': [],
            'contracts': [],
            'dates': []
        }
        
        for chunk in tqdm(chunks, desc="Extracting entities"):
            text = chunk.get('text', '')
            source = chunk.get('source', 'unknown')
            metadata = chunk.get('metadata', {})
            
            # Extract companies
            for pattern in self.company_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    company_name = match.group(1).strip()
                    if len(company_name) > 3:  # Filter out short matches
                        entities['companies'].append({
                            'name': company_name,
                            'source': source,
                            'context': text[max(0, match.start()-50):match.end()+50],
                            'chunk_id': metadata.get('chunk_id'),
                            'document_type': metadata.get('document_type', 'unknown')
                        })
            
            # Extract people
            for pattern in self.person_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    person_name = match.group(1).strip()
                    entities['people'].append({
                        'name': person_name,
                        'source': source,
                        'context': text[max(0, match.start()-50):match.end()+50],
                        'chunk_id': metadata.get('chunk_id'),
                        'document_type': metadata.get('document_type', 'unknown')
                    })
            
            # Extract financial metrics
            for pattern in self.financial_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities['financial_metrics'].append({
                        'value': match.group(0),
                        'source': source,
                        'context': text[max(0, match.start()-100):match.end()+100],
                        'chunk_id': metadata.get('chunk_id'),
                        'document_type': metadata.get('document_type', 'unknown')
                    })
        
        return entities

class RelationshipExtractor:
    """Extract relationships between entities"""
    
    def __init__(self):
        self.relationship_patterns = [
            # Company relationships
            (r'(.+?)\s+(?:acquired|purchased|bought)\s+(.+)', 'ACQUIRED'),
            (r'(.+?)\s+(?:merged with|combined with)\s+(.+)', 'MERGED_WITH'),
            (r'(.+?)\s+(?:partnered with|partnership with)\s+(.+)', 'PARTNERSHIP'),
            (r'(.+?)\s+(?:invested in|investment in)\s+(.+)', 'INVESTED_IN'),
            (r'(.+?)\s+(?:subsidiary of|owned by)\s+(.+)', 'SUBSIDIARY_OF'),
            
            # Person-company relationships
            (r'(.+?)\s+(?:CEO|CFO|CTO|President|Director)\s+(?:of|at)\s+(.+)', 'EXECUTIVE_OF'),
            (r'(.+?)\s+(?:founded|established|started)\s+(.+)', 'FOUNDED'),
            (r'(.+?)\s+(?:joined|hired by)\s+(.+)', 'EMPLOYED_BY'),
            
            # Contract relationships
            (r'(.+?)\s+(?:signed|executed|entered into).+?(?:with|and)\s+(.+)', 'CONTRACT_WITH'),
        ]

    def extract_relationships(self, entities: Dict[str, List[Dict]], chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Extract relationships from text using pattern matching"""
        relationships = []
        
        # Create entity lookup for quick matching
        entity_names = set()
        for entity_type in entities:
            for entity in entities[entity_type]:
                if 'name' in entity and entity['name']:
                    entity_names.add(entity['name'].lower())
        
        for chunk in tqdm(chunks, desc="Extracting relationships"):
            text = chunk.get('text', '')
            source = chunk.get('source', 'unknown')
            
            for pattern, relationship_type in self.relationship_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity1 = match.group(1).strip()
                    entity2 = match.group(2).strip()
                    
                    # Validate that both entities exist in our entity list
                    if (entity1.lower() in entity_names and 
                        entity2.lower() in entity_names and 
                        entity1 != entity2):
                        
                        relationships.append({
                            'source_entity': entity1,
                            'target_entity': entity2,
                            'relationship_type': relationship_type,
                            'source_document': source,
                            'context': text[max(0, match.start()-100):match.end()+100],
                            'confidence': 0.8  # Pattern-based confidence
                        })
        
        return relationships

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
    print(f"\n{GREEN}Processing knowledge graph for: {store_name}{NC}")
    
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
        
        # Extract entities
        entity_extractor = EntityExtractor()
        entities = entity_extractor.extract_entities(chunks)
        
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        print(f"🏷️ Extracted {total_entities} entities")
        
        # Extract relationships
        relationship_extractor = RelationshipExtractor()
        relationships = relationship_extractor.extract_relationships(entities, chunks)
        
        print(f"🔗 Extracted {len(relationships)} relationships")
        
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
    
    print(f"✅ Successfully processed: {len(successful)} companies")
    for result in successful:
        metrics = result.get('metrics', {})
        print(f"   • {result['store_name']}: {metrics.get('num_nodes', 0)} entities, {metrics.get('num_edges', 0)} relationships")
    
    if failed:
        print(f"❌ Failed to process: {len(failed)} companies")
        for result in failed:
            print(f"   • {result['store_name']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n🎉 Knowledge graph building complete!")
    print(f"📁 Files saved in: {config.paths['faiss_dir'] / 'knowledge_graphs'}")

if __name__ == "__main__":
    main()

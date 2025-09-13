#!/usr/bin/env python3
"""
Knowledge Graph Tab

This tab provides an interface for exploring pre-computed knowledge graphs
generated from due diligence documents. It offers entity search, relationship
exploration, and graph analysis capabilities.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any, Optional

from app.core.knowledge_graph import KnowledgeGraphManager, get_available_knowledge_graphs
from app.ui.tabs.tab_base import TabBase
from app.ui.error_handler import handle_ui_errors
from app.core.logging import logger

class GraphTab(TabBase):
    """Knowledge Graph exploration tab"""
    
    def __init__(self, session_manager, config, ai_handler, export_handler):
        super().__init__(session_manager, config, ai_handler, export_handler)
        self.tab_name = "Knowledge Graph"
        self.tab_key = "graph"
    
    @handle_ui_errors("Knowledge Graph", "Please try refreshing the page")
    def render(self):
        """Render the knowledge graph tab"""
        st.header("🧠 Knowledge Graph Explorer")
        
        # Check if we have a loaded company
        if not self.session.vdr_store:
            st.info("📋 Please load a company first using the sidebar.")
            return
        
        company_name = self.session.vdr_store
        
        # Initialize knowledge graph manager
        if f'kg_manager_{company_name}' not in st.session_state:
            st.session_state[f'kg_manager_{company_name}'] = KnowledgeGraphManager(company_name)
        
        kg_manager = st.session_state[f'kg_manager_{company_name}']
        
        # Load graph if not already loaded
        if not kg_manager.is_available():
            with st.spinner("Loading knowledge graph..."):
                if not kg_manager.load_graph():
                    st.error("❌ Knowledge graph not found for this company.")
                    st.info("💡 Run `python scripts/build_knowledge_graphs.py` to generate knowledge graphs.")
                    return
        
        # Display graph summary
        self._render_graph_summary(kg_manager)
        
        # Main interface tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Entity Search", 
            "🔗 Relationship Explorer", 
            "📊 Graph Analysis",
            "🎯 Path Finder",
            "🧠 Semantic Search"
        ])
        
        with tab1:
            self._render_entity_search(kg_manager)
        
        with tab2:
            self._render_relationship_explorer(kg_manager)
        
        with tab3:
            self._render_graph_analysis(kg_manager)
        
        with tab4:
            self._render_path_finder(kg_manager)
        
        with tab5:
            self._render_semantic_search(kg_manager)
    
    def _render_graph_summary(self, kg_manager: KnowledgeGraphManager):
        """Render graph summary statistics"""
        stats = kg_manager.get_summary_stats()
        
        if not stats:
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entities", stats.get('num_entities', 0))
        
        with col2:
            st.metric("Relationships", stats.get('num_relationships', 0))
        
        with col3:
            entity_types = stats.get('entity_types', {})
            st.metric("Entity Types", len(entity_types))
        
        with col4:
            rel_types = stats.get('relationship_types', {})
            st.metric("Relationship Types", len(rel_types))
        
        # Entity distribution chart
        if entity_types:
            with st.expander("📊 Entity Distribution", expanded=False):
                fig = px.pie(
                    values=list(entity_types.values()),
                    names=list(entity_types.keys()),
                    title="Distribution of Entity Types"
                )
                st.plotly_chart(fig, width='stretch')
    
    def _render_entity_search(self, kg_manager: KnowledgeGraphManager):
        """Render entity search interface"""
        st.subheader("🔍 Search Entities")
        
        # Search controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search for entities (companies, people, contracts, etc.)",
                placeholder="e.g., Microsoft, John Smith, acquisition...",
                key="entity_search_query"
            )
        
        with col2:
            entity_types = ['All'] + list(kg_manager.get_summary_stats().get('entity_types', {}).keys())
            selected_type = st.selectbox(
                "Filter by type",
                entity_types,
                key="entity_type_filter"
            )
        
        if search_query:
            # Perform search
            filter_type = None if selected_type == 'All' else selected_type
            results = kg_manager.search_entities(
                search_query, 
                entity_type=filter_type,
                limit=20
            )
            
            if results:
                st.success(f"Found {len(results)} matching entities")
                
                # Display results
                for i, entity in enumerate(results):
                    with st.expander(f"🏷️ {entity['name']} ({entity['type']})", expanded=i==0):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Type:** {entity['type']}")
                            st.write(f"**Sources:** {entity['sources']}")
                            st.write(f"**Document Type:** {entity['document_type']}")
                            
                            # Show context samples
                            if entity.get('context_samples'):
                                st.write("**Context:**")
                                for context in entity['context_samples']:
                                    if context.strip():
                                        st.write(f"_{context.strip()}_")
                        
                        with col2:
                            st.metric("Relevance Score", f"{entity['score']:.2f}")
                            
                            # Button to explore relationships
                            if st.button(f"Explore Relationships", key=f"explore_{i}"):
                                st.session_state['selected_entity'] = entity['name']
                                st.rerun()
            else:
                st.info("No entities found matching your search criteria.")
    
    def _render_relationship_explorer(self, kg_manager: KnowledgeGraphManager):
        """Render relationship exploration interface"""
        st.subheader("🔗 Relationship Explorer")
        
        # Entity selection
        selected_entity = st.session_state.get('selected_entity', '')
        entity_input = st.text_input(
            "Enter entity name to explore relationships",
            value=selected_entity,
            placeholder="e.g., Microsoft, John Smith...",
            key="relationship_entity_input"
        )
        
        if entity_input:
            # Get relationships
            relationships = kg_manager.get_entity_relationships(entity_input)
            
            if relationships['outgoing'] or relationships['incoming']:
                # Display outgoing relationships
                if relationships['outgoing']:
                    st.write("### ➡️ Outgoing Relationships")
                    outgoing_data = []
                    for rel in relationships['outgoing']:
                        outgoing_data.append({
                            'Target': rel['target'],
                            'Type': rel['target_type'],
                            'Relationship': rel['relationship'],
                            'Source Doc': rel['source_document'],
                            'Confidence': f"{rel['confidence']:.2f}"
                        })
                    
                    df_out = pd.DataFrame(outgoing_data)
                    st.dataframe(df_out, width='stretch')
                    
                    # Show relationship context on selection
                    if st.checkbox("Show relationship contexts", key="show_outgoing_context"):
                        for i, rel in enumerate(relationships['outgoing']):
                            if rel['context'].strip():
                                st.write(f"**{rel['target']} ({rel['relationship']}):**")
                                st.write(f"_{rel['context']}_")
                                st.write("---")
                
                # Display incoming relationships
                if relationships['incoming']:
                    st.write("### ⬅️ Incoming Relationships")
                    incoming_data = []
                    for rel in relationships['incoming']:
                        incoming_data.append({
                            'Source': rel['source'],
                            'Type': rel['source_type'],
                            'Relationship': rel['relationship'],
                            'Source Doc': rel['source_document'],
                            'Confidence': f"{rel['confidence']:.2f}"
                        })
                    
                    df_in = pd.DataFrame(incoming_data)
                    st.dataframe(df_in, width='stretch')
                    
                    # Show relationship context on selection
                    if st.checkbox("Show relationship contexts", key="show_incoming_context"):
                        for i, rel in enumerate(relationships['incoming']):
                            if rel['context'].strip():
                                st.write(f"**{rel['source']} ({rel['relationship']}):**")
                                st.write(f"_{rel['context']}_")
                                st.write("---")
                
                # Relationship type distribution
                all_rels = relationships['outgoing'] + relationships['incoming']
                rel_types = {}
                for rel in all_rels:
                    rel_type = rel['relationship']
                    rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
                
                if rel_types:
                    st.write("### 📊 Relationship Type Distribution")
                    fig = px.bar(
                        x=list(rel_types.keys()),
                        y=list(rel_types.values()),
                        title=f"Relationships for {entity_input}"
                    )
                    st.plotly_chart(fig, width='stretch')
            
            else:
                st.info(f"No relationships found for '{entity_input}'. Try a different entity name.")
    
    def _render_graph_analysis(self, kg_manager: KnowledgeGraphManager):
        """Render graph analysis interface"""
        st.subheader("📊 Graph Analysis")
        
        # Central entities
        st.write("### 🎯 Most Important Entities")
        central_entities = kg_manager.get_central_entities(limit=15)
        
        if central_entities:
            # Create a bar chart of centrality scores
            names = [e['name'] for e in central_entities]
            scores = [e['centrality_score'] for e in central_entities]
            types = [e['type'] for e in central_entities]
            
            fig = px.bar(
                x=scores,
                y=names,
                orientation='h',
                color=types,
                title="Entity Centrality Scores",
                labels={'x': 'Centrality Score', 'y': 'Entity'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width='stretch')
            
            # Display detailed table
            with st.expander("📋 Detailed Central Entities", expanded=False):
                central_df = pd.DataFrame([{
                    'Entity': e['name'],
                    'Type': e['type'],
                    'Centrality Score': e['centrality_score'],
                    'Connections': e['num_connections'],
                    'Sources': e['sources']
                } for e in central_entities])
                st.dataframe(central_df, width='stretch')
        
        # Entity clusters
        st.write("### 🎭 Entity Clusters")
        clusters = kg_manager.get_entity_clusters()
        
        if clusters:
            st.info(f"Found {len(clusters)} clusters of related entities")
            
            for i, cluster in enumerate(clusters):
                with st.expander(f"Cluster {i+1} ({len(cluster)} entities)", expanded=i==0):
                    # Display cluster as tags
                    cluster_html = " • ".join([f"**{entity}**" for entity in cluster])
                    st.write(cluster_html)
        else:
            st.info("No significant entity clusters found.")
    
    def _render_path_finder(self, kg_manager: KnowledgeGraphManager):
        """Render path finding interface"""
        st.subheader("🎯 Path Finder")
        st.write("Find connections between two entities in the knowledge graph.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_entity = st.text_input(
                "Source Entity",
                placeholder="e.g., Microsoft",
                key="path_source_entity"
            )
        
        with col2:
            target_entity = st.text_input(
                "Target Entity",
                placeholder="e.g., OpenAI",
                key="path_target_entity"
            )
        
        max_length = st.slider("Maximum Path Length", 1, 5, 3, key="max_path_length")
        
        if source_entity and target_entity and st.button("Find Paths", key="find_paths_btn"):
            with st.spinner("Searching for paths..."):
                paths = kg_manager.find_paths(source_entity, target_entity, max_length)
            
            if paths:
                st.success(f"Found {len(paths)} path(s) between {source_entity} and {target_entity}")
                
                for i, path in enumerate(paths):
                    st.write(f"**Path {i+1}:**")
                    path_str = " → ".join(path)
                    st.write(f"🔗 {path_str}")
                    
                    # Show path length
                    st.write(f"_Length: {len(path)-1} steps_")
                    st.write("---")
            else:
                st.info(f"No paths found between {source_entity} and {target_entity} within {max_length} steps.")
        
        # Path finding tips
        with st.expander("💡 Path Finding Tips", expanded=False):
            st.write("""
            - **Entity names**: Use exact or partial entity names as they appear in the documents
            - **Path length**: Shorter paths show direct connections, longer paths reveal indirect relationships
            - **Multiple paths**: Different paths can reveal different types of business relationships
            - **Use cases**: 
                - Find how two companies are connected
                - Trace investment or acquisition chains
                - Discover business partnerships and alliances
            """)
    
    def _render_semantic_search(self, kg_manager: KnowledgeGraphManager):
        """Render semantic search interface using FAISS embeddings"""
        st.subheader("🧠 Semantic Search")
        st.write("Search entities using natural language queries powered by your existing FAISS embeddings.")
        
        # Semantic entity search
        st.write("### 🔍 Semantic Entity Search")
        semantic_query = st.text_input(
            "Describe what you're looking for (e.g., 'technology companies', 'financial partnerships', 'recent acquisitions')",
            placeholder="e.g., companies involved in AI partnerships",
            key="semantic_entity_query"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            semantic_limit = st.slider("Max results", 5, 20, 10, key="semantic_limit")
        with col2:
            similarity_threshold = st.slider("Similarity threshold", 0.1, 0.8, 0.3, key="similarity_threshold")
        
        if semantic_query and st.button("🔍 Semantic Search", key="semantic_search_btn"):
            with st.spinner("Searching using AI embeddings..."):
                results = kg_manager.semantic_search_entities(
                    semantic_query, 
                    limit=semantic_limit, 
                    similarity_threshold=similarity_threshold
                )
            
            if results:
                st.success(f"Found {len(results)} semantically relevant entities")
                
                for i, entity in enumerate(results):
                    with st.expander(f"🏷️ {entity['name']} ({entity['type']}) - Score: {entity['similarity_score']:.3f}", expanded=i==0):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Type:** {entity['type']}")
                            st.write(f"**Sources:** {entity['sources']}")
                            st.write(f"**Document Type:** {entity['document_type']}")
                            
                            # Show matching context
                            if entity.get('matching_context'):
                                st.write("**Relevant Context:**")
                                st.write(f"_{entity['matching_context']}_")
                            
                            # Show original context samples
                            if entity.get('context_samples'):
                                st.write("**Entity Context:**")
                                for context in entity['context_samples']:
                                    if context.strip():
                                        st.write(f"_{context.strip()}_")
                        
                        with col2:
                            st.metric("Similarity Score", f"{entity['similarity_score']:.3f}")
                            
                            # Button to explore relationships
                            if st.button(f"Explore Relations", key=f"semantic_explore_{i}"):
                                st.session_state['selected_entity'] = entity['name']
                                st.rerun()
            else:
                st.info("No entities found matching your semantic query. Try adjusting the similarity threshold or rephrasing your query.")
        
        # Context-based related entities
        st.write("### 🔗 Find Related by Context")
        st.write("Find entities that appear in similar contexts to a reference entity.")
        
        context_entity = st.text_input(
            "Reference entity name",
            placeholder="e.g., Microsoft",
            key="context_reference_entity"
        )
        
        context_limit = st.slider("Max related entities", 3, 15, 5, key="context_limit")
        
        if context_entity and st.button("Find Related by Context", key="find_context_related_btn"):
            with st.spinner("Finding contextually related entities..."):
                related = kg_manager.find_related_entities_by_context(context_entity, limit=context_limit)
            
            if related:
                st.success(f"Found {len(related)} contextually related entities")
                
                related_data = []
                for entity in related:
                    related_data.append({
                        'Entity': entity['name'],
                        'Type': entity['type'],
                        'Similarity': f"{entity['similarity_score']:.3f}",
                        'Reason': entity['relationship_reason'],
                        'Sources': entity['sources']
                    })
                
                df_related = pd.DataFrame(related_data)
                st.dataframe(df_related, width='stretch')
                
                # Show context samples for selected entities
                if st.checkbox("Show context samples", key="show_related_contexts"):
                    for entity in related:
                        if entity.get('context_samples'):
                            st.write(f"**{entity['name']}:**")
                            for context in entity['context_samples']:
                                if context.strip():
                                    st.write(f"_{context.strip()}_")
                            st.write("---")
            else:
                st.info(f"No contextually related entities found for '{context_entity}'.")
        
        # Semantic path search
        st.write("### 🎯 Semantic Path Discovery")
        st.write("Find connection paths that are semantically relevant to your query.")
        
        path_query = st.text_input(
            "Describe the type of connections you want to find",
            placeholder="e.g., investment relationships, technology partnerships",
            key="semantic_path_query"
        )
        
        max_semantic_paths = st.slider("Max paths", 3, 10, 5, key="max_semantic_paths")
        
        if path_query and st.button("Find Semantic Paths", key="semantic_paths_btn"):
            with st.spinner("Discovering relevant connection paths..."):
                paths = kg_manager.semantic_path_search(path_query, max_paths=max_semantic_paths)
            
            if paths:
                st.success(f"Found {len(paths)} relevant connection paths")
                
                for i, path_info in enumerate(paths):
                    st.write(f"**Path {i+1}:** (Relevance: {path_info['relevance_score']:.3f})")
                    path_str = " → ".join(path_info['path'])
                    st.write(f"🔗 {path_str}")
                    st.write(f"_{path_info['query_relevance']}_")
                    st.write(f"Length: {path_info['path_length']} steps")
                    st.write("---")
            else:
                st.info(f"No semantically relevant paths found for '{path_query}'.")
        
        # Semantic search tips
        with st.expander("💡 Semantic Search Tips", expanded=False):
            st.write("""
            **Semantic Search Benefits:**
            - Uses your existing FAISS embeddings for intelligent matching
            - Finds entities based on meaning, not just keywords
            - Discovers hidden relationships through context similarity
            - Leverages the same AI models used in your document analysis
            
            **Query Examples:**
            - "technology companies with AI focus"
            - "recent merger and acquisition activity"
            - "financial services partnerships"
            - "regulatory compliance issues"
            - "key executive leadership"
            
            **How it works:**
            1. Your query is embedded using the same model as your documents
            2. FAISS finds the most similar document chunks
            3. Entities from those chunks are returned with similarity scores
            4. Results are ranked by semantic relevance
            
            **Performance Notes:**
            - Requires existing FAISS indices (same as your document search)
            - No additional models or external services needed
            - Leverages your pre-computed embeddings for fast results
            """)

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the knowledge graph tab"""
        if not self.session.vdr_store:
            return {
                'ready': False,
                'message': 'No company loaded'
            }
        
        company_name = self.session.vdr_store
        available_graphs = get_available_knowledge_graphs()
        
        if company_name not in available_graphs:
            return {
                'ready': False,
                'message': f'Knowledge graph not available for {company_name}'
            }
        
        return {
            'ready': True,
            'message': f'Knowledge graph ready for {company_name}'
        }

#!/usr/bin/env python3
"""
UI Components Module

This module contains reusable Streamlit UI components and layout functions.
Separates UI logic from business logic for better maintainability.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def render_project_selector() -> Tuple[Optional[str], Optional[str]]:
    """
    Render project and data room selector in sidebar
    
    Returns:
        Tuple of (selected_project_path, selected_data_room_path)
    """
    st.subheader("🎯 Select Project")
    
    # Scan for available projects
    projects = []
    data_base_path = Path("data").resolve() if Path("data").exists() else None
    
    if data_base_path and data_base_path.exists():
        # First check if there's a vdrs folder with projects
        vdrs_path = data_base_path / "vdrs"
        if vdrs_path.exists():
            # Look for project directories in vdrs
            for project_dir in vdrs_path.iterdir():
                if project_dir.is_dir() and not project_dir.name.startswith('.'):
                    # Check if it has subdirectories that could be data rooms
                    subdirs = [d for d in project_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    if subdirs:
                        # Count total documents in all data rooms
                        total_docs = sum(1 for f in project_dir.rglob('*') 
                                       if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md'])
                        if total_docs > 0:
                            projects.append({
                                'name': project_dir.name.replace('-', ' ').replace('_', ' ').title(),
                                'path': str(project_dir),
                                'data_rooms': len(subdirs),
                                'total_docs': total_docs
                            })
        
        # Also look for project directories directly in data folder (excluding special folders)
        for project_dir in data_base_path.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.') and project_dir.name not in ['checklist', 'questions', 'vdrs', 'strategy']:
                # Check if it has subdirectories that could be data rooms
                subdirs = [d for d in project_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                if subdirs:
                    # Count total documents in all data rooms
                    total_docs = sum(1 for f in project_dir.rglob('*') 
                                   if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md'])
                    if total_docs > 0:
                        projects.append({
                            'name': project_dir.name.replace('-', ' ').replace('_', ' ').title(),
                            'path': str(project_dir),
                            'data_rooms': len(subdirs),
                            'total_docs': total_docs
                        })
    
    selected_project_path = None
    selected_data_room_path = None
    
    if projects:
        # Sort projects by name
        projects.sort(key=lambda x: x['name'])
        
        # Create display names with counts
        project_options = [f"{proj['name']} ({proj['data_rooms']} rooms, {proj['total_docs']} docs)" for proj in projects]
        selected_project_idx = st.selectbox(
            "Select Project",
            range(len(project_options)),
            format_func=lambda x: project_options[x],
            help=f"Found {len(projects)} projects",
            key="project_selector"
        )
        selected_project = projects[selected_project_idx]
        selected_project_path = selected_project['path']
        
        # Data room selector (filtered by selected project)
        st.subheader("📁 Select Data Room")
        selected_data_room_path = render_data_room_selector(selected_project['path'])
        
    else:
        # No projects found - fall back to manual entry
        st.warning("No projects found in the data directory.")
        st.info("Projects should be folders in 'data/' containing company data rooms")
        selected_data_room_path = st.text_input(
            "Data Room Path (manual):",
            value="data/sample-project/sample-company",
            help="Enter the path to your data room directory"
        )
    
    return selected_project_path, selected_data_room_path


def render_data_room_selector(project_path: str) -> Optional[str]:
    """
    Render data room selector for a given project
    
    Args:
        project_path: Path to the selected project
        
    Returns:
        Selected data room path or None
    """
    # Scan for data rooms within the selected project
    data_rooms = []
    project_path_obj = Path(project_path)
    
    for data_room_dir in project_path_obj.iterdir():
        if data_room_dir.is_dir() and not data_room_dir.name.startswith('.'):
            # Count documents for display
            doc_count = sum(1 for f in data_room_dir.rglob('*') 
                          if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md'])
            if doc_count > 0:  # Only show directories with documents
                data_rooms.append({
                    'name': data_room_dir.name.replace('-', ' ').replace('_', ' ').title(),
                    'path': str(data_room_dir),
                    'docs': doc_count
                })
    
    if data_rooms:
        # Sort data rooms by name
        data_rooms.sort(key=lambda x: x['name'])
        # Create display names with document counts
        room_options = [f"{room['name']} ({room['docs']} docs)" for room in data_rooms]
        selected_room_idx = st.selectbox(
            "Select Data Room",
            range(len(room_options)),
            format_func=lambda x: room_options[x],
            help=f"Found {len(data_rooms)} data rooms",
            key="dataroom_selector"
        )
        return data_rooms[selected_room_idx]['path']
    else:
        st.warning(f"No data rooms found in project")
        return st.text_input(
            "Data Room Path (manual):",
            value=f"{project_path}/company",
            help="Enter the path to your data room directory"
        )


def render_ai_settings() -> Tuple[bool, Optional[str], str]:
    """
    Render AI enhancement settings in sidebar
    
    Returns:
        Tuple of (use_ai_features, api_key, model_choice)
    """
    st.subheader("🤖 AI Enhancement Settings")
    
    # Single toggle for AI features
    use_ai_features = st.checkbox(
        "Enable AI Features", 
        value=True, 
        help="Enable Claude AI for document summaries, intelligent matching, and enhanced Q&A"
    )
    
    api_key = None
    model_choice = "claude-3-haiku-20240307"
    
    if use_ai_features:
        # Check if API key is in environment
        import os
        env_key = os.getenv('ANTHROPIC_API_KEY')
        if env_key:
            st.success("✅ API key loaded from .env file")
            api_key = env_key
        else:
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                placeholder="sk-ant-...",
                help="Enter your Anthropic API key or set ANTHROPIC_API_KEY environment variable"
            )
        
        # Model selection
        model_choice = st.radio(
            "Claude Model",
            ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
            index=0,
            help="Haiku: Fast & cheap | Sonnet: Balanced | Opus: Most capable"
        )
    else:
        st.info("🔧 AI features disabled - using traditional embedding-based matching")
        st.caption("Enable the toggle above to use AI-powered enhancements")
    
    return use_ai_features, api_key, model_choice


def render_file_selector(directory: str, file_type: str, key_suffix: str) -> Tuple[Optional[str], str]:
    """
    Render file selector for checklists, questions, or strategies
    
    Args:
        directory: Directory to scan for files
        file_type: Type of file (for display)
        key_suffix: Unique suffix for streamlit keys
        
    Returns:
        Tuple of (selected_file_path, file_content)
    """
    files = []
    dir_path = Path(directory)
    
    if dir_path.exists():
        for file in dir_path.glob("*.md"):
            if not file.name.startswith('.'):
                files.append({
                    'name': file.stem.replace('_', ' ').replace('-', ' ').title(),
                    'path': str(file),
                    'filename': file.name
                })
    
    file_content = ""
    selected_file_path = None
    
    if files:
        files.sort(key=lambda x: x['name'])
        file_names = [f['name'] for f in files]
        selected_file_idx = st.selectbox(
            f"📋 Select {file_type}:",
            range(len(file_names)),
            format_func=lambda x: file_names[x],
            help=f"Found {len(files)} {file_type.lower()}s",
            key=f"{file_type.lower()}_selector_{key_suffix}"
        )
        selected_file = files[selected_file_idx]
        selected_file_path = selected_file['path']
        file_content = Path(selected_file['path']).read_text(encoding='utf-8')
    else:
        st.info(f"No {file_type.lower()}s found in {directory}/")
    
    return selected_file_path, file_content


def render_progress_section(total_steps: int = 10) -> Tuple[Any, Any]:
    """
    Render processing progress section
    
    Args:
        total_steps: Total number of processing steps
        
    Returns:
        Tuple of (progress_bar, status_text)
    """
    st.markdown("### 🚀 Processing Data Room")
    overall_progress = st.progress(0, text="Initializing...")
    status_text = st.empty()
    
    return overall_progress, status_text


def render_metrics_row(metrics: Dict[str, Any]) -> None:
    """
    Render a row of metrics
    
    Args:
        metrics: Dictionary of metric_name -> metric_value
    """
    cols = st.columns(len(metrics))
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(name, value)


def render_checklist_results(
    checklist_results: Dict, 
    relevancy_threshold: float = 0.4,
    primary_threshold: float = 0.5
) -> None:
    """
    Render checklist matching results with threshold controls
    
    Args:
        checklist_results: Results from checklist matching
        relevancy_threshold: Minimum relevancy threshold
        primary_threshold: Threshold for primary documents
    """
    if not checklist_results:
        st.info("👈 Configure and process data room to see checklist matching results")
        return
    
    # Add relevancy threshold controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        relevancy_threshold = st.slider(
            "Relevancy Threshold",
            min_value=0.2,
            max_value=0.8,
            value=relevancy_threshold,
            step=0.05,
            help="Documents below this threshold won't be shown"
        )
    with col2:
        primary_threshold = st.slider(
            "Primary Document Threshold",
            min_value=0.3,
            max_value=0.9,
            value=primary_threshold,
            step=0.05,
            help="Documents above this are marked as PRIMARY"
        )
    
    # Show matching method used
    if hasattr(st.session_state, 'doc_embeddings_data') and st.session_state.doc_embeddings_data:
        # Check if checklist has descriptions
        has_descriptions = False
        if st.session_state.get('checklist'):
            for category in st.session_state.checklist.values():
                for item in category.get('items', []):
                    if item.get('description'):
                        has_descriptions = True
                        break
                if has_descriptions:
                    break
        
        if has_descriptions:
            st.success("🤖 Results generated using AI-enhanced matching with document summaries and LLM-generated checklist descriptions")
        else:
            st.success("🤖 Results generated using AI-enhanced matching with document summaries")
    else:
        st.info("📊 Results generated using traditional embedding matching")
    
    # Overall progress metrics
    total_items = sum(cat['total_items'] for cat in checklist_results.values())
    matched_items = sum(cat['matched_items'] for cat in checklist_results.values())
    
    render_metrics_row({
        "Total Items": total_items,
        "Matched Items": matched_items,
        "Coverage": f"{matched_items/total_items*100:.1f}%"
    })
    
    # Display each category
    for letter, category in checklist_results.items():
        with st.expander(
            f"**{letter}. {category['name']}** "
            f"({category['matched_items']}/{category['total_items']} items matched)",
            expanded=False
        ):
            # Category progress bar
            progress = category['matched_items'] / category['total_items'] if category['total_items'] > 0 else 0
            st.progress(progress)
            
            # Display each item
            for idx, item in enumerate(category['items'], 1):
                render_checklist_item(item, idx, relevancy_threshold, primary_threshold)


def render_checklist_item(item: Dict, idx: int, relevancy_threshold: float, primary_threshold: float) -> None:
    """
    Render a single checklist item with its matches
    
    Args:
        item: Checklist item data
        idx: Item index
        relevancy_threshold: Minimum relevancy threshold
        primary_threshold: Threshold for primary documents
    """
    if item['matches']:
        st.markdown(f"✅ **{idx}. {item['text']}**")
        
        # Show AI-generated description if available
        if item.get('description'):
            with st.expander("🤖 AI Description", expanded=False):
                st.info(item['description'])
        
        # Sort matches by score and apply relevancy threshold
        sorted_matches = sorted(item['matches'], key=lambda x: x['score'], reverse=True)
        relevant_matches = [m for m in sorted_matches if m['score'] >= relevancy_threshold]
        
        if relevant_matches:
            for match in relevant_matches:
                render_document_match(match, idx, primary_threshold)
        else:
            st.caption("   Documents found but below relevancy threshold")
    else:
        st.markdown(f"❌ **{idx}. {item['text']}**")
        
        # Show AI-generated description even for unmatched items
        if item.get('description'):
            with st.expander("🤖 AI Description", expanded=False):
                st.info(item['description'])
        
        st.caption("   No matching documents found")


def render_document_match(match: Dict, item_idx: int, primary_threshold: float) -> None:
    """
    Render a single document match
    
    Args:
        match: Document match data
        item_idx: Item index for unique keys
        primary_threshold: Threshold for primary documents
    """
    # Get document title (use name without extension)
    doc_name = match.get('name', match.get('path', 'Unknown'))
    if '.' in doc_name:
        doc_title = doc_name.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
    else:
        doc_title = doc_name.replace('_', ' ').replace('-', ' ').title()
    
    # Compact display with columns
    col1, col2, col3 = st.columns([0.8, 3.5, 0.5])
    
    with col1:
        # Tag without container
        if match['score'] >= primary_threshold:
            st.markdown("🔹 PRIMARY")
        else:
            st.markdown("🔸 ANCILLARY")
    
    with col2:
        # Document title without bold and more compact
        st.write(f"📄 {doc_title}")
    
    with col3:
        # Download button
        render_download_button(match, item_idx, doc_name, doc_title)


def render_download_button(match: Dict, item_idx: int, doc_name: str, doc_title: str) -> None:
    """
    Render download button for a document
    
    Args:
        match: Document match data
        item_idx: Item index for unique keys
        doc_name: Document name
        doc_title: Document title for display
    """
    full_path = match.get('full_path', match.get('path', ''))
    if full_path:
        try:
            file_path = Path(full_path)
            if not file_path.is_absolute():
                file_path = Path("data") / file_path
            
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                
                button_key = f"dl_{item_idx}_{match['score']:.0f}_{doc_name[:20]}".replace(" ", "_").replace("/", "_").replace(".", "_")
                
                st.download_button(
                    label="📥",
                    data=file_bytes,
                    file_name=doc_name,
                    key=button_key,
                    help=f"Download {doc_title}"
                )
        except Exception:
            pass


def render_question_results(question_answers: Dict) -> None:
    """
    Render question answering results
    
    Args:
        question_answers: Results from question answering
    """
    if not question_answers:
        st.info("👈 Configure and process data room to see question answers")
        return
    
    # Group questions by category
    categories = {}
    for answer_data in question_answers.values():
        category = answer_data['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(answer_data)
    
    # Display overall stats
    total_questions = len(question_answers)
    answered_questions = sum(1 for a in question_answers.values() if a['has_answer'])
    
    render_metrics_row({
        "Total Questions": total_questions,
        "Answered": answered_questions,
        "Coverage": f"{answered_questions/total_questions*100:.1f}%"
    })
    
    # Display questions by category
    for category, questions in sorted(categories.items()):
        answered_in_cat = sum(1 for q in questions if q['has_answer'])
        with st.expander(
            f"**{category}** ({answered_in_cat}/{len(questions)} answered)",
            expanded=False
        ):
            for idx, answer_data in enumerate(questions, 1):
                render_question_answer(answer_data, idx)


def render_question_answer(answer_data: Dict, idx: int) -> None:
    """
    Render a single question and its answer
    
    Args:
        answer_data: Question answer data
        idx: Question index
    """
    question = answer_data['question']
    chunks = answer_data['chunks']
    
    if chunks:
        st.markdown(f"✅ **{idx}. {question}**")
        
        # Show relevant documents with relevance scores - compact style
        for chunk_idx, chunk in enumerate(chunks[:3], 1):  # Show top 3 sources
            render_question_source(chunk, chunk_idx, question)
        
        # AI answer generation button
        render_ai_answer_button(answer_data, chunks)
    else:
        st.markdown(f"❌ **{idx}. {question}**")
        st.caption("No relevant documents found")


def render_question_source(chunk: Dict, chunk_idx: int, question: str) -> None:
    """
    Render a source document for a question
    
    Args:
        chunk: Document chunk data
        chunk_idx: Chunk index
        question: Original question
    """
    # Compact display with columns
    col1, col2, col3 = st.columns([0.8, 3.5, 0.5])
    
    with col1:
        # Relevance indicator
        if chunk['score'] >= 0.5:
            st.markdown("🔹 PRIMARY")
        else:
            st.markdown("🔸 ANCILLARY")
    
    with col2:
        # Get clean document title
        doc_name = chunk['source']
        if '.' in doc_name:
            doc_title = doc_name.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
        else:
            doc_title = doc_name.replace('_', ' ').replace('-', ' ').title()
        
        # Document title without bold
        st.write(f"📄 {doc_title}")
    
    with col3:
        # Add download button for the source document
        doc_path = chunk.get('path', '')
        if doc_path:
            try:
                file_path = Path(doc_path)
                if not file_path.is_absolute():
                    file_path = Path("data") / file_path
                
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                    
                    button_key = f"qa_dl_{question[:20]}_{chunk_idx}".replace(" ", "_").replace("?", "").replace("/", "_")
                    
                    st.download_button(
                        label="📥",
                        data=file_bytes,
                        file_name=chunk['source'],
                        key=button_key,
                        help=f"Download {chunk['source']}"
                    )
            except Exception:
                pass


def render_ai_answer_button(answer_data: Dict, chunks: List[Dict]) -> None:
    """
    Render AI answer generation button
    
    Args:
        answer_data: Question answer data
        chunks: Document chunks for context
    """
    # Use AI to generate comprehensive answer if agent is available
    if hasattr(st.session_state, 'agent') and st.session_state.agent and hasattr(st.session_state.agent, 'llm'):
        with st.container():
            if st.button(f"🤖 Generate AI Answer", key=f"ai_answer_{answer_data['question'][:50]}"):
                with st.spinner("AI generating comprehensive answer..."):
                    # Combine chunk texts for context
                    context = "\n\n".join([f"From {c['source']}: {c['text']}" for c in chunks[:3]])
                    # Use LLM directly for more reliable answers
                    from langchain_core.messages import HumanMessage
                    from src.document_processing import escape_markdown_math
                    
                    prompt = f"Question: {answer_data['question']}\n\nContext from documents:\n{context}\n\nProvide a comprehensive answer based on the context."
                    response = st.session_state.agent.llm.invoke([HumanMessage(content=prompt)])
                    # Clean up any leading whitespace and escape math characters
                    cleaned_response = escape_markdown_math(response.content.strip())
                    st.info(cleaned_response)


def render_quick_questions() -> Optional[str]:
    """
    Render quick question buttons for Q&A
    
    Returns:
        Selected question or None
    """
    st.markdown("#### 🔍 Quick Questions")
    
    question = None
    
    # Financial questions
    st.markdown("**Financial & Performance**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📊 Financial Status", use_container_width=True):
            question = "What is the company's financial status and key metrics?"
    with col2:
        if st.button("📈 Revenue", use_container_width=True):
            question = "What is the company's revenue and growth trends?"
    with col3:
        if st.button("💰 Profitability", use_container_width=True):
            question = "What is the company's profitability and margins?"
    with col4:
        if st.button("💸 Cash Flow", use_container_width=True):
            question = "What is the company's cash flow situation?"
    
    # Legal & Compliance
    st.markdown("**Legal & Compliance**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("⚖️ Legal Issues", use_container_width=True):
            question = "Are there any legal issues, litigation, or disputes?"
    with col2:
        if st.button("📜 Contracts", use_container_width=True):
            question = "What are the key contracts and agreements?"
    with col3:
        if st.button("🔒 IP Rights", use_container_width=True):
            question = "What intellectual property does the company own?"
    with col4:
        if st.button("✅ Compliance", use_container_width=True):
            question = "What are the regulatory compliance requirements?"
    
    # Business & Operations
    st.markdown("**Business & Operations**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🎯 Business Model", use_container_width=True):
            question = "What is the company's business model?"
    with col2:
        if st.button("👥 Customers", use_container_width=True):
            question = "Who are the main customers and what is customer concentration?"
    with col3:
        if st.button("🏭 Operations", use_container_width=True):
            question = "What are the key operational processes and capabilities?"
    with col4:
        if st.button("🤝 Partnerships", use_container_width=True):
            question = "What are the key partnerships and strategic relationships?"
    
    # Risk & Strategy
    st.markdown("**Risk & Strategy**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("⚠️ Key Risks", use_container_width=True):
            question = "What are the main risks and challenges?"
    with col2:
        if st.button("🎲 Competition", use_container_width=True):
            question = "Who are the competitors and what is the competitive position?"
    with col3:
        if st.button("📅 Strategy", use_container_width=True):
            question = "What is the company's strategy and future plans?"
    with col4:
        if st.button("🚀 Growth", use_container_width=True):
            question = "What are the growth opportunities and expansion plans?"
    
    return question

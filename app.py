#!/usr/bin/env python3
"""DD-Checklist Complete - RAG + Data Room Processing + Checklist Matching"""
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import io
import re
from pathlib import Path
from typing import Dict, List, Tuple
import os

# Load environment variables from .env file
from dotenv import load_dotenv

def escape_markdown_math(text: str) -> str:
    """Escape dollar signs and other LaTeX-like patterns to prevent Streamlit from interpreting them as math."""
    if not text:
        return text
    # Replace dollar signs with escaped version
    text = text.replace('$', '\\$')
    # Also escape other potential math delimiters
    text = text.replace('\\(', '\\\\(')
    text = text.replace('\\)', '\\\\)')
    text = text.replace('\\[', '\\\\[')
    text = text.replace('\\]', '\\\\]')
    return text
load_dotenv()

# Import LangGraph + Anthropic configuration
try:
    from langgraph_config import (
        DDChecklistAgent,
        LANGGRAPH_AVAILABLE,
        batch_summarize_documents,
        create_document_embeddings_with_summaries,
        match_checklist_with_summaries
    )
    LLM_AVAILABLE = LANGGRAPH_AVAILABLE
except ImportError:
    LLM_AVAILABLE = False
    DDChecklistAgent = None

st.set_page_config(page_title="AI Due Diligence", page_icon="🤖", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_file(file_path: Path) -> Tuple[str, Dict]:
    """Extract text from file with metadata"""
    metadata = {'pages': [], 'type': 'unknown'}
    text_content = ""
    
    try:
        if file_path.suffix.lower() == '.pdf':
            with open(file_path, 'rb') as file:
                pdf = PyPDF2.PdfReader(file)
                texts = []
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        texts.append(page_text)
                        metadata['pages'].append(i)
                text_content = '\n'.join(texts)[:10000]
                metadata['type'] = 'pdf'
                
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            doc = docx.Document(str(file_path))
            text_content = '\n'.join(p.text for p in doc.paragraphs)[:10000]
            metadata['type'] = 'docx'
            
        elif file_path.suffix.lower() in ['.txt', '.md']:
            text_content = file_path.read_text(encoding='utf-8', errors='ignore')[:10000]
            metadata['type'] = 'text'
    except Exception as e:
        st.warning(f"Could not read {file_path.name}: {e}")
        
    return text_content, metadata

def scan_data_room(data_room_path: str) -> Dict[str, Dict]:
    """Scan entire data room directory"""
    documents = {}
    path = Path(data_room_path)
    
    if not path.exists():
        return documents
    
    # Scan all files recursively
    for file_path in path.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            if file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt', '.md']:
                text, metadata = extract_text_from_file(file_path)
                if text:
                    # Store relative path for display
                    rel_path = file_path.relative_to(path)
                    documents[str(file_path)] = {
                        'text': text,
                        'name': file_path.name,
                        'rel_path': str(rel_path),
                        'metadata': metadata
                    }
    return documents

def create_chunks_with_metadata(documents: Dict[str, Dict]) -> List[Dict]:
    """Create searchable chunks with full metadata"""
    chunks = []
    
    for doc_path, doc_info in documents.items():
        text = doc_info['text']
        words = text.split()
        
        # Create overlapping chunks
        chunk_size = 400
        overlap = 50
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = ' '.join(words[i:i + chunk_size])
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'source': doc_info['name'],
                    'path': doc_info['rel_path'],
                    'full_path': doc_path,
                    'chunk_id': f"chunk_{i}",
                    'metadata': doc_info['metadata']
                })
    
    return chunks

def answer_questions_with_chunks(questions: List[Dict], chunks: List[Dict], embeddings: np.ndarray,
                                 model, threshold: float = 0.4) -> Dict:
    """Answer questions using document chunks with citations"""
    answers = {}
    
    for question in questions:
        # Encode question
        question_embedding = model.encode(question['question'])
        
        # Find matching chunks
        similarities = np.dot(embeddings, question_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding)
        )
        
        # Get top matching chunks
        top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5 chunks
        relevant_chunks = []
        
        for idx in top_indices:
            if similarities[idx] > threshold:
                chunk_info = chunks[idx]
                relevant_chunks.append({
                    'text': chunk_info['text'][:500],  # Limit text length
                    'source': chunk_info['source'],
                    'path': chunk_info['path'],
                    'score': float(similarities[idx]),
                    'metadata': chunk_info.get('metadata', {})
                })
        
        answers[question['id']] = {
            'question': question['question'],
            'category': question['category'],
            'chunks': relevant_chunks,
            'has_answer': len(relevant_chunks) > 0
        }
    
    return answers

def parse_questions(questions_text: str) -> List[Dict]:
    """Parse markdown questions into a list of questions with categories"""
    questions = []
    current_category = None
    
    for line in questions_text.split('\n'):
        # Category header (e.g., "### A. Organizational and Corporate Documents")
        if line.startswith('### '):
            match = re.match(r'### ([A-Z])\. (.+)', line)
            if match:
                letter, name = match.groups()
                current_category = f"{letter}. {name.strip()}"
        # Question lines (numbered items)
        elif current_category and line.strip():
            match = re.match(r'^\d+\.\s+(.+)', line.strip())
            if match:
                question_text = match.group(1).strip()
                if question_text:
                    questions.append({
                        'category': current_category,
                        'question': question_text,
                        'id': f"q_{len(questions)}"
                    })
    
    return questions

def parse_checklist(checklist_text: str) -> Dict:
    """Parse markdown checklist into categories and items"""
    categories = {}
    current_category = None
    
    for line in checklist_text.split('\n'):
        # Category header (e.g., "A. Corporate Organization" or "## A. Corporate Organization")
        # Try both formats
        match = None
        if line.startswith('## '):
            match = re.match(r'## ([A-Z])\. (.+)', line)
        elif line.strip() and not line.startswith('\t') and not line.startswith(' '):
            # Try plain format (no ##)
            match = re.match(r'^([A-Z])\. (.+)', line.strip())
        
        if match:
            letter, name = match.groups()
            current_category = letter
            categories[letter] = {
                'name': name.strip(),
                'items': []
            }
        # Numbered items (may be indented with tabs or spaces)
        elif current_category:
            # Check for numbered items with various indentation
            line_stripped = line.strip()
            if re.match(r'^\d+\.', line_stripped):
                item_text = re.sub(r'^\d+\.\s*', '', line_stripped)
                if item_text:
                    # Clean up [bracketed] content but keep the text
                    clean_text = re.sub(r'\[.*?\]', '', item_text).strip()
                    if not clean_text:
                        clean_text = item_text
                    categories[current_category]['items'].append({
                        'text': clean_text,
                        'original': item_text
                    })
    
    return categories

def match_checklist_to_documents(checklist: Dict, chunks: List[Dict], embeddings: np.ndarray, 
                                 model, threshold: float = 0.35) -> Dict:
    """Match each checklist item to relevant documents"""
    results = {}
    
    for cat_letter, category in checklist.items():
        cat_results = {
            'name': category['name'],
            'items': [],
            'total_items': len(category['items']),
            'matched_items': 0
        }
        
        for item_idx, item in enumerate(category['items']):
            # Encode checklist item with category context
            item_text = f"{category['name']} {item['text']}"
            item_embedding = model.encode(item_text)
            
            # Find matching chunks
            similarities = np.dot(embeddings, item_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(item_embedding)
            )
            
            # Get unique documents that match
            doc_matches = {}
            for idx, sim in enumerate(similarities):
                if sim > threshold:
                    doc_path = chunks[idx]['path']
                    if doc_path not in doc_matches or sim > doc_matches[doc_path]['score']:
                        doc_matches[doc_path] = {
                            'name': chunks[idx]['source'],
                            'path': doc_path,
                            'full_path': chunks[idx].get('full_path', doc_path),
                            'score': float(sim),
                            'metadata': chunks[idx]['metadata']
                        }
            
            # Sort by score (no limit, let the UI filter by threshold)
            sorted_matches = sorted(doc_matches.values(), key=lambda x: x['score'], reverse=True)
            
            item_result = {
                'text': item['text'],
                'original': item['original'],
                'matches': sorted_matches
            }
            
            if sorted_matches:
                cat_results['matched_items'] += 1
            
            cat_results['items'].append(item_result)
        
        results[cat_letter] = cat_results
    
    return results

def search_with_citations(query: str, chunks: List[Dict], embeddings: np.ndarray, 
                         model, top_k: int = 5) -> List[Dict]:
    """Search documents and return with citations"""
    if not chunks:
        return []
    
    query_embedding = model.encode(query)
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = []
    seen_texts = set()
    
    for idx in top_indices:
        if similarities[idx] > 0.3:
            # Avoid duplicates
            text_preview = chunks[idx]['text'][:100]
            if text_preview not in seen_texts:
                seen_texts.add(text_preview)
                
                # Format citation based on file type
                metadata = chunks[idx]['metadata']
                if metadata['type'] == 'pdf' and metadata.get('pages'):
                    citation = f"page {metadata['pages'][0]}"
                else:
                    citation = "document"
                
                results.append({
                    'text': chunks[idx]['text'],
                    'source': chunks[idx]['source'],
                    'path': chunks[idx]['path'],
                    'citation': citation,
                    'score': float(similarities[idx])
                })
    
    return results

# MAIN APP
st.title("🤖 AI Due Diligence")
st.markdown("**Intelligent M&A Analysis:** Strategic assessment, automated document review, and AI-powered insights")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'checklist' not in st.session_state:
    st.session_state.checklist = {}
if 'checklist_results' not in st.session_state:
    st.session_state.checklist_results = {}
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'question_answers' not in st.session_state:
    st.session_state.question_answers = {}
if 'strategy_text' not in st.session_state:
    st.session_state.strategy_text = ""
if 'strategy_analysis' not in st.session_state:
    st.session_state.strategy_analysis = ""

# Sidebar configuration
with st.sidebar:
    # Project selector at the top
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
        
        # Data room selector (filtered by selected project)
        st.subheader("📁 Select Data Room")
        
        # Scan for data rooms within the selected project
        data_rooms = []
        project_path = Path(selected_project['path'])
        
        for data_room_dir in project_path.iterdir():
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
                help=f"Found {len(data_rooms)} data rooms in {selected_project['name']}",
                key="dataroom_selector"
            )
            data_room_path = data_rooms[selected_room_idx]['path']
        else:
            st.warning(f"No data rooms found in project '{selected_project['name']}'")
            data_room_path = st.text_input(
                "Data Room Path (manual):",
                value=f"{selected_project['path']}/company",
                help="Enter the path to your data room directory"
            )
    else:
        # No projects found - fall back to manual entry
        st.warning("No projects found in the data directory.")
        st.info("Projects should be folders in 'data/' containing company data rooms")
        data_room_path = st.text_input(
            "Data Room Path (manual):",
            value="data/sample-project/sample-company",
            help="Enter the path to your data room directory"
        )
    
    # Process button after project/data room selection
    process_button = st.button("🚀 Process Data Room", type="primary", use_container_width=True)
    
    if process_button:
        st.success("Processing... Check main area for progress")
    
    # Initialize variables for processing (will be loaded from tabs if available)
    checklist_text = ""
    questions_text = ""
    
    st.divider()
    
    # LLM Configuration
    st.subheader("🤖 AI Enhancement Settings")
    
    # Single toggle for AI features
    use_ai_features = st.checkbox(
        "Enable AI Features", 
        value=True, 
        help="Enable Claude AI for document summaries, intelligent matching, and enhanced Q&A"
    )
    
    if use_ai_features:
        if LLM_AVAILABLE and DDChecklistAgent:
            # Check if API key is in environment
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
            
            # Initialize agent
            if api_key or os.getenv('ANTHROPIC_API_KEY'):
                with st.spinner("Initializing AI agent..."):
                    agent = DDChecklistAgent(api_key, model_choice)
                    
                    if agent.is_available():
                        st.success("✅ AI Agent ready")
                        st.session_state.agent = agent
                        
                        # Show agent capabilities
                        with st.expander("Agent Capabilities"):
                            st.markdown("""
                            - **Smart Parsing**: Understands complex checklist formats
                            - **Document Summaries**: Creates brief AI summaries for each document
                            - **Intelligent Matching**: Uses AI summaries for better categorization
                            - **Enhanced Q&A**: Contextual answers with citations
                            - **Executive Summaries**: Generates insights and gap analysis
                            """)
                    else:
                        st.error("❌ Failed to initialize agent")
                        st.session_state.agent = None
            else:
                st.warning("⚠️ Enter your Anthropic API key above to enable AI features")
                st.session_state.agent = None
        else:
            st.error("❌ AI packages not installed")
            st.caption("Check requirements.txt and ensure langchain-anthropic and langgraph are installed")
            st.session_state.agent = None
    else:
        st.info("🔧 AI features disabled - using traditional embedding-based matching")
        st.caption("Enable the toggle above to use AI-powered enhancements")
        st.session_state.agent = None

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Summary & Analysis", "📊 Checklist Matching", "❓ Due Diligence Questions", "💬 Q&A with Citations"])

# Track if we just completed processing
if 'just_processed' not in st.session_state:
    st.session_state.just_processed = False

# Store process_button state for use in processing section
process_triggered = process_button

# Reset the just_processed flag if we're not processing
if not process_triggered and st.session_state.just_processed:
    st.session_state.just_processed = False

with tab1:
    # Strategy selector at the top of the tab
    # Scan for available strategy files
    strategy_files = []
    strategy_path = Path("data/strategy")
    
    if strategy_path.exists():
        for strategy_file in strategy_path.glob("*.md"):
            if not strategy_file.name.startswith('.'):
                strategy_files.append({
                    'name': strategy_file.stem.replace('_', ' ').replace('-', ' ').title(),
                    'path': str(strategy_file),
                    'filename': strategy_file.name
                })
    
    if strategy_files:
        strategy_files.sort(key=lambda x: x['name'])
        strategy_names = [s['name'] for s in strategy_files]
        selected_strategy_idx = st.selectbox(
            "🎯 Select Strategy:",
            range(len(strategy_names)),
            format_func=lambda x: strategy_names[x],
            help=f"Found {len(strategy_files)} strategy files",
            key="strategy_selector_tab"
        )
        selected_strategy = strategy_files[selected_strategy_idx]
        strategy_text = Path(selected_strategy['path']).read_text(encoding='utf-8')
        st.session_state.strategy_text = strategy_text
    else:
        st.info("No strategy files found in data/strategy/")
        st.session_state.strategy_text = ""
    
    # Check if we have documents to display summaries
    if 'documents' in st.session_state and st.session_state.documents:
        # Company Summary (AI-powered)
        if hasattr(st.session_state, 'agent') and st.session_state.agent and hasattr(st.session_state.agent, 'llm'):
            st.subheader("🏢 Company Overview")
            
            # Check if we already have a company summary in session state
            if 'company_summary' not in st.session_state:
                st.session_state.company_summary = ""
            
            # Auto-generate summary if not already present
            if not st.session_state.company_summary and st.session_state.documents:
                with st.spinner("🤖 Generating company overview..."):
                    # Gather key information from documents
                    doc_summaries = []
                    for path, doc_info in list(st.session_state.documents.items())[:10]:  # Use top 10 docs
                        if 'summary' in doc_info:
                            doc_summaries.append(f"{doc_info['name']}: {doc_info['summary']}")
                        else:
                            # Use first 500 chars of content if no summary
                            content_preview = doc_info.get('content', '')[:500]
                            if content_preview:
                                doc_summaries.append(f"{doc_info['name']}: {content_preview}")
                    
                    if doc_summaries:
                        # Create prompt for company summary
                        from langchain_core.messages import HumanMessage
                        prompt = f"""Based on the following document summaries from a due diligence data room, provide a comprehensive company overview.
                        
                        Company: {Path(list(st.session_state.documents.keys())[0]).parent.name if st.session_state.documents else 'Unknown'}
                        
                        Document Summaries:
                        {chr(10).join(doc_summaries[:10])}
                        
                        Please provide:
                        1. Company name and industry
                        2. Business model and key products/services
                        3. Market position and competitive advantages
                        4. Key financials (if available)
                        5. Organizational structure
                        6. Notable risks or concerns
                        7. Overall assessment for M&A consideration
                        
                        Format the response in clear sections with bullet points where appropriate."""
                        
                        try:
                            response = st.session_state.agent.llm.invoke([HumanMessage(content=prompt)])
                            st.session_state.company_summary = escape_markdown_math(response.content.strip())
                        except Exception as e:
                            st.error(f"Failed to generate company summary: {str(e)}")
            
            # Display the company summary if available
            if st.session_state.company_summary:
                st.info(st.session_state.company_summary)
                
                # Add export button for company summary
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.download_button(
                        "📥 Export Summary",
                        data=f"# Company Overview\n\n{st.session_state.company_summary}",
                        file_name=f"company_overview_{Path(list(st.session_state.documents.keys())[0]).parent.name if st.session_state.documents else 'export'}.md",
                        mime="text/markdown"
                    )
                with col2:
                    if st.button("🔄 Regenerate Overview"):
                        st.session_state.company_summary = ""
                        st.rerun()
        
        # Strategic Analysis (AI-powered)
        if hasattr(st.session_state, 'agent') and st.session_state.agent and hasattr(st.session_state.agent, 'llm'):
            st.subheader("🎯 Strategic Analysis")
            
            # Initialize strategy_analysis if not exists
            if 'strategy_analysis' not in st.session_state:
                st.session_state.strategy_analysis = ""
            
            # Auto-generate analysis if not already present and we have checklist results
            if not st.session_state.strategy_analysis and st.session_state.checklist_results:
                with st.spinner("🤖 Generating strategic analysis..."):
                    # Build context from checklist results
                    checklist_context = []
                    for cat_id, cat_data in st.session_state.checklist_results.items():
                        cat_name = cat_data['name']
                        matched_items = sum(1 for item in cat_data['items'] if item['matches'])
                        total_items = len(cat_data['items'])
                        coverage = (matched_items / total_items * 100) if total_items > 0 else 0
                        
                        checklist_context.append(f"- {cat_name}: {coverage:.0f}% coverage ({matched_items}/{total_items} items)")
                        
                        # Add details about specific gaps
                        missing_items = [item['text'] for item in cat_data['items'] if not item['matches']]
                        if missing_items and len(missing_items) <= 3:
                            checklist_context.append(f"  Missing: {', '.join(missing_items[:3])}")
                    
                    # Build prompt
                    prompt = f"""Based on the due diligence checklist results and the selected strategy, provide a strategic analysis.
                    
                    Strategy Document:
                    {st.session_state.strategy_text}
                    
                    Checklist Coverage:
                    {chr(10).join(checklist_context)}
                    
                    Please provide:
                    1. Strategic alignment assessment
                    2. Key risks and gaps identified
                    3. Opportunities and synergies
                    4. Recommended next steps
                    5. Overall recommendation
                    
                    Format the response with clear sections and bullet points."""
                    
                    try:
                        # Use the LLM directly
                        if hasattr(st.session_state.agent, 'llm') and st.session_state.agent.llm:
                            from langchain_core.messages import HumanMessage
                            response = st.session_state.agent.llm.invoke([HumanMessage(content=prompt)])
                            st.session_state.strategy_analysis = escape_markdown_math(response.content.strip())
                    except Exception as e:
                        st.error(f"Failed to generate strategic analysis: {str(e)}")
            
            # Display warning if no checklist results yet
            if not st.session_state.checklist_results:
                st.warning("⚠️ Process data room with checklist first to enable strategic analysis")
            elif st.session_state.strategy_analysis:
                st.info(st.session_state.strategy_analysis)
                
                # Add export buttons
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    # Combine both summaries for export
                    combined_report = f"# Due Diligence Report\n\n"
                    combined_report += f"## Company Overview\n\n{st.session_state.company_summary}\n\n"
                    combined_report += f"## Strategic Analysis\n\n"
                    combined_report += st.session_state.strategy_analysis
                    
                    st.download_button(
                        "📥 Export Report",
                        data=combined_report,
                        file_name=f"dd_report_{Path(list(st.session_state.documents.keys())[0]).parent.name if st.session_state.documents else 'export'}.md",
                        mime="text/markdown"
                    )
                with col2:
                    if st.button("🔄 Regenerate Analysis"):
                        st.session_state.strategy_analysis = ""
                        st.rerun()

    elif not process_triggered:
        st.info("👈 Configure and process data room to see analysis")
    else:
        # Processing was triggered but no documents found yet
        st.info("⏳ Processing data room... Please wait for results to appear.")


with tab2:
    
    # Checklist selector at the top of the tab
    # Scan for available checklists
    checklists = []
    checklist_path = Path("data/checklist")
    
    if checklist_path.exists():
        for checklist_file in checklist_path.glob("*.md"):
            if not checklist_file.name.startswith('.'):
                checklists.append({
                    'name': checklist_file.stem.replace('_', ' ').replace('-', ' ').title(),
                    'path': str(checklist_file),
                    'filename': checklist_file.name
                })
    
    checklist_text = ""
    if checklists:
        checklists.sort(key=lambda x: x['name'])
        checklist_names = [c['name'] for c in checklists]
        selected_checklist_idx = st.selectbox(
            "📋 Select Checklist:",
            range(len(checklist_names)),
            format_func=lambda x: checklist_names[x],
            help=f"Found {len(checklists)} checklists",
            key="checklist_selector_tab"
        )
        selected_checklist = checklists[selected_checklist_idx]
        checklist_text = Path(selected_checklist['path']).read_text(encoding='utf-8')
    else:
        st.error("No checklists found in data/checklist directory")
    
    if st.session_state.checklist_results:
        # Add relevancy threshold control
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            relevancy_threshold = st.slider(
                "Relevancy Threshold",
                min_value=0.2,
                max_value=0.8,
                value=0.4,
                step=0.05,
                help="Documents below this threshold won't be shown"
            )
        with col2:
            primary_threshold = st.slider(
                "Primary Document Threshold",
                min_value=0.3,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Documents above this are marked as PRIMARY"
            )
        # Show matching method used
        if hasattr(st.session_state, 'doc_embeddings_data') and st.session_state.doc_embeddings_data:
            st.success("🤖 Results generated using AI-enhanced matching with document summaries")
        else:
            st.info("📊 Results generated using traditional embedding matching")
        
        # Add executive summary if agent is available
        if hasattr(st.session_state, 'agent') and st.session_state.agent:
            with st.container():
                st.markdown("### 📝 Executive Summary")
                with st.spinner("Agent generating executive summary..."):
                    summary = st.session_state.agent.summarize_findings(st.session_state.checklist_results)
                    st.info(summary)
                st.divider()
        # Overall progress
        total_items = sum(cat['total_items'] for cat in st.session_state.checklist_results.values())
        matched_items = sum(cat['matched_items'] for cat in st.session_state.checklist_results.values())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Items", total_items)
        with col2:
            st.metric("Matched Items", matched_items)
        with col3:
            st.metric("Coverage", f"{matched_items/total_items*100:.1f}%")
        
        # Display each category
        for letter, category in st.session_state.checklist_results.items():
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
                    if item['matches']:
                        st.markdown(f"✅ **{idx}. {item['text']}**")
                        
                        # Sort matches by score and apply relevancy threshold
                        sorted_matches = sorted(item['matches'], key=lambda x: x['score'], reverse=True)
                        
                        # Display all matches above the user-defined threshold
                        relevant_matches = [m for m in sorted_matches if m['score'] >= relevancy_threshold]
                        
                        if relevant_matches:
                            for match in relevant_matches:
                                # Determine if primary or ancillary based on user-defined threshold
                                if match['score'] >= primary_threshold:
                                    tag = "🔹 PRIMARY"
                                    tag_color = "green"
                                else:
                                    tag = "🔸 ANCILLARY"
                                    tag_color = "orange"
                                
                                # Get document title (use name without extension)
                                doc_name = match.get('name', match.get('path', 'Unknown'))
                                if '.' in doc_name:
                                    doc_title = doc_name.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
                                else:
                                    doc_title = doc_name.replace('_', ' ').replace('-', ' ').title()
                                
                                # Get full path for the file link
                                full_path = match.get('full_path', match.get('path', ''))
                                
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
                                    full_path = match.get('full_path', match.get('path', ''))
                                    if full_path:
                                        try:
                                            file_path = Path(full_path)
                                            if not file_path.is_absolute():
                                                file_path = Path("data") / file_path
                                            
                                            if file_path.exists():
                                                with open(file_path, 'rb') as f:
                                                    file_bytes = f.read()
                                                
                                                button_key = f"dl_{idx}_{match['score']:.0f}_{doc_name[:20]}".replace(" ", "_").replace("/", "_").replace(".", "_")
                                                
                                                st.download_button(
                                                    label="📥",
                                                    data=file_bytes,
                                                    file_name=doc_name,
                                                    key=button_key,
                                                    help=f"Download {doc_title}"
                                                )
                                        except Exception:
                                            pass
                                

                        else:
                            st.caption("   Documents found but below relevancy threshold")
                    else:
                        st.markdown(f"❌ **{idx}. {item['text']}**")
                        st.caption("   No matching documents found")
    else:
        st.info("👈 Configure and process data room to see checklist matching results")

with tab3:
    
    # Question list selector at the top of the tab
    # Scan for available question lists
    question_lists = []
    questions_path = Path("data/questions")
    
    if questions_path.exists():
        for question_file in questions_path.glob("*.md"):
            if not question_file.name.startswith('.'):
                question_lists.append({
                    'name': question_file.stem.replace('_', ' ').replace('-', ' ').title(),
                    'path': str(question_file),
                    'filename': question_file.name
                })
    
    questions_text = ""
    if question_lists:
        question_lists.sort(key=lambda x: x['name'])
        question_names = [q['name'] for q in question_lists]
        selected_question_idx = st.selectbox(
            "❓ Select Question List:",
            range(len(question_names)),
            format_func=lambda x: question_names[x],
            help=f"Found {len(question_lists)} question lists",
            key="question_selector_tab"
        )
        selected_question_list = question_lists[selected_question_idx]
        questions_text = Path(selected_question_list['path']).read_text(encoding='utf-8')
    else:
        st.info("No question lists found in data/questions/")
    
    if st.session_state.question_answers:
        # Group questions by category
        categories = {}
        for answer_data in st.session_state.question_answers.values():
            category = answer_data['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(answer_data)
        
        # Display overall stats
        total_questions = len(st.session_state.question_answers)
        answered_questions = sum(1 for a in st.session_state.question_answers.values() if a['has_answer'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", total_questions)
        with col2:
            st.metric("Answered", answered_questions)
        with col3:
            st.metric("Coverage", f"{answered_questions/total_questions*100:.1f}%")
        
        # Display questions by category
        for category, questions in sorted(categories.items()):
            answered_in_cat = sum(1 for q in questions if q['has_answer'])
            with st.expander(
                f"**{category}** ({answered_in_cat}/{len(questions)} answered)",
                expanded=False
            ):
                for idx, answer_data in enumerate(questions, 1):
                    question = answer_data['question']
                    chunks = answer_data['chunks']
                    
                    if chunks:
                        st.markdown(f"✅ **{idx}. {question}**")
                        
                        # Show relevant documents with relevance scores - compact style
                        for chunk_idx, chunk in enumerate(chunks[:3], 1):  # Show top 3 sources
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
                            

                        
                        # Use AI to generate comprehensive answer if agent is available
                        if hasattr(st.session_state, 'agent') and st.session_state.agent and hasattr(st.session_state.agent, 'llm'):
                            with st.container():
                                if st.button(f"🤖 Generate AI Answer", key=f"ai_answer_{answer_data['question'][:50]}"):
                                    with st.spinner("AI generating comprehensive answer..."):
                                        # Combine chunk texts for context
                                        context = "\n\n".join([f"From {c['source']}: {c['text']}" for c in chunks[:3]])
                                        # Use LLM directly for more reliable answers
                                        from langchain_core.messages import HumanMessage
                                        prompt = f"Question: {question}\n\nContext from documents:\n{context}\n\nProvide a comprehensive answer based on the context."
                                        response = st.session_state.agent.llm.invoke([HumanMessage(content=prompt)])
                                        # Clean up any leading whitespace and escape math characters
                                        cleaned_response = escape_markdown_math(response.content.strip())
                                        st.info(cleaned_response)
                    else:
                        st.markdown(f"❌ **{idx}. {question}**")
                        st.caption("No relevant documents found")
    else:
        st.info("👈 Configure and process data room to see question answers")

with tab4:
    
    if st.session_state.chunks:
        # Question input
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main risks? What is the revenue model? Who are the key customers?"
        )
        
        # Sample questions organized by category
        st.markdown("#### 🔍 Quick Questions")
        
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
        
        st.divider()
        
        if question:
            model = load_model()
            results = search_with_citations(
                question, 
                st.session_state.chunks,
                st.session_state.embeddings,
                model
            )
            
            if results:
                # Use agent to synthesize answer if available
                if hasattr(st.session_state, 'agent') and st.session_state.agent and hasattr(st.session_state.agent, 'llm'):
                    st.markdown("### 🤖 AI Agent's Answer")
                    with st.spinner("Agent analyzing documents..."):
                        # Convert results to document format for context
                        context = "\n\n".join([f"From {r['source']}:\n{r['text']}" for r in results[:3]])
                        # Use LLM directly for more reliable answers
                        from langchain_core.messages import HumanMessage
                        prompt = f"Question: {question}\n\nRelevant document excerpts:\n{context}\n\nProvide a comprehensive answer with citations to the sources."
                        response = st.session_state.agent.llm.invoke([HumanMessage(content=prompt)])
                        # Clean up any leading whitespace and escape math characters
                        answer_text = escape_markdown_math(response.content.strip())
                        st.markdown(answer_text)
                    st.divider()
                
                st.markdown("### 📚 Source Documents")
                
                # Generate answer with citations
                for i, result in enumerate(results[:3], 1):
                    with st.container():
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            excerpt = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                            st.markdown(f"{i}. \"{excerpt}\"")
                            st.caption(f"   📄 {result['source']} ({result['citation']})")
                        
                        with col2:
                            # Add download button
                            doc_path = result.get('path', '')
                            if doc_path:
                                try:
                                    file_path = Path(doc_path)
                                    if not file_path.is_absolute():
                                        file_path = Path("data") / file_path
                                    
                                    if file_path.exists():
                                        with open(file_path, 'rb') as f:
                                            file_bytes = f.read()
                                        
                                        button_key = f"qacit_dl_{i}_{question[:20]}".replace(" ", "_").replace("?", "")
                                        
                                        st.download_button(
                                            label="📥 Download",
                                            data=file_bytes,
                                            file_name=result['source'],
                                            key=button_key,
                                            help=f"Download {result['source']}"
                                        )
                                except Exception:
                                    pass
                
                # Expandable full sources with download buttons
                with st.expander("View full source excerpts"):
                    for idx, result in enumerate(results):
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"**{result['source']}**")
                            st.caption(f"Path: {result['path']}")
                            st.text(result['text'][:500])
                        
                        with col2:
                            # Add download button
                            doc_path = result.get('path', '')
                            if doc_path:
                                try:
                                    file_path = Path(doc_path)
                                    if not file_path.is_absolute():
                                        file_path = Path("data") / file_path
                                    
                                    if file_path.exists():
                                        with open(file_path, 'rb') as f:
                                            file_bytes = f.read()
                                        
                                        button_key = f"qaexp_dl_{idx}_{question[:15]}".replace(" ", "_").replace("?", "")
                                        
                                        st.download_button(
                                            label="📥",
                                            data=file_bytes,
                                            file_name=result['source'],
                                            key=button_key,
                                            help=f"Download {result['source']}"
                                        )
                                except Exception:
                                    pass
                        
                        st.divider()
            else:
                st.warning("No relevant information found for your question.")
    else:
        st.info("👈 Process data room first to enable Q&A")

# Show success message if we just completed processing
if st.session_state.just_processed:
    st.success("✅ Data room processing complete! View results in the tabs above.")
    st.session_state.just_processed = False

# Processing section - displays below all tabs
if process_triggered:
    if not Path(data_room_path).exists():
        st.error(f"Data room path not found: {data_room_path}")
    else:
        # Create a container for progress updates
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 🚀 Processing Data Room")
            overall_progress = st.progress(0, text="Initializing...")
            status_text = st.empty()
            
            # Step 1: Load model
            status_text.text("📦 Loading AI model...")
            model = load_model()
            overall_progress.progress(0.1, text="Model loaded")
        
        # Step 2: Scan data room
        status_text.text(f"🔍 Scanning data room: {Path(data_room_path).name}")
        st.session_state.documents = scan_data_room(data_room_path)
        doc_count = len(st.session_state.documents)
        status_text.text(f"✅ Found {doc_count} documents")
        overall_progress.progress(0.2, text=f"Scanned {doc_count} documents")
    
        # Step 3: Generate document summaries if agent is available
        if hasattr(st.session_state, 'agent') and st.session_state.agent and st.session_state.agent.llm:
            status_text.text("🤖 AI generating document summaries...")
            summary_progress = st.progress(0)
            
            # Convert documents to format for summarization
            docs_for_summary = []
            for path, doc_info in st.session_state.documents.items():
                docs_for_summary.append({
                    'name': doc_info['name'],
                    'path': doc_info['rel_path'],
                    'content': doc_info.get('content', '')[:1500],  # First 1500 chars
                    'metadata': doc_info.get('metadata', {})
                })
            
            # Batch summarize documents with progress
            total_docs = len(docs_for_summary)
            
            # Store progress in session state for the batch function to update
            st.session_state.summary_progress = summary_progress
            
            # Use larger batch size for better performance with many documents
            # Anthropic can handle 10-20 concurrent requests well
            batch_size = 10 if total_docs > 100 else 5
            
            summary_progress.progress(0, text=f"🚀 Summarizing {total_docs} documents in batches of {batch_size}...")
            
            summarized_docs = batch_summarize_documents(
                docs_for_summary, 
                st.session_state.agent.llm,
                batch_size=batch_size
            )
            
            # Store summaries back in documents
            for doc in summarized_docs:
                if doc['path'] in st.session_state.documents:
                    st.session_state.documents[doc['path']]['summary'] = doc.get('summary', '')
            
            summary_progress.progress(1.0, text=f"✅ Generated {len(summarized_docs)} summaries")
            overall_progress.progress(0.4, text="AI summaries complete")
            
            # Create embeddings using summaries
            status_text.text("🧮 Creating document embeddings...")
            st.session_state.doc_embeddings_data = create_document_embeddings_with_summaries(
                summarized_docs, model
            )
            overall_progress.progress(0.5, text="Embeddings created")
    
        # Step 4: Create chunks for RAG
        status_text.text("📄 Creating searchable document chunks...")
        chunk_progress = st.progress(0)
        st.session_state.chunks = create_chunks_with_metadata(st.session_state.documents)
        chunk_count = len(st.session_state.chunks)
        chunk_progress.progress(0.5, text=f"Created {chunk_count} chunks")
        
        # Step 5: Create embeddings for chunks
        status_text.text("🧮 Creating search embeddings...")
        texts = [chunk['text'] for chunk in st.session_state.chunks]
        
        # Process embeddings in batches for large datasets
        batch_size = 100
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            chunk_progress.progress(
                0.5 + (0.5 * (i + len(batch)) / len(texts)),
                text=f"Encoding chunks {i+1}-{min(i+batch_size, len(texts))} of {len(texts)}"
            )
            batch_embeddings = model.encode(batch)
            embeddings_list.append(batch_embeddings)
        
        import numpy as np
        st.session_state.embeddings = np.vstack(embeddings_list) if embeddings_list else np.array([])
        chunk_progress.progress(1.0, text=f"✅ Encoded {chunk_count} chunks")
        overall_progress.progress(0.6, text="Search index ready")
    
        # Step 6: Parse checklist (load default if not selected)
        if not checklist_text:
            # Load default checklist
            checklist_path = Path("data/checklist")
            if checklist_path.exists():
                checklist_files = list(checklist_path.glob("*.md"))
                if checklist_files:
                    checklist_text = checklist_files[0].read_text(encoding='utf-8')
        
        if checklist_text:
            status_text.text("📋 Parsing checklist...")
            st.session_state.checklist = parse_checklist(checklist_text)
            
            if st.session_state.checklist:
                category_names = [f"{k}. {v['name']}" for k, v in st.session_state.checklist.items()]
                status_text.text(f"✅ Parsed {len(st.session_state.checklist)} checklist categories")
                overall_progress.progress(0.7, text="Checklist parsed")
            else:
                status_text.warning("⚠️ No categories found in checklist")
        else:
            st.session_state.checklist = {}
            status_text.warning("⚠️ No checklist selected")
    
        # Step 7: Parse questions (load default if not selected)
        if not questions_text:
            # Load default questions
            questions_path = Path("data/questions")
            if questions_path.exists():
                question_files = list(questions_path.glob("*.md"))
                if question_files:
                    questions_text = question_files[0].read_text(encoding='utf-8')
        
        if questions_text:
            status_text.text("❓ Parsing questions...")
            st.session_state.questions = parse_questions(questions_text)
            if st.session_state.questions:
                categories = list(set(q['category'] for q in st.session_state.questions))
                status_text.text(f"✅ Parsed {len(st.session_state.questions)} questions")
                overall_progress.progress(0.75, text="Questions parsed")
    
        # Step 8: Match checklist to documents
        if st.session_state.checklist and st.session_state.chunks:
            if hasattr(st.session_state, 'doc_embeddings_data') and st.session_state.doc_embeddings_data:
                status_text.text("🤖 AI-enhanced matching with document summaries...")
                match_progress = st.progress(0)
                
                # Use summary-based matching
                st.session_state.checklist_results = match_checklist_with_summaries(
                    st.session_state.checklist,
                    st.session_state.doc_embeddings_data,
                    model,
                    0.25  # Use fixed threshold for processing
                )
                match_progress.progress(1.0, text="✅ AI matching complete")
            else:
                status_text.text(f"🔍 Matching checklist items to documents...")
                match_progress = st.progress(0)
                
                st.session_state.checklist_results = match_checklist_to_documents(
                    st.session_state.checklist,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                    model,
                    0.25  # Use fixed threshold for processing
                )
                match_progress.progress(1.0, text="✅ Matching complete")
            
            overall_progress.progress(0.85, text="Checklist matched")
        
        # Step 9: Answer questions if we have them
        if st.session_state.questions and st.session_state.chunks and st.session_state.embeddings is not None:
            status_text.text(f"📝 Answering {len(st.session_state.questions)} due diligence questions...")
            qa_progress = st.progress(0)
            
            st.session_state.question_answers = answer_questions_with_chunks(
                st.session_state.questions,
                st.session_state.chunks,
                st.session_state.embeddings,
                model,
                0.25  # Use fixed threshold for processing
            )
            answered_count = sum(1 for a in st.session_state.question_answers.values() if a['has_answer'])
            qa_progress.progress(1.0, text=f"✅ Answered {answered_count}/{len(st.session_state.questions)} questions")
            overall_progress.progress(0.95, text="Questions answered")
        
        # Step 10: Generate company summary if AI is available
        if hasattr(st.session_state, 'agent') and st.session_state.agent and hasattr(st.session_state.agent, 'llm'):
            status_text.text("🏢 Generating company overview...")
            
            # Gather key information from documents
            doc_summaries = []
            for path, doc_info in list(st.session_state.documents.items())[:10]:  # Use top 10 docs
                if 'summary' in doc_info:
                    doc_summaries.append(f"{doc_info['name']}: {doc_info['summary']}")
                else:
                    content_preview = doc_info.get('content', '')[:500]
                    if content_preview:
                        doc_summaries.append(f"{doc_info['name']}: {content_preview}")
            
            if doc_summaries:
                from langchain_core.messages import HumanMessage
                prompt = f"""Based on the following document summaries from a due diligence data room, provide a comprehensive company overview.
                
                Company: {Path(data_room_path).name}
                
                Document Summaries:
                {chr(10).join(doc_summaries[:10])}
                
                Please provide:
                1. Company name and industry
                2. Business model and key products/services
                3. Market position and competitive advantages
                4. Key financials (if available)
                5. Organizational structure
                6. Notable risks or concerns
                7. Overall assessment for M&A consideration
                
                Format the response in clear sections with bullet points where appropriate."""
                
                response = st.session_state.agent.llm.invoke([HumanMessage(content=prompt)])
                st.session_state.company_summary = escape_markdown_math(response.content.strip())
                overall_progress.progress(0.98, text="Company summary generated")
        
        # Final step
        overall_progress.progress(1.0, text="✅ Processing complete!")
        status_text.success(f"🎉 Successfully processed {doc_count} documents")
        
        # Add a small delay to show completion before clearing
        import time
        time.sleep(1.5)
        
        # Clear the progress container after completion
        progress_container.empty()
        
        # Mark that we just completed processing
        st.session_state.just_processed = True
        
        # Force a rerun to refresh the UI with the new data
        st.rerun()

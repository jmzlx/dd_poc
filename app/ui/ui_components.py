#!/usr/bin/env python3
"""
UI Components Module

This module contains reusable Streamlit UI components and layout functions.
Separates UI logic from business logic for better maintainability.
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Tuple

from app.core import get_config, format_document_title, count_documents_in_directory
from functools import wraps
from typing import Callable, Any


def _resolve_document_path(doc_path: str) -> Optional[Path]:
    """
    Resolve a document path, handling both relative and absolute paths.
    
    Args:
        doc_path: Document path (relative or absolute)
        
    Returns:
        Resolved Path object or None if path cannot be resolved
    """
    if not doc_path:
        return None
        
    path_obj = Path(doc_path)
    
    # If it's already absolute and exists, return it
    if path_obj.is_absolute():
        return path_obj if path_obj.exists() else None
    
    # For relative paths, try to resolve against the data room path
    data_room_path = getattr(st.session_state, 'data_room_path', None)
    if data_room_path:
        resolved_path = Path(data_room_path) / path_obj
        if resolved_path.exists():
            return resolved_path
    
    # Fallback: try relative to data directory
    data_dir = Path('data')
    fallback_path = data_dir / path_obj
    if fallback_path.exists():
        return fallback_path
        
    # Enhanced search: Look in the currently selected data room only
    # This handles cases where files like "company-profile.pdf" are stored with just filename
    # but should only be resolved within the current data room context
    
    # Try using the data room path from session state
    current_data_room = getattr(st.session_state, 'data_room_path', None)
    if current_data_room and Path(current_data_room).exists():
        potential_path = Path(current_data_room) / path_obj
        if potential_path.exists():
            return potential_path
    
    # Also check for selected_data_room_path as fallback
    selected_data_room = getattr(st.session_state, 'selected_data_room_path', None)
    if selected_data_room and Path(selected_data_room).exists():
        potential_path = Path(selected_data_room) / path_obj
        if potential_path.exists():
            return potential_path
    
    # Last resort: check if original path exists as-is
    if path_obj.exists():
        return path_obj
        
    return None


def render_project_selector() -> Tuple[Optional[str], Optional[str]]:
    """
    Render project and data room selector in sidebar

    Returns:
        Tuple of (selected_project_path, data_room_path)
    """
    st.subheader("🗂️ Select Project")

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
                        total_docs = count_documents_in_directory(project_dir)
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
                    total_docs = count_documents_in_directory(project_dir)
                    if total_docs > 0:
                        projects.append({
                            'name': project_dir.name.replace('-', ' ').replace('_', ' ').title(),
                            'path': str(project_dir),
                            'data_rooms': len(subdirs),
                            'total_docs': total_docs
                        })

    selected_project_path = None
    data_room_path = None

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
        data_room_path = render_data_room_selector(selected_project['path'])

    else:
        # No projects found - fall back to manual entry
        st.warning("No projects found in the data directory.")
        st.info("Projects should be folders in 'data/' containing company data rooms")
        data_room_path = st.text_input(
            "Data Room Path (manual):",
            value="data/sample-project/sample-company",
            help="Enter the path to your data room directory",
            key="manual_data_room_path"
        )

    return selected_project_path, data_room_path


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
            doc_count = count_documents_in_directory(data_room_dir)
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
            help="Enter the path to your data room directory",
            key="manual_project_data_room_path"
        )


def render_ai_settings() -> Tuple[Optional[str], str]:
    """
    Render AI settings in sidebar

    Returns:
        Tuple of (api_key, model_choice)
    """
    st.subheader("🤖 AI Settings")

    config = get_config()

    # Only accept API key from .env file
    config_api_key = config.anthropic['api_key']
    if config_api_key:
        api_key = config_api_key
    else:
        st.error("❌ Please set ANTHROPIC_API_KEY in your .env file")
        api_key = None

    # Model selection
    available_models = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-1-20250805",
        "claude-3-5-haiku-20241022"
    ]
    default_index = 0
    if config.model['claude_model'] in available_models:
        default_index = available_models.index(config.model['claude_model'])

    model_choice = st.radio(
        "Claude Model",
        available_models,
        index=default_index,
        help="Sonnet 4: High-performance model (default) | Opus 4.1: Most capable | Haiku 3.5: Fastest and most cost-effective"
    )

    return api_key, model_choice


# UI Components

def action_button(label: str, key: Optional[str] = None, help: Optional[str] = None, type: Optional[str] = None) -> bool:
    """
    Create an action button that returns True when clicked.

    Args:
        label: Button text
        key: Unique key for the button
        help: Help text for the button
        type: Button type (primary, secondary, etc.)

    Returns:
        True if button was clicked, False otherwise
    """
    if type:
        return st.button(label, key=key, help=help, type=type)
    else:
        return st.button(label, key=key, help=help)


def status_message(message: str, message_type: str = "info"):
    """
    Display a status message with appropriate styling.

    Args:
        message: Message to display
        message_type: Type of message ("info", "success", "warning", "error")
    """
    if message_type == "success":
        st.success(message)
    elif message_type == "warning":
        st.warning(message)
    elif message_type == "error":
        st.error(message)
    else:  # info or default
        st.info(message)


def progress_indicator():
    """
    Create a progress indicator placeholder.

    Returns:
        A context manager for progress indication
    """
    return st.empty()


def processing_guard(session_attr: str = "processing_active", message: str = "⚠️ Another operation is currently running. Please wait.") -> Callable:
    """
    Decorator to guard against concurrent processing operations.

    This decorator checks if processing is already active before allowing the decorated
    function to execute. If processing is active, it displays a warning message and
    prevents the function from running.

    Args:
        session_attr: Name of the session attribute to check (default: "processing_active")
        message: Warning message to display if processing is active

    Returns:
        Decorated function that checks processing status before execution

    Example:
        @processing_guard()
        def generate_report(self):
            # This function will only run if processing is not active
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Check if processing is already active
            if getattr(self.session, session_attr, False):
                status_message(message, "warning")
                return None

            # Set processing active
            setattr(self.session, session_attr, True)

            try:
                # Execute the decorated function
                result = func(self, *args, **kwargs)
                return result
            finally:
                # Always reset processing state
                setattr(self.session, session_attr, False)

        return wrapper
    return decorator


def render_file_selector(directory: str, file_type: str, key_suffix: str, icon: str = "📋") -> Tuple[Optional[str], str]:
    """
    Render file selector for checklists, questions, or strategies

    Args:
        directory: Directory to scan for files
        file_type: Type of file (for display)
        key_suffix: Unique suffix for streamlit keys
        icon: Icon to display with the selector (default: 📋)

    Returns:
        Tuple of (selected_file_path, file_content)
    """
    files = []
    dir_path = Path(directory)

    if dir_path.exists():
        for file in dir_path.glob("*.md"):
            if not file.name.startswith('.'):
                files.append({
                    'name': format_document_title(file.stem),
                    'path': str(file),
                    'filename': file.name
                })

    file_content = ""
    selected_file_path = None

    if files:
        files.sort(key=lambda x: x['name'])
        file_names = [f['name'] for f in files]
        selected_file_idx = st.selectbox(
            f"{icon} Select {file_type}:",
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


# =============================================================================
# ERROR HANDLING COMPONENTS - Standardized error message patterns
# =============================================================================

def display_generation_error(operation_type: str, error: Exception = None):
    """
    Display a standardized error message for generation failures.

    Args:
        operation_type: Type of operation that failed (e.g., "question analysis", "checklist analysis")
        error: The exception that occurred (optional)
    """
    if error:
        st.error(f"❌ Failed to generate {operation_type}: {str(error)}")
    else:
        st.error(f"❌ Failed to generate {operation_type}")


def display_processing_error(operation_type: str, error: Exception = None):
    """
    Display a standardized error message for processing failures.

    Args:
        operation_type: Type of operation that failed (e.g., "question", "data room")
        error: The exception that occurred (optional)
    """
    if error:
        st.error(f"❌ Failed to process {operation_type}: {str(error)}")
    else:
        st.error(f"❌ Failed to process {operation_type}")


def display_initialization_error(component_type: str, error: Exception = None):
    """
    Display a standardized error message for initialization failures.

    Args:
        component_type: Type of component that failed to initialize (e.g., "document processor")
        error: The exception that occurred (optional)
    """
    if error:
        st.error(f"❌ Failed to initialize {component_type}: {str(error)}")
    else:
        st.error(f"❌ Failed to initialize {component_type}")


def display_download_error(error: Exception = None):
    """
    Display a standardized error message for download failures.

    Args:
        error: The exception that occurred (optional)
    """
    if error:
        st.error(f"❌ Download failed: {str(error)}")
    else:
        st.error("❌ Download failed")


# =============================================================================
# RESULTS RENDERING COMPONENTS - Display search and analysis results
# =============================================================================

def render_checklist_results(results: dict, relevancy_threshold: float):
    """
    Render checklist matching results in Streamlit UI with nested collapsible elements.

    Args:
        results: Dictionary of checklist results by category
        relevancy_threshold: Threshold for displaying results
    """
    import streamlit as st
    from pathlib import Path

    st.subheader("📊 Checklist Matching Results")

    for cat_letter, category in results.items():
        with st.expander(f"**{cat_letter}. {category['name']}** ({category['matched_items']}/{category['total_items']} items matched)", expanded=False):
            for item_idx, item in enumerate(category['items']):
                item_text = item['text']
                matches = item['matches']

                # Filter matches by relevancy threshold
                relevant_matches = [m for m in matches if m['score'] >= relevancy_threshold]

                # Create a nested expander for each checklist item
                if relevant_matches:
                    # Show item as matched with number of documents found
                    item_status = "✅"
                    item_summary = f"{len(relevant_matches)} document(s) found"
                    expanded_default = False
                else:
                    # Show item as not matched
                    item_status = "❌"
                    item_summary = "No relevant documents found"
                    expanded_default = False

                with st.expander(f"**{item_status} Item {item_idx + 1}:** {item_text} ({item_summary})", expanded=expanded_default):
                    if relevant_matches:
                        for match in relevant_matches:
                            score = match['score']
                            doc_name = match['name']
                            doc_path = match['path']

                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                resolved_path = _resolve_document_path(doc_path)
                                if resolved_path and resolved_path.exists():
                                    try:
                                        with open(resolved_path, 'rb') as f:
                                            st.download_button(
                                                f"📄 {doc_name}",
                                                data=f.read(),
                                                file_name=resolved_path.name,
                                                mime="application/octet-stream",
                                                key=f"download_{hash(doc_path) % 10000}_{item_idx}"
                                            )
                                    except Exception:
                                        st.write(f"📄 {doc_name} (unavailable)")
                                else:
                                    st.write(f"📄 {doc_name} (unavailable)")
                            with col2:
                                st.caption(f"{score:.3f}")
                            with col3:
                                if score >= 0.5:
                                    st.caption("🔹 PRIMARY")
                                else:
                                    st.caption("🔸 ANCILLARY")
                    else:
                        st.info("No documents found matching the relevancy threshold for this checklist item.")


def render_question_results(answers: dict):
    """
    Render question answering results in Streamlit UI.

    Args:
        answers: Dictionary of question answers
    """
    import streamlit as st
    from pathlib import Path

    st.subheader("❓ Question Answers")

    for question_key, answer_data in answers.items():
        question = answer_data.get('question', question_key)
        answer = answer_data.get('answer', 'No answer available')
        sources = answer_data.get('sources', [])

        with st.expander(f"**{question}**", expanded=True):
            if answer:
                st.markdown(f"**Answer:** {answer}")

            if sources:
                st.markdown("**Source Documents:**")
                for i, source in enumerate(sources):
                    doc_name = source.get('name', 'Unknown')
                    doc_path = source.get('path', '')
                    score = source.get('score', 0.0)

                    col1, col2 = st.columns([4, 1])
                    with col1:
                        resolved_path = _resolve_document_path(doc_path)
                        if resolved_path and resolved_path.exists():
                            try:
                                with open(resolved_path, 'rb') as f:
                                    st.download_button(
                                        f"📄 {doc_name}",
                                        data=f.read(),
                                        file_name=resolved_path.name,
                                        mime="application/octet-stream",
                                        key=f"q_download_{hash(doc_path) % 10000}_{i}"
                                    )
                            except Exception:
                                st.write(f"📄 {doc_name} (unavailable)")
                        else:
                            st.write(f"📄 {doc_name} (unavailable)")
                    with col2:
                        st.caption(f"{score:.3f}")


def create_document_link(doc_path: str, doc_name: str, doc_title: str, unique_key: str):
    """
    Create a download link for a document.

    Args:
        doc_path: Path to the document file (can be relative or absolute)
        doc_name: Display name for the document
        doc_title: Title for the document
        unique_key: Unique key for the download button
    """
    import streamlit as st
    from pathlib import Path

    # Resolve the path - handle both relative and absolute paths
    resolved_path = _resolve_document_path(doc_path)
    
    if resolved_path and resolved_path.exists():
        try:
            with open(resolved_path, 'rb') as f:
                st.download_button(
                    f"📄 {doc_title}",
                    data=f.read(),
                    file_name=resolved_path.name,
                    mime="application/octet-stream",
                    key=f"link_{unique_key}"
                )
        except Exception as e:
            st.error(f"Error reading document: {doc_name}")
    else:
        st.write(f"📄 {doc_title} (unavailable)")


# =============================================================================
# GENERATE/REGENERATE BUTTON COMPONENTS - Common 2-column button layout
# =============================================================================

def render_generate_buttons(generate_label: str, regenerate_key: str, session_attr: str,
                          help_text: str, session_manager) -> bool:
    """
    Render a single generate/regenerate button that handles both cases.

    This component creates a single button that changes text based on whether there's existing data.
    If there's existing data, it shows "🔄 Regenerate", otherwise shows the generate_label.

    Args:
        generate_label: Label for the generate button when no data exists (e.g., "🤖 Generate Overview")
        regenerate_key: Unique key for the button
        session_attr: Session attribute name to check for existing data (e.g., "overview_summary")
        help_text: Help text for the button tooltip
        session_manager: Session manager instance to access session data

    Returns:
        Boolean flag indicating if the button was clicked
    """
    # Check if there's existing data
    existing_data = getattr(session_manager, session_attr, None)

    # Determine button text and whether to clear data
    if existing_data:
        button_text = "🔄 Regenerate"
        should_clear_data = True
    else:
        button_text = generate_label
        should_clear_data = False

    # Render single button
    button_clicked = action_button(
        button_text,
        type="primary",
        key=regenerate_key,
        help=help_text
    )

    # Handle regenerate case - clear existing data
    if button_clicked and should_clear_data:
        if isinstance(existing_data, str):
            setattr(session_manager, session_attr, "")
        else:
            setattr(session_manager, session_attr, {})
        st.rerun()

    st.divider()
    return button_clicked


# =============================================================================
# END OF FILE
# =============================================================================
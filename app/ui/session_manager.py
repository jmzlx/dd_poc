#!/usr/bin/env python3
"""
Session State Manager

Manages Streamlit session state with type-safe access.
"""

import streamlit as st
from typing import Any

from app.ui.error_handler import ErrorHandler



class SessionProperty:
    """
    Descriptor for session state properties with type-safe access.

    This descriptor provides a clean interface to Streamlit's session state,
    eliminating repetitive property definitions while maintaining type safety.
    """

    def __init__(self, default_value: Any = None):
        self.default_value = default_value
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return st.session_state.get(self.name, self.default_value)

    def __set__(self, instance, value):
        st.session_state[self.name] = value


class SessionManager:
    """Session state manager with type-safe access to session data."""

    # Document processing state
    documents = SessionProperty({})
    chunks = SessionProperty([])
    embeddings = SessionProperty(None)

    # Analysis results
    checklist_results = SessionProperty({})
    question_answers = SessionProperty({})
    overview_summary = SessionProperty("")
    strategic_summary = SessionProperty("")
    strategic_company_summary = SessionProperty("")
    # Note: Citations are now inline in the strategic_company_summary content

    # User selections
    strategy_path = SessionProperty(None)
    strategy_text = SessionProperty("")
    checklist_path = SessionProperty(None)
    checklist_text = SessionProperty("")
    questions_path = SessionProperty(None)
    questions_text = SessionProperty("")
    vdr_store = SessionProperty(None)
    data_room_path = SessionProperty(None)

    # Processing state
    processing_active = SessionProperty(False)
    agent = SessionProperty(None)

    # Cached data
    checklist = SessionProperty({})
    questions = SessionProperty({})
    analysis_vector_store = SessionProperty(None)
    document_type_embeddings = SessionProperty({})

    def __init__(self) -> None:
        """Initialize session state manager with default values."""
        self._init_defaults()

    def _init_defaults(self) -> None:
        """Initialize default session state values."""
        try:
            # Get all descriptor properties and their defaults
            all_properties = {
                name: getattr(self.__class__, name).default_value
                for name in dir(self.__class__)
                if isinstance(getattr(self.__class__, name), SessionProperty)
            }

            for key, default_value in all_properties.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value

        except Exception as e:
            ErrorHandler.handle_error(
                e,
                "Session initialization failed",
                recovery_hint="Please refresh the page and try again"
            )
            # Initialize with minimal defaults on error
            st.session_state.clear()
            st.session_state.update({
                'documents': {},
                'processing_active': False,
                'agent': None,
            })


    def reset(self) -> None:
        """Reset analysis results and cached data for fresh analysis."""
        self.overview_summary = ""
        self.strategic_summary = ""
        # Note: strategic_company_summary and citations are preserved across document reprocessing
        # They are only cleared when explicitly generating new company analysis
        self.checklist_results = {}
        self.question_answers = {}

    def reset_processing(self) -> None:
        """Reset processing flags to allow new operations."""
        self.processing_active = False

    def ready(self) -> bool:
        """Check if system is ready for analysis operations."""
        return bool(self.documents is not None and len(self.documents) > 0 and not self.processing_active)

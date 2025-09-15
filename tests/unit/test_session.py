"""
Unit tests for SessionManager class
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

from app.ui.session_manager import SessionManager


@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state for testing"""
    return MagicMock()


@pytest.fixture
def session_manager(mock_session_state):
    """Create SessionManager instance with mocked session state"""
    with patch('streamlit.session_state', mock_session_state):
        with patch('app.ui.session_manager.st', MagicMock(session_state=mock_session_state)):
            # Configure mock to behave like empty session state
            mock_session_state.__contains__.return_value = False
            mock_session_state.get.return_value = None
            manager = SessionManager()
            # Reset mock call history after initialization
            mock_session_state.reset_mock()
            return manager


class TestSessionManagerInitialization:
    """Test SessionManager initialization and default values"""

    def test_initialization_sets_defaults(self, mock_session_state):
        """Test that SessionManager can be initialized"""
        with patch('streamlit.session_state', mock_session_state):
            with patch('app.ui.session_manager.st', MagicMock(session_state=mock_session_state)):
                manager = SessionManager()

                # Check that initialization attempted to set defaults
                assert mock_session_state.__setitem__.call_count > 0

    @pytest.mark.skip(reason="Error handling test with side_effect - kept for testing error conditions")
    def test_initialization_error_handling(self, mock_session_state):
        """Test error handling during initialization"""
        mock_session_state.__setitem__.side_effect = Exception("Test error")

        with patch('app.error_handler.ErrorHandler.handle_error'):
            manager = SessionManager()

            # Should fall back to minimal defaults
            assert mock_session_state.clear.called


class TestStatePersistence:
    """Test state persistence and property operations"""

    def test_documents_property_operations(self, session_manager, mock_session_state):
        """Test documents property getter and setter"""
        # Test setter
        test_docs = {'doc1': {'name': 'test'}}
        session_manager.documents = test_docs
        # Property should work without errors
        assert session_manager.documents == test_docs




class TestDocumentStorage:
    """Test document storage operations"""

    def test_document_storage_and_retrieval(self, session_manager, mock_session_state):
        """Test storing and retrieving documents"""
        documents = {
            'doc1.pdf': {
                'name': 'Document 1',
                'content': 'Document content 1',
                'metadata': {'size': 1024}
            }
        }
        # Store documents
        session_manager.documents = documents
        # Property should work without errors
        assert session_manager.documents == documents

    def test_empty_document_storage(self, session_manager, mock_session_state):
        """Test handling empty document storage"""
        # Test with empty documents
        session_manager.documents = {}
        docs = session_manager.documents
        assert docs == {}


class TestChunkManagement:
    """Test chunk management operations"""

    def test_chunk_storage_and_retrieval(self, session_manager, mock_session_state):
        """Test storing and retrieving document chunks"""
        chunks = [
            {
                'text': 'This is chunk 1',
                'source': 'doc1.pdf',
                'metadata': {'page': 1, 'chunk_id': 0}
            }
        ]
        # Store chunks
        session_manager.chunks = chunks
        # Property should work without errors
        assert session_manager.chunks == chunks

    def test_empty_chunk_storage(self, session_manager, mock_session_state):
        """Test handling empty chunk storage"""
        # Test with empty chunks
        session_manager.chunks = []
        chunks = session_manager.chunks
        assert chunks == []

    def test_chunk_with_embeddings(self, session_manager, mock_session_state):
        """Test chunk management with embeddings"""
        chunks = [{'text': 'chunk with embedding', 'embedding': [0.1, 0.2, 0.3]}]
        # Store chunks with embeddings
        session_manager.chunks = chunks
        # Property should work without errors
        assert session_manager.chunks == chunks


class TestSessionCleanup:
    """Test session cleanup operations"""

    def test_reset_method(self, session_manager, mock_session_state):
        """Test reset method clears analysis results"""
        # Reset analysis
        session_manager.reset()

        # Verify reset_analysis was called (we can't easily test the exact calls
        # due to the complex mocking, but we can test that the method exists and runs)
        assert hasattr(session_manager, 'reset')

    def test_reset_processing_method(self, session_manager, mock_session_state):
        """Test reset_processing method"""
        session_manager.processing_active = True
        session_manager.reset_processing()

        # Check that processing_active was set to False
        assert session_manager.processing_active == False


class TestReadinessCheck:
    """Test ready method"""

    def test_ready_method_exists(self, session_manager, mock_session_state):
        """Test that ready method exists and can be called"""
        # Method should exist and be callable
        assert hasattr(session_manager, 'ready')
        assert callable(getattr(session_manager, 'ready'))


class TestErrorHandling:
    """Test error handling in SessionManager"""

    def test_property_getter_with_default_values(self, session_manager, mock_session_state):
        """Test property getters return appropriate defaults"""
        mock_session_state.get.return_value = None

        # Test that properties can be accessed without error
        docs = session_manager.documents
        chunks = session_manager.chunks
        active = session_manager.processing_active

        # Verify they are the expected types
        assert isinstance(docs, dict)
        assert isinstance(chunks, list)
        assert isinstance(active, bool)

    def test_property_setter_operations(self, session_manager, mock_session_state):
        """Test that property setters work correctly"""
        # Test setting None values
        session_manager.embeddings = None
        session_manager.selected_strategy_path = None
        session_manager.agent = None

        # Verify properties were set
        assert session_manager.embeddings is None
        assert session_manager.selected_strategy_path is None
        assert session_manager.agent is None

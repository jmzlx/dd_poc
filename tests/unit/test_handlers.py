"""
Unit tests for handler classes

Tests for AIHandler, DocumentHandler, and ExportHandler classes
"""
import pytest
from unittest.mock import MagicMock, patch

from app.handlers.ai_handler import AIHandler
from app.handlers.document_handler import DocumentHandler
from app.handlers.export_handler import ExportHandler
from app.ui.session_manager import SessionManager
from app.core.exceptions import AIError, ProcessingError


@pytest.fixture
def mock_session():
    """Create a mock session manager for testing"""
    session = MagicMock(spec=SessionManager)
    return session


@pytest.fixture
def ai_handler(mock_session):
    """Create AIHandler instance for testing"""
    return AIHandler(mock_session)


@pytest.fixture
def document_handler(mock_session):
    """Create DocumentHandler instance for testing"""
    return DocumentHandler(mock_session)


@pytest.fixture
def export_handler(mock_session):
    """Create ExportHandler instance for testing"""
    return ExportHandler(mock_session)


class TestAIHandler:
    """Test cases for AIHandler class"""

    def test_generate_report_success(self, ai_handler):
        """Test successful report generation"""
        mock_ai_service = MagicMock()
        mock_ai_service.is_available = True
        mock_ai_service.analyze_documents.return_value = "Generated report"
        ai_handler._ai_service = mock_ai_service

        result = ai_handler.generate_report("overview", documents={'doc1': 'content'})

        assert result == "Generated report"
        mock_ai_service.analyze_documents.assert_called_once()

    def test_generate_report_no_ai_service(self, ai_handler):
        """Test report generation without AI service"""
        ai_handler._ai_service = None

        with pytest.raises(AIError):
            ai_handler.generate_report("overview")

    @patch('app.handlers.ai_handler.create_ai_service')
    def test_setup_agent_success(self, mock_create_service, ai_handler, mock_session):
        """Test successful AI agent setup"""
        mock_ai_service = MagicMock()
        mock_ai_service.is_available = True
        mock_create_service.return_value = mock_ai_service

        result = ai_handler.setup_agent("test_key", "model")

        assert result is True
        assert ai_handler._ai_service == mock_ai_service

    @patch('app.handlers.ai_handler.create_ai_service')
    def test_setup_agent_failure(self, mock_create_service, ai_handler):
        """Test AI agent setup failure"""
        mock_create_service.return_value = None

        with pytest.raises(AIError):
            ai_handler.setup_agent("test_key", "model")

    def test_is_agent_available_true(self, ai_handler):
        """Test agent availability when available"""
        mock_ai_service = MagicMock()
        mock_ai_service.is_available = True
        ai_handler._ai_service = mock_ai_service

        assert ai_handler.is_agent_available() is True

    def test_is_agent_available_false(self, ai_handler, mock_session):
        """Test agent availability when unavailable"""
        ai_handler._ai_service = None
        mock_session.agent = None

        assert ai_handler.is_agent_available() is False


class TestDocumentHandler:
    """Test cases for DocumentHandler class"""

    @patch('app.core.document_processor.DocumentProcessor')
    def test_process_data_room_fast_success(self, mock_doc_processor, document_handler, mock_session):
        """Test successful data room processing"""
        mock_processor_instance = MagicMock()
        mock_processor_instance.vector_store = MagicMock()
        mock_doc_processor.return_value = mock_processor_instance

        with patch.object(document_handler, '_quick_document_scan') as mock_scan, \
             patch.object(document_handler, '_extract_chunks_from_faiss') as mock_extract:
            mock_scan.return_value = {'doc1': 'content1'}
            mock_extract.return_value = [{'text': 'chunk1'}]

            result = document_handler.process_data_room_fast("/test/path")

            assert result == (1, 1)
            assert mock_session.documents == {'doc1': 'content1'}
            assert mock_session.chunks == [{'text': 'chunk1'}]

    @patch('app.core.document_processor.DocumentProcessor')
    def test_process_data_room_fast_no_faiss(self, mock_doc_processor, document_handler):
        """Test data room processing without FAISS index"""
        mock_processor_instance = MagicMock()
        mock_processor_instance.vector_store = None
        mock_doc_processor.return_value = mock_processor_instance

        with pytest.raises(ProcessingError):
            document_handler.process_data_room_fast("/test/path")

    @patch('app.core.document_processor.DocumentProcessor')
    def test_get_document_processor(self, mock_doc_processor, document_handler):
        """Test getting document processor"""
        mock_processor_instance = MagicMock()
        mock_doc_processor.return_value = mock_processor_instance

        result = document_handler.get_document_processor("test_store")

        assert result == mock_processor_instance
        mock_doc_processor.assert_called_once_with(store_name="test_store")

    def test_validate_data_room_invalid_path(self, document_handler):
        """Test validating data room with invalid path"""
        result = document_handler.validate_data_room("/invalid/path")
        assert result is False


class TestExportHandler:
    """Test cases for ExportHandler class"""

    def test_export_overview_report_with_content(self, export_handler, mock_session):
        """Test overview report export with content"""
        mock_session.overview_summary = "Test overview content"

        with patch.object(export_handler, '_get_company_name', return_value='testcompany'):
            file_name, content = export_handler.export_overview_report()

            assert file_name == "company_overview_testcompany.md"
            assert "# Company Overview" in content
            assert "Test overview content" in content

    def test_export_overview_report_no_content(self, export_handler, mock_session):
        """Test overview report export without content"""
        mock_session.overview_summary = ""

        # Should return None when no content is available (handle_ui_errors decorator)
        result = export_handler.export_overview_report()
        assert result is None

    def test_export_strategic_report_success(self, export_handler, mock_session):
        """Test strategic report export"""
        mock_session.overview_summary = "Overview content"
        mock_session.strategic_summary = "Strategic content"

        with patch.object(export_handler, '_get_company_name', return_value='testcompany'):
            file_name, content = export_handler.export_strategic_report()

            assert file_name == "dd_report_testcompany.md"
            assert "# Due Diligence Report" in content

    def test_export_combined_report_success(self, export_handler, mock_session):
        """Test combined report export"""
        mock_session.overview_summary = "Overview content"
        mock_session.strategic_summary = "Strategic content"
        mock_session.checklist_results = {'Category': [{'text': 'Item'}]}
        mock_session.question_answers = {'Q1': {'has_answer': True, 'answer': 'A1'}}

        with patch.object(export_handler, '_get_company_name', return_value='testcompany'):
            file_name, content = export_handler.export_combined_report()

            assert file_name == "complete_dd_report_testcompany.md"
            assert "# Complete Due Diligence Report" in content

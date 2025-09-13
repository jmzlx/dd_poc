"""
Unit tests for core service functions

Tests essential functionality for search_documents(), parse_checklist(), and search_and_analyze() functions.
"""
import pytest
import json
from unittest.mock import Mock, patch

from app.core.search import search_documents, search_and_analyze
from app.core.parsers import parse_checklist
from app.core.document_processor import DocumentProcessor


class TestSearchDocuments:
    """Test cases for search_documents function"""

    def test_search_documents_success(self):
        """Test successful document search"""
        mock_processor = Mock(spec=DocumentProcessor)
        mock_results = [
            {
                'text': 'Sample document text',
                'source': 'test.pdf',
                'path': 'test.pdf',
                'score': 0.85,
                'metadata': {'chunk_id': 'chunk_1'}
            }
        ]
        mock_processor.search.return_value = mock_results

        result = search_documents("test query", mock_processor, top_k=5)

        assert result == mock_results
        mock_processor.search.assert_called_once_with("test query", top_k=5, threshold=None)

    def test_search_documents_no_processor(self):
        """Test search with None document processor"""
        result = search_documents("query", None)
        assert result == []


class TestParseChecklist:
    """Test cases for parse_checklist function"""

    def test_parse_checklist_success(self):
        """Test successful checklist parsing"""
        mock_llm = Mock()

        expected_json = {
            "categories": {
                "A": {
                    "name": "Corporate Structure",
                    "items": [
                        {"text": "Review articles", "original": "Review articles"},
                        {"text": "Verify agent", "original": "Verify agent"}
                    ]
                }
            }
        }

        mock_response = Mock()
        mock_response.content = json.dumps(expected_json)
        mock_llm.invoke.return_value = mock_response

        result = parse_checklist("Sample checklist text", mock_llm)

        assert "A" in result
        assert result["A"]["name"] == "Corporate Structure"
        assert len(result["A"]["items"]) == 2

    def test_parse_checklist_no_llm(self):
        """Test error when LLM is not available"""
        with pytest.raises(ValueError, match="LLM parameter is required"):
            parse_checklist("Sample text", None)


class TestSearchAndAnalyze:
    """Test cases for search_and_analyze function"""

    @patch('app.core.search.rerank_results')
    def test_search_and_analyze_checklist_mode(self, mock_rerank):
        """Test search_and_analyze in checklist mode"""
        mock_checklist_data = {
            "A": {
                "name": "Corporate Structure",
                "items": [
                    {"text": "Review articles", "original": "Review articles"},
                    {"text": "Verify agent", "original": "Verify agent"}
                ]
            }
        }

        mock_store = Mock()
        mock_store.similarity_search_with_score.return_value = [
            (Mock(page_content="Document content", metadata={"source": "/path/doc.pdf"}), 0.2)
        ]

        mock_rerank.return_value = [
            {
                'text': 'Document content',
                'source': 'doc.pdf',
                'path': 'doc.pdf',
                'score': 0.9,
                'metadata': {'source': '/path/doc.pdf'}
            }
        ]

        result = search_and_analyze(
            mock_checklist_data,
            mock_store,
            threshold=0.1,
            search_type='items'
        )

        assert "A" in result
        assert result["A"]["name"] == "Corporate Structure"
        assert len(result["A"]["items"]) == 2

    @patch('app.core.search.rerank_results')
    def test_search_and_analyze_questions_mode(self, mock_rerank):
        """Test search_and_analyze in questions mode"""
        mock_questions = [
            {"question": "What is the revenue?", "category": "A. Financial", "id": "q_0"}
        ]

        mock_store = Mock()
        mock_store.similarity_search_with_score.return_value = [
            (Mock(page_content="Financial content", metadata={"source": "/path/financial.pdf"}), 0.2)
        ]

        mock_rerank.return_value = [
            {
                'text': 'Financial document content',
                'source': 'financial.pdf',
                'path': 'financial.pdf',
                'score': 0.8,
                'metadata': {'source': '/path/financial.pdf'}
            }
        ]

        result = search_and_analyze(
            mock_questions,
            mock_store,
            threshold=0.1,
            search_type='questions'
        )

        assert "questions" in result
        assert len(result["questions"]) == 1
        assert result["questions"][0]["question"] == "What is the revenue?"

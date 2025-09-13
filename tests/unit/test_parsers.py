"""
Unit tests for parsing functions (parse_checklist and parse_questions)

Tests core functionality for the parser functions.
"""
import pytest
import json
from unittest.mock import Mock
from app.core.parsers import parse_checklist, parse_questions


class TestParseQuestions:
    """Test cases for parse_questions function"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        return Mock()

    def test_parse_questions_basic_format(self, mock_llm):
        """Test parsing questions with standard markdown format"""
        expected_json = {
            "questions": [
                {
                    "category": "A. Corporate Structure",
                    "question": "What is the company's legal structure?",
                    "id": "q_0"
                }
            ]
        }

        mock_response = Mock()
        mock_response.content = json.dumps(expected_json)
        mock_llm.invoke.return_value = mock_response

        questions_text = """
### A. Corporate Structure
1. What is the company's legal structure?
"""
        result = parse_questions(questions_text, mock_llm)

        assert len(result) == 1
        assert result[0]['category'] == 'A. Corporate Structure'
        assert result[0]['question'] == 'What is the company\'s legal structure?'
        assert result[0]['id'] == 'q_0'

    def test_parse_questions_empty_input(self, mock_llm):
        """Test parsing empty input"""
        expected_json = {
            "questions": []
        }

        mock_response = Mock()
        mock_response.content = json.dumps(expected_json)
        mock_llm.invoke.return_value = mock_response

        result = parse_questions("", mock_llm)
        assert result == []


class TestParseChecklist:
    """Test cases for parse_checklist function"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        return Mock()

    def test_parse_checklist_successful_parsing(self, mock_llm):
        """Test successful checklist parsing with valid LLM response"""
        # Expected JSON should match StructuredChecklist format with "categories" wrapper
        expected_structured_json = {
            "categories": {
                "A": {
                    "name": "Corporate Structure",
                    "items": [
                        {"text": "Review articles of incorporation", "original": "Review articles of incorporation"}
                    ]
                }
            }
        }

        # Mock LLM to return the JSON string that PydanticOutputParser expects
        mock_response = Mock()
        mock_response.content = json.dumps(expected_structured_json)
        mock_llm.invoke.return_value = mock_response

        result = parse_checklist("Sample checklist text", mock_llm)

        assert "A" in result
        assert result["A"]["name"] == "Corporate Structure"
        assert len(result["A"]["items"]) == 1

    def test_parse_checklist_no_llm_available(self, mock_llm):
        """Test error when LLM is not available"""
        # Pass None as llm to test error handling
        with pytest.raises(ValueError, match="LLM parameter is required"):
            parse_checklist("Sample text", None)

    def test_parse_checklist_invalid_json_response(self, mock_llm):
        """Test handling of invalid JSON from LLM"""
        mock_response = Mock()
        mock_response.content = "Invalid JSON response"
        mock_llm.invoke.return_value = mock_response

        with pytest.raises(RuntimeError, match="Structured parsing failed"):
            parse_checklist("Sample text", mock_llm)

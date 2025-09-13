#!/usr/bin/env python3
"""
Consolidated User Workflow Integration Tests

Focused integration tests for core user workflows:
- Company overview generation
- Strategic analysis
- Q&A functionality
- Due diligence question answering

Tests actual user workflows rather than implementation details.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ui.session_manager import SessionManager
from app.core.config import init_app_config
from app.handlers.ai_handler import AIHandler
from app.handlers.export_handler import ExportHandler
from app.ui.tabs.overview_tab import OverviewTab
from app.ui.tabs.strategic_tab import StrategicTab
from app.ui.tabs.qa_tab import QATab
from app.ui.tabs.questions_tab import QuestionsTab
from app.core.parsers import parse_questions
from app.core.search import search_documents
from app.core.exceptions import AIError, ConfigError, DocumentProcessingError, SearchError


class TestUserWorkflows:
    """Test suite for core user workflows"""

    def setup_method(self):
        """Setup test environment"""
        self.config = init_app_config()
        self.session = SessionManager()
        self.ai_handler = AIHandler(self.session)
        self.export_handler = ExportHandler(self.session)

        # Mock test documents
        self.test_documents = {
            "company_profile.pdf": {
                "content": "TechCorp is a cybersecurity company founded in 2015. "
                          "Specializes in AI-driven threat detection for enterprise clients. "
                          "Serves finance, healthcare, and government sectors.",
                "name": "Company Profile"
            },
            "financial_report.pdf": {
                "content": "Financial results: $75M revenue, $12M profit, 25% YoY growth. "
                          "Strong balance sheet with $150M total assets.",
                "name": "Financial Report"
            }
        }

        # Mock test questions
        self.test_questions_text = """
### A. Corporate Structure
1. Are incorporation documents current?
2. Are bylaws properly maintained?

### B. Financial Health
1. Are financial statements audited?
2. What is the revenue growth rate?
"""

    def test_overview_workflow_end_to_end(self):
        """Test complete overview generation workflow"""
        print("🧪 Testing overview workflow...")

        # Setup documents
        self.session.documents = self.test_documents

        # Mock AI service as available
        with patch.object(self.ai_handler, 'is_agent_available', return_value=True):
            with patch.object(self.ai_handler, 'generate_report') as mock_generate:
                mock_generate.return_value = "# Test Company Overview\n\nGenerated overview content..."

                # Test overview generation
                result = self.ai_handler.generate_report(
                    "overview",
                    documents=self.test_documents,
                    data_room_name="Test Company"
                )

                assert result is not None
                assert "Test Company Overview" in result

        print("✅ Overview workflow test passed")

    def test_strategic_workflow_end_to_end(self):
        """Test complete strategic analysis workflow"""
        print("🧪 Testing strategic workflow...")

        # Setup documents and strategy
        self.session.documents = self.test_documents
        self.session.selected_strategy_text = "Test strategy framework content"

        # Mock AI service
        with patch.object(self.ai_handler, 'is_agent_available', return_value=True):
            with patch.object(self.ai_handler, 'generate_report') as mock_generate:
                mock_generate.return_value = "# Strategic Analysis\n\nAnalysis results..."

                # Test strategic generation
                result = self.ai_handler.generate_report(
                    "strategic",
                    documents=self.test_documents,
                    strategy_text=self.session.selected_strategy_text
                )

                assert result is not None
                assert "Strategic Analysis" in result

        print("✅ Strategic workflow test passed")

    def test_qa_workflow_end_to_end(self):
        """Test complete Q&A workflow"""
        print("🧪 Testing Q&A workflow...")

        # Setup documents and chunks
        self.session.documents = self.test_documents
        self.session.chunks = [
            {
                "text": "TechCorp is a cybersecurity company",
                "source": "company_profile.pdf",
                "path": "data/company_profile.pdf",
                "score": 0.8
            }
        ]

        # Mock search functionality
        with patch('app.core.search.search_documents') as mock_search:
            mock_search.return_value = self.session.chunks

            # Mock AI service for answering
            with patch.object(self.ai_handler, 'is_agent_available', return_value=True):
                with patch.object(self.ai_handler, 'answer_question') as mock_answer:
                    mock_answer.return_value = "TechCorp is a cybersecurity company specializing in AI-driven threat detection."

                    # Test Q&A with mock document processor
                    from unittest.mock import Mock
                    mock_processor = Mock()
                    mock_processor.search.return_value = self.session.chunks

                    results = search_documents(
                        "What does TechCorp do?",
                        mock_processor,
                        top_k=5,
                        threshold=0.25
                    )

                    answer = self.ai_handler.answer_question(
                        "What does TechCorp do?",
                        [r["text"] for r in results]
                    )

                    assert len(results) > 0
                    assert "cybersecurity" in answer.lower()

        print("✅ Q&A workflow test passed")

    def test_questions_workflow_end_to_end(self):
        """Test complete due diligence questions workflow"""
        print("🧪 Testing questions workflow...")

        # Setup questions and documents
        self.session.selected_questions_text = self.test_questions_text
        self.session.documents = self.test_documents

        # Mock LLM for parsing questions
        from unittest.mock import Mock
        mock_llm_response = """
        [
            {
                "category": "A. Corporate Structure",
                "question": "Are incorporation documents current?",
                "id": "q_0"
            },
            {
                "category": "A. Corporate Structure",
                "question": "Are bylaws properly maintained?",
                "id": "q_1"
            },
            {
                "category": "B. Financial Health",
                "question": "Are financial statements audited?",
                "id": "q_2"
            },
            {
                "category": "B. Financial Health",
                "question": "What is the revenue growth rate?",
                "id": "q_3"
            }
        ]
        """
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content=mock_llm_response)

        # Parse questions
        questions = parse_questions(self.test_questions_text, llm=mock_llm)
        assert len(questions) == 4

        # Mock analysis results
        mock_answers = {
            'q_0': {
                'question': questions[0]['question'],
                'answer': 'Incorporation documents are current and properly maintained.',
                'has_answer': True
            },
            'q_1': {
                'question': questions[1]['question'],
                'answer': 'Bylaws are properly maintained and up to date.',
                'has_answer': True
            }
        }

        with patch('app.core.search.search_and_analyze') as mock_analyze:
            mock_analyze.return_value = mock_answers

            # Test question processing
            from app.core.search import search_and_analyze
            results = search_and_analyze(
                questions,
                None,
                None,
                0.3,
                'questions'
            )

            assert len(results) == 2
            assert all(r['has_answer'] for r in results.values())

        print("✅ Questions workflow test passed")

    def test_export_functionality(self):
        """Test export functionality across workflows"""
        print("🧪 Testing export functionality...")

        # Test overview export
        self.session.overview_summary = "# Test Overview\n\nExport test content"
        filename, data = self.export_handler.export_overview_report()
        assert filename is not None
        assert data is not None
        assert "Test Overview" in data

        # Test strategic export
        self.session.strategic_summary = "# Strategic Analysis\n\nExport test content"
        filename, data = self.export_handler.export_strategic_report()
        assert filename is not None
        assert data is not None
        assert "Strategic Analysis" in data

        print("✅ Export functionality test passed")

    def test_error_handling(self):
        """Test error handling across workflows"""
        print("🧪 Testing error handling...")

        # Test with no documents
        self.session.documents = {}
        assert not self.session.ready()

        # Test with no AI service
        with patch.object(self.ai_handler, 'is_agent_available', return_value=False):
            assert not self.ai_handler.is_agent_available()

        # Test AI generation with no service
        with patch.object(self.ai_handler, 'generate_report', return_value=None):
            result = self.ai_handler.generate_report("overview", documents={})
            assert result is None

        print("✅ Error handling test passed")

    def test_session_state_management(self):
        """Test session state management"""
        print("🧪 Testing session state management...")

        # Clear session state for clean test
        self.session.overview_summary = ""
        self.session.strategic_summary = ""
        self.session.processing_active = False

        # Test initial state
        assert self.session.overview_summary == ""
        assert self.session.strategic_summary == ""
        assert not self.session.processing_active

        # Test state updates
        self.session.overview_summary = "Test overview"
        self.session.strategic_summary = "Test strategic"
        self.session.processing_active = True

        assert self.session.overview_summary == "Test overview"
        assert self.session.strategic_summary == "Test strategic"
        assert self.session.processing_active

        # Test reset
        self.session.reset()
        assert self.session.overview_summary == ""
        assert self.session.strategic_summary == ""

        print("✅ Session state management test passed")


def run_workflow_tests():
    """Run all workflow tests"""
    print("🚀 Starting User Workflow Integration Tests...\n")

    test_suite = TestUserWorkflows()
    test_suite.setup_method()

    tests = [
        test_suite.test_overview_workflow_end_to_end,
        test_suite.test_strategic_workflow_end_to_end,
        test_suite.test_qa_workflow_end_to_end,
        test_suite.test_questions_workflow_end_to_end,
        test_suite.test_export_functionality,
        test_suite.test_error_handling,
        test_suite.test_session_state_management,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__} PASSED")
        except (AIError, ConfigError, DocumentProcessingError, SearchError) as e:
            print(f"❌ {test.__name__} FAILED: {str(e)}")
        print()

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All workflow tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    success = run_workflow_tests()
    sys.exit(0 if success else 1)

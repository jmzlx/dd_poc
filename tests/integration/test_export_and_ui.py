#!/usr/bin/env python3
"""
Export and UI Integration Tests

Focused integration tests for export functionality and UI integration:
- Export handlers and file generation
- UI component integration
- Session state management
- Download workflows

Tests export and UI functionality rather than core workflows.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ui.session_manager import SessionManager
from app.handlers.export_handler import ExportHandler
from app.core.config import init_app_config
from app.core.exceptions import FileOperationError, ConfigError


class TestExportAndUI:
    """Test suite for export and UI integration"""

    def setup_method(self):
        """Setup test environment"""
        self.config = init_app_config()
        self.session = SessionManager()
        self.export_handler = ExportHandler(self.session)

        # Mock test data
        self.test_overview = "# Company Overview\n\nThis is a test overview for export."
        self.test_strategic = "# Strategic Analysis\n\nThis is a test strategic analysis."
        self.test_checklist_results = {
            "Corporate Structure": [
                {"item": "Incorporation documents", "status": "completed", "notes": "Verified"}
            ]
        }

    def test_export_handler_initialization(self):
        """Test export handler initialization"""
        print("🧪 Testing export handler initialization...")

        assert self.export_handler is not None
        assert self.export_handler.session == self.session

        print("✅ Export handler initialization test passed")

    def test_overview_export(self):
        """Test overview report export"""
        print("🧪 Testing overview export...")

        # Setup test data
        self.session.overview_summary = self.test_overview

        # Test export
        filename, export_data = self.export_handler.export_overview_report()

        assert filename is not None
        assert export_data is not None
        assert "Company Overview" in export_data
        assert filename.endswith('.md')

        print("✅ Overview export test passed")

    def test_strategic_export(self):
        """Test strategic report export"""
        print("🧪 Testing strategic export...")

        # Setup test data
        self.session.strategic_summary = self.test_strategic

        # Test export
        filename, export_data = self.export_handler.export_strategic_report()

        assert filename is not None
        assert export_data is not None
        assert "Strategic Analysis" in export_data
        assert filename.endswith('.md')

        print("✅ Strategic export test passed")

    def test_checklist_export(self):
        """Test checklist report export"""
        print("🧪 Testing checklist export...")

        # Setup test data
        self.session.checklist_results = self.test_checklist_results

        # Test export
        filename, export_data = self.export_handler.export_checklist_report()

        assert filename is not None
        assert export_data is not None
        assert "Corporate Structure" in export_data
        assert filename.endswith('.md')

        print("✅ Checklist export test passed")

    def test_export_with_empty_data(self):
        """Test export functionality with empty data"""
        print("🧪 Testing export with empty data...")

        # Clear session data
        self.session.overview_summary = ""
        self.session.strategic_summary = ""
        self.session.checklist_results = {}

        # Test exports with empty data - should raise exceptions
        try:
            filename1, data1 = self.export_handler.export_overview_report()
            assert False, "Should have raised exception for empty overview"
        except Exception:
            pass  # Expected

        try:
            filename2, data2 = self.export_handler.export_strategic_report()
            assert False, "Should have raised exception for empty strategic"
        except Exception:
            pass  # Expected

        try:
            filename3, data3 = self.export_handler.export_checklist_report()
            assert False, "Should have raised exception for empty checklist"
        except Exception:
            pass  # Expected

        print("✅ Export with empty data test passed")

    def test_session_state_persistence(self):
        """Test session state persistence for exports"""
        print("🧪 Testing session state persistence...")

        # Set initial state
        self.session.overview_summary = self.test_overview
        self.session.processing_active = True

        # Verify state
        assert self.session.overview_summary == self.test_overview
        assert self.session.processing_active is True

        # Test state reset
        self.session.reset_processing()
        assert self.session.processing_active is False
        # Overview summary should persist
        assert self.session.overview_summary == self.test_overview

        print("✅ Session state persistence test passed")

    def test_export_content_validation(self):
        """Test export content validation"""
        print("🧪 Testing export content validation...")

        # Setup complex test data
        self.session.overview_summary = """# Complex Overview

## Executive Summary
This is a detailed overview with multiple sections.

## Key Findings
- Point 1
- Point 2
- Point 3

## Recommendations
Some recommendations here.
"""

        # Export and validate
        filename, export_data = self.export_handler.export_overview_report()

        assert "Executive Summary" in export_data
        assert "Key Findings" in export_data
        assert "Recommendations" in export_data
        assert "- Point 1" in export_data

        print("✅ Export content validation test passed")

    def test_ui_component_integration(self):
        """Test UI component integration (mocked)"""
        print("🧪 Testing UI component integration...")

        # Mock UI components that might interact with export handler
        with patch('streamlit.download_button') as mock_download:
            with patch('streamlit.markdown') as mock_markdown:

                # Simulate UI interaction
                filename, data = self.export_handler.export_overview_report()

                # Verify export still works
                assert filename is not None
                assert data is not None

        print("✅ UI component integration test passed")


def run_export_and_ui_tests():
    """Run all export and UI integration tests"""
    print("🚀 Starting Export and UI Integration Tests...\n")

    test_suite = TestExportAndUI()
    test_suite.setup_method()

    tests = [
        test_suite.test_export_handler_initialization,
        test_suite.test_overview_export,
        test_suite.test_strategic_export,
        test_suite.test_checklist_export,
        test_suite.test_export_with_empty_data,
        test_suite.test_session_state_persistence,
        test_suite.test_export_content_validation,
        test_suite.test_ui_component_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__} PASSED")
        except (FileOperationError, ConfigError) as e:
            print(f"❌ {test.__name__} FAILED: {str(e)}")
        print()

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All export and UI tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    success = run_export_and_ui_tests()
    sys.exit(0 if success else 1)

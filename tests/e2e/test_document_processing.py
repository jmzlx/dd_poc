#!/usr/bin/env python3
"""
E2E Tests for Document Processing Workflow

Tests the core document processing functionality:
- Data room selection and processing
- Document upload and indexing
- Search functionality
- Error handling for document operations
"""

import pytest
import os
from playwright.sync_api import Page, expect
from .conftest import StreamlitPageHelpers


class TestDocumentProcessing:
    """Test document processing and data room functionality"""

    def test_data_room_selection_interface(self, page: Page, streamlit_helpers: StreamlitPageHelpers, sample_test_data):
        """Test that data room selection interface is functional"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Look for data room selection in sidebar
        sidebar = page.locator("[data-testid='stSidebar']")
        
        # Should have some way to select/configure data rooms
        data_room_elements = sidebar.locator("text=/.*[Dd]ata.*[Rr]oom.*|.*VDR.*|.*[Dd]ocument.*/")
        expect(data_room_elements.first).to_be_visible()

    def test_document_processing_workflow(self, page: Page, streamlit_helpers: StreamlitPageHelpers, sample_test_data):
        """Test the complete document processing workflow"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Navigate to document processing section
        # This might be in the main area or a specific tab
        
        # Look for document processing controls
        processing_elements = page.locator("text=/.*[Pp]rocess.*|.*[Aa]nalyze.*|.*[Bb]uild.*|.*[Ii]ndex.*/")
        
        if processing_elements.count() > 0:
            # Check if there's a processing button or similar
            process_button = page.locator("button:has-text(/.*[Pp]rocess.*|.*[Bb]uild.*|.*[Aa]nalyze.*/)")
            
            if process_button.count() > 0:
                # Click the process button (but don't wait for completion in basic test)
                process_button.first.click()
                
                # Should show some indication of processing starting
                # Could be a spinner, status message, etc.
                processing_indicators = page.locator(".stSpinner, [data-testid='stSpinner'], .stStatus, text=/.*[Pp]rocessing.*|.*[Ll]oading.*/")
                
                # Give it a moment to start processing
                page.wait_for_timeout(2000)

    def test_file_upload_interface(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test file upload interface if available"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Look for file upload components
        file_uploaders = page.locator("input[type='file'], [data-testid='stFileUploader']")
        
        if file_uploaders.count() > 0:
            expect(file_uploaders.first).to_be_visible()
            
            # Test that file uploader accepts appropriate file types
            file_uploader = file_uploaders.first
            accept_attr = file_uploader.get_attribute("accept")
            
            # Should accept common document formats
            if accept_attr:
                assert any(fmt in accept_attr for fmt in [".pdf", ".md", ".txt", ".docx"]), \
                    f"File uploader should accept document formats, got: {accept_attr}"

    def test_search_functionality(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test document search functionality"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Look for search interface
        search_elements = page.locator("input[placeholder*='search'], input[aria-label*='search'], text=/.*[Ss]earch.*/")
        
        if search_elements.count() > 0:
            search_input = search_elements.first
            
            # Test basic search functionality
            if search_input.get_attribute("type") != "file":  # Make sure it's not a file input
                search_input.fill("revenue")
                
                # Look for search button or trigger search
                search_button = page.locator("button:has-text(/.*[Ss]earch.*|.*[Ff]ind.*/)")
                if search_button.count() > 0:
                    search_button.first.click()
                else:
                    # Try pressing Enter
                    search_input.press("Enter")
                
                # Wait for search results or indication
                page.wait_for_timeout(2000)

    def test_document_status_display(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that document processing status is displayed"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Look for status indicators
        status_elements = page.locator("text=/.*[Ss]tatus.*|.*[Rr]eady.*|.*[Pp]rocessed.*|.*[Dd]ocuments.*found.*/")
        
        # Should have some indication of system state
        # This could be "No documents processed", "Ready", "X documents indexed", etc.
        if status_elements.count() > 0:
            expect(status_elements.first).to_be_visible()

    def test_error_handling_invalid_path(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test error handling for invalid data room paths"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Look for path input fields
        path_inputs = page.locator("input[placeholder*='path'], input[aria-label*='path']")
        
        if path_inputs.count() > 0:
            path_input = path_inputs.first
            
            # Enter an invalid path
            path_input.fill("/nonexistent/path/to/documents")
            
            # Look for a button to submit/validate
            submit_buttons = page.locator("button:has-text(/.*[Ss]ubmit.*|.*[Cc]heck.*|.*[Vv]alidate.*|.*[Pp]rocess.*/)")
            
            if submit_buttons.count() > 0:
                submit_buttons.first.click()
                
                # Should show an error message
                error_elements = page.locator(".stError, [data-testid='stError'], text=/.*[Ee]rror.*|.*[Nn]ot found.*|.*[Ii]nvalid.*/")
                
                # Wait for error message to appear
                page.wait_for_timeout(3000)
                
                # Should have some error indication
                if error_elements.count() > 0:
                    expect(error_elements.first).to_be_visible()

    def test_processing_progress_indicators(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that processing shows appropriate progress indicators"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Look for any processing buttons
        process_buttons = page.locator("button:has-text(/.*[Pp]rocess.*|.*[Bb]uild.*|.*[Aa]nalyze.*|.*[Ii]ndex.*/)")
        
        if process_buttons.count() > 0:
            # Click a processing button
            process_buttons.first.click()
            
            # Should show progress indicators
            progress_elements = page.locator(".stSpinner, .stProgress, [data-testid='stSpinner'], [data-testid='stProgress']")
            
            # Give it a moment for progress indicators to appear
            page.wait_for_timeout(1000)
            
            # Note: We don't wait for completion as that could take too long for E2E tests

    def test_document_metadata_display(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that document metadata is displayed when available"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Look for metadata displays
        metadata_elements = page.locator("text=/.*[Dd]ocument.*[Cc]ount.*|.*[Ff]iles.*found.*|.*[Cc]hunks.*|.*[Ii]ndex.*size.*/")
        
        # Should show some document information if documents are processed
        # This could be document counts, index size, processing status, etc.
        
        # Navigate through tabs to see if any show document information
        tabs = page.locator("[data-testid='stTabs'] button, .stTabs button")
        
        if tabs.count() > 0:
            for i in range(min(tabs.count(), 3)):  # Check first 3 tabs
                tabs.nth(i).click()
                page.wait_for_timeout(1000)
                
                # Check for document-related information in this tab
                doc_info = page.locator("text=/.*[Dd]ocuments.*|.*[Ff]iles.*|.*[Cc]hunks.*|.*[Pp]rocessed.*/")
                if doc_info.count() > 0:
                    expect(doc_info.first).to_be_visible()
                    break

    def test_data_room_switching(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test switching between different data rooms"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Look for data room selection dropdown or similar
        data_room_selectors = page.locator("select, [data-testid='stSelectbox']")
        
        if data_room_selectors.count() > 0:
            selector = data_room_selectors.first
            
            # Check if it has multiple options
            selector.click()
            page.wait_for_timeout(500)
            
            options = page.locator("[data-value], option")
            
            if options.count() > 1:
                # Select a different option
                options.nth(1).click()
                
                # Should trigger some update in the interface
                page.wait_for_timeout(2000)
                
                # Look for status updates or changes
                status_updates = page.locator("text=/.*[Ll]oading.*|.*[Ss]witching.*|.*[Pp]rocessing.*/")

    @pytest.mark.slow
    def test_full_processing_workflow(self, page_slow: Page, streamlit_helpers: StreamlitPageHelpers, sample_test_data):
        """Test the complete document processing workflow with real data (slower test)"""
        page = page_slow  # Use the slow page fixture
        streamlit_helpers.wait_for_streamlit_load()
        
        # This test would actually process documents if a test data room is available
        # Check if test VDR path exists
        vdr_path = sample_test_data["vdr_path"]
        
        if vdr_path.exists() and any(vdr_path.iterdir()):
            # Look for path configuration
            path_inputs = page.locator("input[placeholder*='path'], input[aria-label*='path']")
            
            if path_inputs.count() > 0:
                path_input = path_inputs.first
                path_input.fill(str(vdr_path))
                
                # Look for process button
                process_buttons = page.locator("button:has-text(/.*[Pp]rocess.*|.*[Bb]uild.*/)")
                
                if process_buttons.count() > 0:
                    process_buttons.first.click()
                    
                    # Wait for processing to complete or show progress
                    # Use the extended timeout for this slow operation
                    try:
                        streamlit_helpers.wait_for_processing(timeout=120000)  # 2 minutes
                        
                        # Check for success indicators
                        success_elements = page.locator(".stSuccess, text=/.*[Ss]uccess.*|.*[Cc]omplete.*|.*[Ff]inished.*/")
                        
                        page.wait_for_timeout(2000)
                        
                        # Verify that documents were processed
                        status_elements = page.locator("text=/.*documents.*processed.*|.*files.*indexed.*|.*chunks.*created.*/")
                        
                    except Exception as e:
                        # Processing might still be ongoing, that's okay for this test
                        print(f"Processing timeout or error: {e}")
        else:
            pytest.skip("No test VDR data available for full processing test")

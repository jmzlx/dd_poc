#!/usr/bin/env python3
"""
E2E Tests for App Startup and Basic Navigation

Tests the basic functionality of the Streamlit AI Due Diligence app:
- App loads successfully
- Main UI components are present
- Navigation between tabs works
- Basic error handling
"""

import pytest
from playwright.sync_api import Page, expect
from .conftest import StreamlitPageHelpers


class TestAppStartup:
    """Test basic app startup and navigation functionality"""

    def test_app_loads_successfully(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that the app loads and displays main components"""
        # Wait for Streamlit to fully load
        streamlit_helpers.wait_for_streamlit_load()
        
        # Check that main app container is present
        expect(page.locator("[data-testid='stApp']")).to_be_visible()
        
        # Check for the main title
        expect(page.locator("h1")).to_contain_text("AI Due Diligence")
        
        # Check that sidebar is present
        expect(page.locator("[data-testid='stSidebar']")).to_be_visible()
        
        # Verify no critical errors are displayed
        error_elements = page.locator(".stException, [data-testid='stException']")
        expect(error_elements).to_have_count(0)

    def test_sidebar_components_present(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that sidebar contains expected components"""
        streamlit_helpers.wait_for_streamlit_load()
        
        sidebar = page.locator("[data-testid='stSidebar']")
        
        # Check for key sidebar sections
        expect(sidebar).to_be_visible()
        
        # Should have some form of data room selection
        data_room_section = sidebar.locator("text=/.*[Dd]ata.*[Rr]oom.*/")
        expect(data_room_section.first).to_be_visible()
        
        # Should have AI configuration section
        ai_section = sidebar.locator("text=/.*AI.*|.*[Aa]nthropric.*|.*API.*/")
        expect(ai_section.first).to_be_visible()

    def test_main_tabs_present(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that main navigation tabs are present"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Look for tab-like elements
        tab_container = page.locator("[data-testid='stTabs'], .stTabs")
        
        if tab_container.count() > 0:
            expect(tab_container.first).to_be_visible()
            
            # Check for expected tab names
            expected_tabs = ["Overview", "Strategic", "Checklist", "Questions", "Q&A", "Graph"]
            
            for tab_name in expected_tabs:
                tab_element = page.locator(f"text='{tab_name}'").first
                if tab_element.count() > 0:
                    expect(tab_element).to_be_visible()

    def test_tab_navigation_works(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that clicking on tabs changes the content"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Find available tabs
        tabs = page.locator("[data-testid='stTabs'] button, .stTabs button")
        
        if tabs.count() > 1:
            # Get initial tab content
            initial_content = page.locator("[data-testid='stTabContent'], .stTabContent").first
            initial_text = initial_content.inner_text() if initial_content.count() > 0 else ""
            
            # Click on second tab
            tabs.nth(1).click()
            page.wait_for_timeout(1000)  # Wait for content to update
            
            # Check that content changed
            updated_content = page.locator("[data-testid='stTabContent'], .stTabContent").first
            if updated_content.count() > 0:
                updated_text = updated_content.inner_text()
                assert updated_text != initial_text, "Tab content should change when switching tabs"

    def test_responsive_design(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that the app works on different screen sizes"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Test mobile viewport
        page.set_viewport_size({"width": 375, "height": 667})
        page.wait_for_timeout(1000)
        
        # App should still be functional
        expect(page.locator("[data-testid='stApp']")).to_be_visible()
        
        # Test desktop viewport
        page.set_viewport_size({"width": 1920, "height": 1080})
        page.wait_for_timeout(1000)
        
        # App should still be functional
        expect(page.locator("[data-testid='stApp']")).to_be_visible()
        expect(page.locator("[data-testid='stSidebar']")).to_be_visible()

    def test_error_handling_for_missing_config(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that the app handles missing configuration gracefully"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # The app should load even without API keys configured
        expect(page.locator("[data-testid='stApp']")).to_be_visible()
        
        # Should not show critical errors, but might show warnings
        critical_errors = page.locator(".stException, [data-testid='stException']")
        expect(critical_errors).to_have_count(0)
        
        # Warnings are acceptable
        warnings = page.locator(".stWarning, [data-testid='stWarning']")
        # Warnings may or may not be present, that's okay

    def test_page_title_and_metadata(self, page: Page):
        """Test that page has proper title and metadata"""
        # Check page title contains relevant keywords
        title = page.title()
        title_lower = title.lower()
        assert any(keyword in title_lower for keyword in ["due diligence", "dd", "ai"]), \
            f"Page title should contain relevant keywords, got: {title}"

    def test_accessibility_basics(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test basic accessibility features"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Check that main content areas have proper structure
        main_content = page.locator("main, [role='main']")
        expect(main_content).to_be_visible()
        
        # Check for heading structure
        headings = page.locator("h1, h2, h3, h4, h5, h6")
        expect(headings.first).to_be_visible()
        
        # Check that interactive elements are focusable
        buttons = page.locator("button")
        if buttons.count() > 0:
            # Focus the first button
            buttons.first.focus()
            # Should be focused (basic accessibility check)
            expect(buttons.first).to_be_focused()

    def test_no_javascript_errors(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that there are no critical JavaScript errors"""
        js_errors = []
        
        def handle_console_message(msg):
            if msg.type == "error":
                js_errors.append(msg.text)
        
        page.on("console", handle_console_message)
        
        streamlit_helpers.wait_for_streamlit_load()
        
        # Wait a bit for any delayed errors
        page.wait_for_timeout(3000)
        
        # Filter out known Streamlit warnings/errors that are not critical
        critical_errors = [
            error for error in js_errors 
            if not any(ignore in error.lower() for ignore in [
                "favicon.ico",
                "websocket",
                "analytics",
                "mixpanel"
            ])
        ]
        
        assert len(critical_errors) == 0, f"JavaScript errors found: {critical_errors}"

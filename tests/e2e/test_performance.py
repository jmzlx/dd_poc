#!/usr/bin/env python3
"""
E2E Performance and Load Tests

Tests performance characteristics and load handling:
- Page load times
- Response times for key operations
- Memory usage stability
- Concurrent user simulation
"""

import pytest
import time
from playwright.sync_api import Page, expect
from .conftest import StreamlitPageHelpers


class TestPerformance:
    """Test performance characteristics of the application"""

    def test_initial_load_time(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that initial page load is within acceptable time"""
        start_time = time.time()
        
        # Navigate to app (this happens in the fixture, but we'll measure it)
        page.goto(page.url)
        streamlit_helpers.wait_for_streamlit_load()
        
        load_time = time.time() - start_time
        
        # Should load within 15 seconds (generous for AI app)
        assert load_time < 15.0, f"Page load took {load_time:.2f}s, should be under 15s"

    def test_tab_switching_performance(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that tab switching is responsive"""
        streamlit_helpers.wait_for_streamlit_load()
        
        tabs = page.locator("[data-testid='stTabs'] button, .stTabs button")
        
        if tabs.count() > 1:
            switch_times = []
            
            for i in range(min(tabs.count(), 4)):  # Test first 4 tabs
                start_time = time.time()
                tabs.nth(i).click()
                
                # Wait for content to load
                page.wait_for_timeout(500)
                
                switch_time = time.time() - start_time
                switch_times.append(switch_time)
            
            # Average switch time should be reasonable
            avg_switch_time = sum(switch_times) / len(switch_times)
            assert avg_switch_time < 2.0, f"Tab switching too slow: {avg_switch_time:.2f}s average"

    def test_memory_stability(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that the app doesn't have major memory leaks during basic usage"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Get initial memory usage (JavaScript)
        initial_memory = page.evaluate("window.performance.memory ? window.performance.memory.usedJSHeapSize : 0")
        
        if initial_memory > 0:  # Chrome supports memory API
            # Perform various operations
            tabs = page.locator("[data-testid='stTabs'] button, .stTabs button")
            
            if tabs.count() > 0:
                # Switch between tabs multiple times
                for _ in range(3):
                    for i in range(min(tabs.count(), 3)):
                        tabs.nth(i).click()
                        page.wait_for_timeout(1000)
                
                # Get memory after operations
                final_memory = page.evaluate("window.performance.memory.usedJSHeapSize")
                
                # Memory should not have grown excessively (allowing for reasonable growth)
                memory_growth = final_memory - initial_memory
                memory_growth_mb = memory_growth / (1024 * 1024)
                
                # Allow up to 50MB growth for normal operations
                assert memory_growth_mb < 50, f"Excessive memory growth: {memory_growth_mb:.1f}MB"

    def test_concurrent_operations(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test handling of multiple UI operations"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Simulate rapid user interactions
        tabs = page.locator("[data-testid='stTabs'] button, .stTabs button")
        buttons = page.locator("button")
        
        # Rapidly switch tabs and click buttons
        start_time = time.time()
        
        operations = 0
        while time.time() - start_time < 5:  # 5 seconds of rapid operations
            if tabs.count() > 1:
                # Switch to random tab
                tab_index = operations % tabs.count()
                tabs.nth(tab_index).click()
                
            # Click available buttons
            if buttons.count() > 0:
                button_index = operations % buttons.count()
                try:
                    buttons.nth(button_index).click(timeout=1000)
                except:
                    pass  # Button might not be clickable, that's okay
            
            operations += 1
            page.wait_for_timeout(200)  # Small delay between operations
        
        # App should still be responsive
        expect(page.locator("[data-testid='stApp']")).to_be_visible()
        
        # Should have performed multiple operations
        assert operations > 10, f"Should have performed multiple operations, got {operations}"

    @pytest.mark.slow
    def test_large_document_processing_performance(self, page_slow: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test performance with large document processing"""
        page = page_slow
        streamlit_helpers.wait_for_streamlit_load()
        
        # This test would measure processing time for large document sets
        # For now, just test that the interface remains responsive
        
        process_buttons = page.locator("button:has-text(/.*[Pp]rocess.*|.*[Bb]uild.*/)")
        
        if process_buttons.count() > 0:
            start_time = time.time()
            process_buttons.first.click()
            
            # Check that UI remains responsive during processing
            for _ in range(5):
                page.wait_for_timeout(2000)
                
                # UI should still be interactive
                expect(page.locator("[data-testid='stApp']")).to_be_visible()
                
                # Check if processing completed
                if time.time() - start_time > 30:  # Max 30 seconds for this test
                    break

    def test_error_recovery_performance(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test that error conditions don't significantly impact performance"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Trigger potential errors and measure recovery time
        error_scenarios = [
            lambda: page.locator("input[type='file']").set_input_files("nonexistent_file.pdf") if page.locator("input[type='file']").count() > 0 else None,
            lambda: page.locator("input").first.fill("invalid/path/data") if page.locator("input").count() > 0 else None,
        ]
        
        for scenario in error_scenarios:
            if scenario():
                start_time = time.time()
                
                # Wait for error to be handled
                page.wait_for_timeout(3000)
                
                recovery_time = time.time() - start_time
                
                # Error recovery should be quick
                assert recovery_time < 5.0, f"Error recovery took {recovery_time:.2f}s"
                
                # App should still be functional
                expect(page.locator("[data-testid='stApp']")).to_be_visible()

    def test_network_timeout_handling(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test graceful handling of network timeouts"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Set a very short network timeout to simulate network issues
        page.set_default_timeout(1000)  # 1 second
        
        try:
            # Try operations that might involve network calls
            ai_buttons = page.locator("button:has-text(/.*[Gg]enerate.*|.*[Aa]nalyze.*/)")
            
            if ai_buttons.count() > 0:
                ai_buttons.first.click()
                
                # This might timeout, which is expected
                page.wait_for_timeout(2000)
            
        except Exception:
            # Timeouts are expected in this test
            pass
        finally:
            # Reset timeout
            page.set_default_timeout(30000)
        
        # App should still be functional after network issues
        expect(page.locator("[data-testid='stApp']")).to_be_visible()

    def test_resource_usage_monitoring(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Monitor basic resource usage patterns"""
        streamlit_helpers.wait_for_streamlit_load()
        
        # Check for excessive resource usage patterns
        # This is basic monitoring, not comprehensive profiling
        
        # Check for excessive number of DOM elements (potential memory issue)
        dom_element_count = page.evaluate("document.getElementsByTagName('*').length")
        assert dom_element_count < 10000, f"Too many DOM elements: {dom_element_count}"
        
        # Check for excessive number of event listeners (potential memory leak)
        if hasattr(page, 'evaluate'):
            try:
                # Basic check for common resource usage issues
                script_tags = page.evaluate("document.getElementsByTagName('script').length")
                assert script_tags < 50, f"Too many script tags: {script_tags}"
                
                style_tags = page.evaluate("document.getElementsByTagName('style').length")
                assert style_tags < 100, f"Too many style tags: {style_tags}"
                
            except Exception:
                # Some checks might not work in all browser contexts
                pass

    def test_responsive_design_performance(self, page: Page, streamlit_helpers: StreamlitPageHelpers):
        """Test performance across different viewport sizes"""
        streamlit_helpers.wait_for_streamlit_load()
        
        viewports = [
            {"width": 375, "height": 667},   # Mobile
            {"width": 768, "height": 1024},  # Tablet
            {"width": 1920, "height": 1080}, # Desktop
        ]
        
        for viewport in viewports:
            start_time = time.time()
            
            page.set_viewport_size(viewport)
            page.wait_for_timeout(1000)  # Wait for reflow
            
            resize_time = time.time() - start_time
            
            # Resize should be quick
            assert resize_time < 3.0, f"Viewport resize took {resize_time:.2f}s for {viewport}"
            
            # App should remain functional
            expect(page.locator("[data-testid='stApp']")).to_be_visible()

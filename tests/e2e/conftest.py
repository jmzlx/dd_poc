#!/usr/bin/env python3
"""
E2E Test Configuration and Fixtures

Shared configuration and fixtures for Playwright E2E tests.
"""

import os
import time
import subprocess
import signal
import pytest
import requests
from playwright.sync_api import Playwright, Browser, BrowserContext, Page
from pathlib import Path

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from playwright.config.py in project root
try:
    import playwright_config
    get_playwright_config = playwright_config.get_playwright_config
    TEST_CONFIG = playwright_config.TEST_CONFIG
except ImportError:
    # Fallback configuration if config file not found
    def get_playwright_config():
        return {
            "base_url": "http://localhost:8501",
            "timeout": 30000,
            "expect_timeout": 10000,
            "headless": True,
            "viewport": {"width": 1280, "height": 720},
            "ignore_https_errors": True,
        }
    
    TEST_CONFIG = {
        "app_startup_timeout": 60,
        "slow_test_timeout": 120,
        "fast_test_timeout": 30,
    }


class StreamlitApp:
    """Helper class to manage Streamlit app lifecycle"""
    
    def __init__(self, app_path: str, port: int = 8501):
        self.app_path = app_path
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
    
    def start(self):
        """Start the Streamlit app"""
        if self.is_running():
            print(f"Streamlit app already running on port {self.port}")
            return
            
        print(f"Starting Streamlit app: {self.app_path}")
        
        # Start Streamlit in the background
        self.process = subprocess.Popen([
            "uv", "run", "streamlit", "run", self.app_path,
            "--server.port", str(self.port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for app to start
        self._wait_for_startup()
        
    def stop(self):
        """Stop the Streamlit app"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            print("Streamlit app stopped")
    
    def is_running(self):
        """Check if the app is running and responsive"""
        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _wait_for_startup(self, timeout=TEST_CONFIG["app_startup_timeout"]):
        """Wait for the Streamlit app to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_running():
                print("Streamlit app is ready!")
                time.sleep(2)  # Give it a moment to fully initialize
                return
            time.sleep(1)
        
        # If health check failed, try the main page
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.base_url, timeout=5)
                if response.status_code == 200:
                    print("Streamlit app is ready!")
                    time.sleep(3)  # Give it a moment to fully initialize
                    return
            except:
                pass
            time.sleep(1)
        
        raise RuntimeError(f"Streamlit app failed to start within {timeout} seconds")


@pytest.fixture(scope="session")
def streamlit_app():
    """Session-scoped fixture to manage Streamlit app lifecycle"""
    app_path = str(Path(__file__).parent.parent.parent / "app" / "main.py")
    app = StreamlitApp(app_path)
    
    app.start()
    
    yield app
    
    app.stop()


@pytest.fixture(scope="session")
def browser_context_args():
    """Configure browser context arguments"""
    config = get_playwright_config()
    return {
        "viewport": config["viewport"],
        "ignore_https_errors": config["ignore_https_errors"],
        "record_video_dir": "test-results/videos/" if config.get("video") else None,
    }


@pytest.fixture
def page(streamlit_app: StreamlitApp, browser: Browser, browser_context_args):
    """Create a new page for each test"""
    config = get_playwright_config()
    
    context = browser.new_context(**browser_context_args)
    page = context.new_page()
    
    # Set timeouts
    page.set_default_timeout(config["timeout"])
    
    # Navigate to the app
    page.goto(streamlit_app.base_url)
    
    # Wait for Streamlit to be fully loaded
    page.wait_for_load_state("networkidle")
    
    yield page
    
    # Cleanup
    context.close()


@pytest.fixture
def page_slow(streamlit_app: StreamlitApp, browser: Browser, browser_context_args):
    """Create a new page with extended timeout for slow operations (AI calls)"""
    config = get_playwright_config()
    
    context = browser.new_context(**browser_context_args)
    page = context.new_page()
    
    # Set extended timeouts for AI operations
    page.set_default_timeout(TEST_CONFIG["slow_test_timeout"] * 1000)
    
    # Navigate to the app
    page.goto(streamlit_app.base_url)
    page.wait_for_load_state("networkidle")
    
    yield page
    
    context.close()


@pytest.fixture
def sample_test_data():
    """Provide sample test data paths"""
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    return {
        "strategy_file": data_dir / "strategy" / "rockman.md",
        "checklist_file": data_dir / "checklist" / "original.md", 
        "questions_file": data_dir / "questions" / "due diligence.md",
        "vdr_path": data_dir / "vdrs" / "automated-services-transformation",
    }


class StreamlitPageHelpers:
    """Helper methods for interacting with Streamlit components"""
    
    def __init__(self, page: Page):
        self.page = page
    
    def wait_for_streamlit_load(self):
        """Wait for Streamlit app to fully load"""
        # Wait for the main container
        self.page.wait_for_selector("[data-testid='stApp']", timeout=10000)
        # Wait for sidebar
        self.page.wait_for_selector("[data-testid='stSidebar']", timeout=5000)
    
    def click_button_by_text(self, text: str):
        """Click a button by its text content"""
        self.page.locator(f"button:has-text('{text}')").click()
    
    def upload_file(self, file_input_selector: str, file_path: str):
        """Upload a file using Streamlit file uploader"""
        self.page.locator(file_input_selector).set_input_files(file_path)
    
    def select_option(self, selectbox_label: str, option: str):
        """Select an option from a Streamlit selectbox"""
        self.page.locator(f"[data-testid='stSelectbox']:has-text('{selectbox_label}')").click()
        self.page.locator(f"[data-value='{option}']").click()
    
    def enter_text_input(self, label: str, text: str):
        """Enter text into a Streamlit text input"""
        input_element = self.page.locator(f"input[placeholder*='{label}'], input[aria-label*='{label}']")
        input_element.clear()
        input_element.fill(text)
    
    def wait_for_success_message(self, timeout: int = 30000):
        """Wait for a success message to appear"""
        self.page.wait_for_selector(".stSuccess, [data-testid='stSuccess']", timeout=timeout)
    
    def wait_for_processing(self, timeout: int = 60000):
        """Wait for processing indicators to disappear"""
        # Wait for spinners to disappear
        self.page.wait_for_selector(".stSpinner", state="hidden", timeout=timeout)


@pytest.fixture
def streamlit_helpers(page: Page):
    """Provide helper methods for Streamlit interactions"""
    return StreamlitPageHelpers(page)

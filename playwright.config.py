#!/usr/bin/env python3
"""
Playwright Configuration for E2E Tests

Configuration for end-to-end testing of the Streamlit AI Due Diligence application.
"""

import os
from playwright.sync_api import Playwright
import pytest

def pytest_configure(config):
    """Configure Playwright for pytest"""
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "0")

# Playwright configuration
def get_playwright_config():
    return {
        "base_url": "http://localhost:8501",  # Default Streamlit port
        "timeout": 30000,  # 30 seconds
        "expect_timeout": 10000,  # 10 seconds for assertions
        "headless": True,  # Set to False for debugging
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
        "video": "retain-on-failure",
        "screenshot": "only-on-failure",
        "browser_args": [
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-gpu"
        ]
    }

# Test configuration
TEST_CONFIG = {
    "app_startup_timeout": 60,  # Time to wait for Streamlit app to start
    "slow_test_timeout": 120,   # Timeout for slow tests (AI operations)
    "fast_test_timeout": 30,    # Timeout for fast UI tests
}

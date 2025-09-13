"""
Pytest configuration and shared fixtures for dd-poc tests
"""
import sys
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock

# Add the app directory to the Python path
app_path = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_path))


@pytest.fixture(scope="session")
def app_config():
    """Provide the application configuration for tests"""
    from app.core.config import get_app_config
    return get_app_config()


@pytest.fixture(scope="session")
def test_data_dir(app_config):
    """Provide the test data directory"""
    return app_config.paths['data_dir']


@pytest.fixture(scope="session")
def checklist_dir(app_config):
    """Provide the checklist directory"""
    return app_config.paths['checklist_dir']


@pytest.fixture(scope="session")
def faiss_dir(app_config):
    """Provide the FAISS directory"""
    return app_config.paths['faiss_dir']


@pytest.fixture
def mock_vector_store():
    """Provide a mock vector store for testing"""
    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = []
    mock_store.index = MagicMock()
    mock_store.index.ntotal = 100
    return mock_store


@pytest.fixture
def sample_checklist_text():
    """Provide sample checklist text for testing"""
    return """
# Due Diligence Checklist

## A. Corporate Structure
1. Review articles of incorporation
2. Verify registered agent information
3. Confirm ownership structure

## B. Financial Health
1. Review last 3 years financial statements
2. Analyze debt obligations
3. Check for outstanding litigation
"""


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests"""
    from app.core.logging import logger
    # Configure logger for test environment
    logger.setLevel("WARNING")  # Reduce log noise during tests

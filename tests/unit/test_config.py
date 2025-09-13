"""
Unit tests for configuration module
"""
import pytest
from unittest.mock import patch

from app.core.config import get_app_config, AppConfig


def test_app_config_creation():
    """Test that app config can be created"""
    config = get_app_config()
    assert isinstance(config, AppConfig)
    assert hasattr(config, 'paths')
    assert hasattr(config, 'processing')
    assert hasattr(config, 'model')


def test_config_paths_exist():
    """Test that configuration paths exist"""
    config = get_app_config()

    # Check that all required paths are defined
    required_paths = ['data_dir', 'checklist_dir', 'faiss_dir']
    paths = config.paths
    for path_name in required_paths:
        assert path_name in paths
        path_value = paths[path_name]
        assert path_value is not None


def test_processing_config():
    """Test processing configuration values"""
    config = get_app_config()

    # Check processing configuration
    processing = config.processing
    assert 'chunk_size' in processing
    assert 'similarity_threshold' in processing

    # Check that values are reasonable
    assert processing['chunk_size'] > 0
    assert 0.0 <= processing['similarity_threshold'] <= 1.0


def test_config_validation_success():
    """Test that configuration validation passes with valid config"""
    config = get_app_config()
    # Should not raise any exceptions
    assert config.validate() is True


@patch.dict('os.environ', {'ANTHROPIC_API_KEY': ''})
def test_config_validation_missing_api_key():
    """Test validation fails when API key is missing"""
    config = AppConfig()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is required"):
        config.validate()


@patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key', 'CLAUDE_MODEL': 'invalid-model'})
def test_config_validation_invalid_claude_model():
    """Test validation fails with invalid Claude model"""
    config = AppConfig()
    with pytest.raises(ValueError, match="Invalid Claude model: invalid-model"):
        config.validate()


def test_config_validation_chunk_size_overlap():
    """Test validation of chunk size vs overlap"""
    config = AppConfig()
    # Temporarily modify config to test validation
    config._config['processing']['chunk_overlap'] = 1000  # Same as chunk_size
    with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
        config.validate()


def test_config_validation_thresholds():
    """Test validation of similarity and relevancy thresholds"""
    config = AppConfig()

    # Test invalid similarity threshold
    config._config['processing']['similarity_threshold'] = 1.5
    with pytest.raises(ValueError, match="similarity_threshold.*Must be between 0 and 1"):
        config.validate()

    # Reset and test invalid relevancy threshold
    config._config['processing']['similarity_threshold'] = 0.2
    config._config['processing']['relevancy_threshold'] = -0.1
    with pytest.raises(ValueError, match="relevancy_threshold.*Must be between 0 and 1"):
        config.validate()


def test_config_validation_file_extensions():
    """Test validation of file extensions"""
    config = AppConfig()

    # Test empty extensions list
    config._config['processing']['supported_file_extensions'] = []
    with pytest.raises(ValueError, match="supported_file_extensions cannot be empty"):
        config.validate()

    # Test invalid extension format
    config._config['processing']['supported_file_extensions'] = ['pdf', 'invalid.ext!']
    with pytest.raises(ValueError, match="File extension must start with"):
        config.validate()

    # Reset to valid extensions
    config._config['processing']['supported_file_extensions'] = ['.pdf', '.docx']

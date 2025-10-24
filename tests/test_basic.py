"""
Basic tests to ensure CI pipeline works.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that basic imports work."""
    try:
        import src
        assert hasattr(src, '__version__')
        assert src.__version__ == "0.1.0"
    except ImportError:
        pytest.fail("Failed to import src module")


def test_basic_functionality():
    """Test basic functionality."""
    # This is a placeholder test
    assert 1 + 1 == 2


def test_project_structure():
    """Test that required directories exist."""
    import os
    
    required_dirs = [
        'src',
        'src/data',
        'src/preprocessing', 
        'src/models',
        'src/training',
        'src/explainability',
        'src/baselines',
        'src/metrics',
        'tests',
        'notebooks',
        'docs',
        'configs',
        'artifacts'
    ]
    
    for dir_path in required_dirs:
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"


if __name__ == "__main__":
    pytest.main([__file__])

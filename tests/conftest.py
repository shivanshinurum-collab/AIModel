"""
tests/conftest.py
=================
Pytest configuration and shared fixtures.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m not slow')"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests requiring GPU"
    )

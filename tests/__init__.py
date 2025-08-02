"""
Test package initialization
Provides common test utilities, fixtures, and configuration for the data ingestion pipeline tests
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Test configuration
TEST_DATA_DIR = project_root / "tests" / "test_data"
TEST_OUTPUT_DIR = project_root / "tests" / "test_output"
TEMP_DIR = tempfile.gettempdir()

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

def setup_test_environment():
    """Setup test environment"""
    # Create test directories
    test_dirs = [
        TEST_DATA_DIR / "csv",
        TEST_DATA_DIR / "json",
        TEST_DATA_DIR / "processed",
        TEST_OUTPUT_DIR / "reports",
        TEST_OUTPUT_DIR / "exports"
    ]
    
    for directory in test_dirs:
        directory.mkdir(parents=True, exist_ok=True)

def cleanup_test_environment():
    """Cleanup test environment"""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Setup test environment on import
setup_test_environment()
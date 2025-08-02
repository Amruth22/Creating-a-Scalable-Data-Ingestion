"""
Pytest configuration and fixtures
Provides common fixtures and test utilities for all tests
"""

import pytest
import pandas as pd
import sqlite3
import tempfile
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory fixture"""
    return TEST_DATA_DIR

@pytest.fixture(scope="session")
def test_output_dir():
    """Test output directory fixture"""
    return TEST_OUTPUT_DIR

@pytest.fixture(scope="function")
def temp_dir():
    """Temporary directory fixture that cleans up after each test"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture(scope="function")
def temp_db():
    """Temporary SQLite database fixture"""
    temp_db_path = tempfile.mktemp(suffix='.db')
    yield temp_db_path
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

@pytest.fixture(scope="session")
def sample_orders_data():
    """Sample orders data for testing"""
    return pd.DataFrame([
        {
            'order_id': 'ORD-2024-001',
            'customer_name': 'John Doe',
            'customer_email': 'john.doe@email.com',
            'product': 'iPhone 15',
            'product_category': 'Electronics',
            'quantity': 1,
            'price': 999.99,
            'discount': 0.0,
            'total_amount': 999.99,
            'order_date': '2024-01-15',
            'store_location': 'New York',
            'source': 'website',
            'created_at': '2024-01-15 10:30:00',
            'updated_at': '2024-01-15 10:30:00'
        },
        {
            'order_id': 'ORD-2024-002',
            'customer_name': 'Jane Smith',
            'customer_email': 'jane.smith@email.com',
            'product': 'MacBook Pro',
            'product_category': 'Electronics',
            'quantity': 1,
            'price': 1999.99,
            'discount': 100.0,
            'total_amount': 1899.99,
            'order_date': '2024-01-16',
            'store_location': 'Los Angeles',
            'source': 'store',
            'created_at': '2024-01-16 14:20:00',
            'updated_at': '2024-01-16 14:20:00'
        },
        {
            'order_id': 'ORD-2024-003',
            'customer_name': 'Bob Wilson',
            'customer_email': 'bob.wilson@email.com',
            'product': 'AirPods Pro',
            'product_category': 'Electronics',
            'quantity': 2,
            'price': 249.99,
            'discount': 0.0,
            'total_amount': 499.98,
            'order_date': '2024-01-17',
            'store_location': 'Chicago',
            'source': 'mobile_app',
            'created_at': '2024-01-17 09:15:00',
            'updated_at': '2024-01-17 09:15:00'
        }
    ])

@pytest.fixture(scope="session")
def invalid_orders_data():
    """Invalid orders data for testing validation"""
    return pd.DataFrame([
        {
            'order_id': 'INVALID-ID',  # Invalid format
            'customer_name': '',  # Empty name
            'customer_email': 'invalid-email',  # Invalid email
            'product': 'Test Product',
            'product_category': 'Electronics',
            'quantity': -1,  # Invalid quantity
            'price': -100.0,  # Invalid price
            'discount': 0.0,
            'total_amount': -100.0,
            'order_date': '2025-12-31',  # Future date
            'store_location': 'Test Store',
            'source': 'test',
            'created_at': '2024-01-15 10:30:00',
            'updated_at': '2024-01-15 10:30:00'
        },
        {
            'order_id': 'ORD-2024-004',
            'customer_name': 'Test Customer',
            'customer_email': 'test@example.com',
            'product': '',  # Empty product
            'product_category': 'Electronics',
            'quantity': 1000000,  # Unreasonably high quantity
            'price': 0.0,  # Zero price
            'discount': 1000.0,  # Discount higher than price
            'total_amount': 999.0,  # Incorrect calculation
            'order_date': '1990-01-01',  # Very old date
            'store_location': 'Test Store',
            'source': 'test',
            'created_at': '2024-01-15 10:30:00',
            'updated_at': '2024-01-15 10:30:00'
        }
    ])

@pytest.fixture(scope="session")
def sample_customers_data():
    """Sample customers data for testing"""
    return pd.DataFrame([
        {
            'customer_id': 'CUST-001',
            'customer_name': 'John Doe',
            'email': 'john.doe@email.com',
            'phone': '+1-555-0123',
            'address': '123 Main St, New York, NY 10001',
            'city': 'New York',
            'state': 'NY',
            'country': 'USA',
            'postal_code': '10001',
            'registration_date': '2023-01-15',
            'last_order_date': '2024-01-15',
            'total_orders': 5,
            'total_spent': 2499.95,
            'customer_segment': 'Premium',
            'created_at': '2023-01-15 10:00:00',
            'updated_at': '2024-01-15 10:30:00'
        },
        {
            'customer_id': 'CUST-002',
            'customer_name': 'Jane Smith',
            'email': 'jane.smith@email.com',
            'phone': '+1-555-0124',
            'address': '456 Oak Ave, Los Angeles, CA 90210',
            'city': 'Los Angeles',
            'state': 'CA',
            'country': 'USA',
            'postal_code': '90210',
            'registration_date': '2023-02-20',
            'last_order_date': '2024-01-16',
            'total_orders': 3,
            'total_spent': 3899.97,
            'customer_segment': 'VIP',
            'created_at': '2023-02-20 15:30:00',
            'updated_at': '2024-01-16 14:20:00'
        }
    ])

@pytest.fixture(scope="function")
def sample_csv_file(temp_dir, sample_orders_data):
    """Create a sample CSV file for testing"""
    csv_file = temp_dir / "test_orders.csv"
    sample_orders_data.to_csv(csv_file, index=False)
    return csv_file

@pytest.fixture(scope="function")
def sample_json_file(temp_dir, sample_orders_data):
    """Create a sample JSON file for testing"""
    json_file = temp_dir / "test_orders.json"
    
    # Convert DataFrame to JSON structure
    orders_data = {
        "app_version": "2.1.0",
        "upload_time": "2024-01-15T12:00:00Z",
        "orders": sample_orders_data.to_dict('records')
    }
    
    with open(json_file, 'w') as f:
        json.dump(orders_data, f, indent=2)
    
    return json_file

@pytest.fixture(scope="function")
def sample_database(temp_db, sample_orders_data):
    """Create a sample SQLite database for testing"""
    conn = sqlite3.connect(temp_db)
    
    # Create orders table
    conn.execute('''
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id VARCHAR(50) UNIQUE NOT NULL,
            customer_name VARCHAR(100) NOT NULL,
            customer_email VARCHAR(100),
            product VARCHAR(200) NOT NULL,
            product_category VARCHAR(50),
            quantity INTEGER NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            discount DECIMAL(10,2) DEFAULT 0,
            total_amount DECIMAL(10,2),
            order_date DATE NOT NULL,
            store_location VARCHAR(100),
            source VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert sample data
    sample_orders_data.to_sql('orders', conn, if_exists='append', index=False)
    
    conn.close()
    return temp_db

@pytest.fixture(scope="function")
def mock_api_response():
    """Mock API response data for testing"""
    return {
        "orders": [
            {
                "id": 1,
                "userId": 1,
                "title": "iPhone 15 Pro Max",
                "body": "Latest iPhone model with advanced features"
            },
            {
                "id": 2,
                "userId": 2,
                "title": "MacBook Pro 16-inch",
                "body": "Professional laptop for developers"
            }
        ],
        "total_count": 2,
        "has_more": False
    }

@pytest.fixture(scope="function")
def pipeline_config():
    """Sample pipeline configuration for testing"""
    return {
        "pipeline": {
            "name": "test_pipeline",
            "batch_size": 100,
            "max_workers": 2,
            "timeout_seconds": 300
        },
        "stages": {
            "ingestion": {"enabled": True},
            "validation": {"enabled": True},
            "transformation": {"enabled": True},
            "storage": {"enabled": True}
        },
        "data_quality": {
            "quality_threshold_percent": 80,
            "error_threshold_percent": 5
        }
    }

@pytest.fixture(scope="function")
def validation_config():
    """Sample validation configuration for testing"""
    return {
        "required_fields": ["order_id", "customer_name", "product", "quantity", "price", "order_date"],
        "optional_fields": ["customer_email", "product_category", "discount", "total_amount"],
        "validation_rules": {
            "order_id": {
                "pattern": "^[A-Z]{3}-\\d{4}-\\d{3}$",
                "required": True
            },
            "customer_name": {
                "min_length": 2,
                "max_length": 100,
                "required": True
            },
            "quantity": {
                "min_value": 1,
                "max_value": 1000,
                "data_type": "integer",
                "required": True
            },
            "price": {
                "min_value": 0.01,
                "max_value": 50000.00,
                "data_type": "float",
                "required": True
            }
        }
    }

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test"""
    yield
    # Cleanup any test files that might have been created
    test_patterns = [
        "test_*.csv",
        "test_*.json",
        "test_*.db",
        "test_*.log"
    ]
    
    for pattern in test_patterns:
        for file_path in Path(".").glob(pattern):
            try:
                file_path.unlink()
            except:
                pass

# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def create_test_csv(file_path: Path, data: pd.DataFrame):
        """Create a test CSV file"""
        data.to_csv(file_path, index=False)
        return file_path
    
    @staticmethod
    def create_test_json(file_path: Path, data: dict):
        """Create a test JSON file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return file_path
    
    @staticmethod
    def create_test_database(db_path: str, table_name: str, data: pd.DataFrame):
        """Create a test SQLite database"""
        conn = sqlite3.connect(db_path)
        data.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        return db_path
    
    @staticmethod
    def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype=False):
        """Assert that two DataFrames are equal"""
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    
    @staticmethod
    def assert_file_exists(file_path: Path):
        """Assert that a file exists"""
        assert file_path.exists(), f"File does not exist: {file_path}"
    
    @staticmethod
    def assert_file_not_empty(file_path: Path):
        """Assert that a file is not empty"""
        assert file_path.stat().st_size > 0, f"File is empty: {file_path}"
    
    @staticmethod
    def get_test_timestamp():
        """Get a test timestamp"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

@pytest.fixture(scope="session")
def test_utils():
    """Test utilities fixture"""
    return TestUtils

# Pytest configuration
def pytest_configure(config):
    """Pytest configuration"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring API access"
    )
    config.addinivalue_line(
        "markers", "database: mark test as requiring database access"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers based on test file names
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests
        if "unit" in item.nodeid or "test_" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Mark API tests
        if "api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        
        # Mark database tests
        if "database" in item.nodeid or "db" in item.nodeid:
            item.add_marker(pytest.mark.database)
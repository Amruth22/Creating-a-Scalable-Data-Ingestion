"""
Database setup script for data ingestion pipeline
Creates SQLite database and tables for storing orders and pipeline metadata
"""

import sqlite3
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import config
from utils.helpers import ensure_directory_exists

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_database_schema(db_path: str) -> bool:
    """
    Create database schema with all required tables
    
    Args:
        db_path (str): Path to SQLite database file
        
    Returns:
        bool: True if successful
    """
    try:
        # Ensure database directory exists
        ensure_directory_exists(os.path.dirname(db_path))
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                customer_name TEXT NOT NULL,
                product TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                order_date TEXT NOT NULL,
                source TEXT,
                store_location TEXT,
                customer_email TEXT,
                product_category TEXT,
                discount REAL DEFAULT 0.0,
                total_amount REAL,
                processed_at TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create customers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                address TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT,
                price REAL NOT NULL,
                in_stock BOOLEAN DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create pipeline_runs table for tracking pipeline executions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                pipeline_name TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                status TEXT NOT NULL,
                records_processed INTEGER DEFAULT 0,
                records_failed INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create data_quality_metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_type TEXT NOT NULL,
                source_table TEXT,
                measured_at TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
            )
        ''')
        
        # Create processing_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
            )
        ''')
        
        # Create file_processing_history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_processing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER,
                file_hash TEXT,
                source_type TEXT NOT NULL,
                processing_status TEXT NOT NULL,
                records_count INTEGER DEFAULT 0,
                error_message TEXT,
                processed_at TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_customer_name ON orders(customer_name)",
            "CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_date)",
            "CREATE INDEX IF NOT EXISTS idx_orders_source ON orders(source)",
            "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_start_time ON pipeline_runs(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_run_id ON data_quality_metrics(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_processing_logs_run_id ON processing_logs(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_file_processing_filename ON file_processing_history(filename)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Database schema created successfully at {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating database schema: {e}")
        return False

def insert_sample_data(db_path: str) -> bool:
    """
    Insert sample data for testing and demonstration
    
    Args:
        db_path (str): Path to SQLite database file
        
    Returns:
        bool: True if successful
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Sample customers
        sample_customers = [
            ('CUST-001', 'John Doe', 'john.doe@email.com', '+1-555-0123', '123 Main St, New York, NY'),
            ('CUST-002', 'Jane Smith', 'jane.smith@email.com', '+1-555-0124', '456 Oak Ave, Los Angeles, CA'),
            ('CUST-003', 'Bob Wilson', 'bob.wilson@email.com', '+1-555-0125', '789 Pine Rd, Chicago, IL'),
            ('CUST-004', 'Alice Johnson', 'alice.johnson@email.com', '+1-555-0126', '321 Elm St, Houston, TX'),
            ('CUST-005', 'Charlie Brown', 'charlie.brown@email.com', '+1-555-0127', '654 Maple Dr, Phoenix, AZ')
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO customers (customer_id, name, email, phone, address)
            VALUES (?, ?, ?, ?, ?)
        ''', sample_customers)
        
        # Sample products
        sample_products = [
            ('PROD-001', 'iPhone 15', 'Electronics', 999.99, 1),
            ('PROD-002', 'MacBook Pro', 'Electronics', 1999.99, 1),
            ('PROD-003', 'AirPods Pro', 'Electronics', 249.99, 1),
            ('PROD-004', 'iPad Air', 'Electronics', 599.99, 1),
            ('PROD-005', 'Apple Watch', 'Electronics', 399.99, 1),
            ('PROD-006', 'Samsung Galaxy S24', 'Electronics', 899.99, 1),
            ('PROD-007', 'Dell XPS 13', 'Electronics', 1299.99, 1),
            ('PROD-008', 'Sony WH-1000XM4', 'Electronics', 349.99, 1)
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO products (product_id, name, category, price, in_stock)
            VALUES (?, ?, ?, ?, ?)
        ''', sample_products)
        
        # Sample orders
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sample_orders = [
            ('ORD-2024-001', 'John Doe', 'iPhone 15', 1, 999.99, '2024-01-15', 'website', None, 'john.doe@email.com', 'Electronics', 0.0, 999.99, current_time),
            ('ORD-2024-002', 'Jane Smith', 'MacBook Pro', 1, 1999.99, '2024-01-15', 'store', 'New York', 'jane.smith@email.com', 'Electronics', 100.0, 1899.99, current_time),
            ('ORD-2024-003', 'Bob Wilson', 'AirPods Pro', 2, 249.99, '2024-01-16', 'mobile_app', None, 'bob.wilson@email.com', 'Electronics', 0.0, 499.98, current_time),
            ('ORD-2024-004', 'Alice Johnson', 'iPad Air', 1, 599.99, '2024-01-16', 'website', None, 'alice.johnson@email.com', 'Electronics', 50.0, 549.99, current_time),
            ('ORD-2024-005', 'Charlie Brown', 'Apple Watch', 1, 399.99, '2024-01-17', 'store', 'Los Angeles', 'charlie.brown@email.com', 'Electronics', 0.0, 399.99, current_time)
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO orders (
                order_id, customer_name, product, quantity, price, order_date, 
                source, store_location, customer_email, product_category, 
                discount, total_amount, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', sample_orders)
        
        # Sample pipeline run
        run_id = f"RUN-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cursor.execute('''
            INSERT OR IGNORE INTO pipeline_runs (
                run_id, pipeline_name, start_time, end_time, status, 
                records_processed, records_failed
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (run_id, 'sample_data_setup', current_time, current_time, 'completed', 5, 0))
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info("✅ Sample data inserted successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error inserting sample data: {e}")
        return False

def verify_database(db_path: str) -> bool:
    """
    Verify database setup by checking tables and data
    
    Args:
        db_path (str): Path to SQLite database file
        
    Returns:
        bool: True if verification successful
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if all tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'orders', 'customers', 'products', 'pipeline_runs',
            'data_quality_metrics', 'processing_logs', 'file_processing_history'
        ]
        
        missing_tables = [table for table in expected_tables if table not in tables]
        if missing_tables:
            logger.error(f"❌ Missing tables: {missing_tables}")
            return False
        
        # Check data counts
        table_counts = {}
        for table in expected_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            table_counts[table] = count
        
        conn.close()
        
        # Print verification results
        logger.info("📊 Database verification results:")
        for table, count in table_counts.items():
            logger.info(f"  {table}: {count} records")
        
        logger.info("✅ Database verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error verifying database: {e}")
        return False

def main():
    """Main function to set up the database"""
    logger.info("🚀 Starting database setup...")
    
    try:
        # Get database path from configuration
        db_path = config.database.path
        logger.info(f"Database path: {db_path}")
        
        # Create database schema
        if not create_database_schema(db_path):
            logger.error("❌ Failed to create database schema")
            return 1
        
        # Insert sample data
        if not insert_sample_data(db_path):
            logger.error("❌ Failed to insert sample data")
            return 1
        
        # Verify database
        if not verify_database(db_path):
            logger.error("❌ Database verification failed")
            return 1
        
        logger.info("🎉 Database setup completed successfully!")
        logger.info(f"📍 Database location: {os.path.abspath(db_path)}")
        logger.info("💡 You can now run the data ingestion pipeline!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
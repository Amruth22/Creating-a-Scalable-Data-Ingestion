"""
Database setup script
Creates SQLite database and initializes required tables
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# Add src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

try:
    from src.utils.config import config
    from src.utils.helpers import ensure_directory_exists
except ImportError:
    # Fallback for basic functionality
    class Config:
        class Database:
            path = "data/orders.db"
    config = Config()
    config.database = Config.Database()
    
    def ensure_directory_exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database_tables(db_path: str) -> bool:
    """
    Create database tables
    
    Args:
        db_path (str): Path to database file
        
    Returns:
        bool: True if successful
    """
    try:
        # Ensure database directory exists
        ensure_directory_exists(os.path.dirname(db_path))
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create orders table with all transformation fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                customer_name TEXT NOT NULL,
                customer_email TEXT,
                product TEXT NOT NULL,
                product_category TEXT,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                discount REAL DEFAULT 0.0,
                total_amount REAL,
                order_date TEXT NOT NULL,
                source TEXT,
                source_type TEXT,
                store_location TEXT,
                notes TEXT,
                
                -- API ingestion fields
                api_post_id INTEGER,
                ingested_at TEXT,
                
                -- Data cleaning fields
                cleaned_at TEXT,
                data_quality_score REAL,
                
                -- Data standardization fields
                standardized_at TEXT,
                standardization_version REAL,
                
                -- Data enrichment fields
                unit_price_after_discount REAL,
                discount_percentage REAL,
                estimated_profit REAL,
                product_brand TEXT,
                product_subcategory TEXT,
                price_tier TEXT,
                popularity_score INTEGER,
                customer_segment TEXT,
                customer_ltv REAL,
                customer_priority INTEGER,
                order_year INTEGER,
                order_month INTEGER,
                order_day INTEGER,
                order_weekday TEXT,
                order_week_of_year INTEGER,
                order_quarter INTEGER,
                is_weekend BOOLEAN,
                is_month_end BOOLEAN,
                is_month_start BOOLEAN,
                days_since_order INTEGER,
                order_timing TEXT,
                order_size_category TEXT,
                quantity_category TEXT,
                revenue_contribution_pct REAL,
                channel_type TEXT,
                risk_score INTEGER,
                risk_level TEXT,
                seasonal_factor REAL,
                seasonally_adjusted_amount REAL,
                season TEXT,
                is_holiday_season BOOLEAN,
                is_summer_season BOOLEAN,
                enriched_at TEXT,
                enrichment_version REAL,
                
                -- Processing metadata
                processed_at TEXT,
                processing_stage TEXT,
                
                -- Standard timestamps
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
                company TEXT,
                website TEXT,
                source_type TEXT,
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
                price REAL,
                in_stock BOOLEAN DEFAULT 1,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create pipeline_runs table
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
                metric_type TEXT,
                source_table TEXT,
                measured_at TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create processing_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                log_level TEXT NOT NULL,
                message TEXT NOT NULL,
                module TEXT,
                function TEXT,
                line_number INTEGER,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_customer_name ON orders(customer_name)",
            "CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_date)",
            "CREATE INDEX IF NOT EXISTS idx_orders_source ON orders(source)",
            "CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)",
            "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_run_id ON pipeline_runs(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_data_quality_metrics_run_id ON data_quality_metrics(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_processing_logs_run_id ON processing_logs(run_id)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info(f"Database tables created successfully: {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        return False

def insert_sample_data(db_path: str) -> bool:
    """
    Insert sample data for testing
    
    Args:
        db_path (str): Path to database file
        
    Returns:
        bool: True if successful
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Sample orders
        sample_orders = [
            ('ORD-2024-001', 'John Doe', 'john.doe@email.com', 'iPhone 15', 'Electronics', 1, 999.99, 0.0, 999.99, '2024-01-15', 'website', 'file_csv', 'New York Store', 'Priority order'),
            ('ORD-2024-002', 'Jane Smith', 'jane.smith@email.com', 'MacBook Pro', 'Electronics', 1, 1999.99, 100.0, 1899.99, '2024-01-15', 'store', 'file_csv', 'Los Angeles Store', ''),
            ('ORD-2024-003', 'Bob Wilson', 'bob.wilson@email.com', 'AirPods Pro', 'Electronics', 2, 249.99, 0.0, 499.98, '2024-01-16', 'mobile_app', 'file_json', '', 'Gift order'),
            ('ORD-2024-004', 'Alice Johnson', 'alice.johnson@email.com', 'iPad Air', 'Electronics', 1, 599.99, 50.0, 549.99, '2024-01-16', 'website', 'api_rest', '', ''),
            ('ORD-2024-005', 'Charlie Brown', 'charlie.brown@email.com', 'Apple Watch', 'Electronics', 1, 399.99, 0.0, 399.99, '2024-01-17', 'store', 'file_csv', 'Chicago Store', 'Express delivery')
        ]
        
        # Insert orders (ignore if already exists)
        for order in sample_orders:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO orders 
                    (order_id, customer_name, customer_email, product, product_category, quantity, price, discount, total_amount, order_date, source, source_type, store_location, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', order)
            except sqlite3.IntegrityError:
                pass  # Order already exists
        
        # Sample customers
        sample_customers = [
            ('CUST-001', 'John Doe', 'john.doe@email.com', '+1-555-0123', '123 Main St, New York, NY', '', '', 'file_csv'),
            ('CUST-002', 'Jane Smith', 'jane.smith@email.com', '+1-555-0124', '456 Oak Ave, Los Angeles, CA', '', '', 'file_csv'),
            ('CUST-003', 'Bob Wilson', 'bob.wilson@email.com', '+1-555-0125', '789 Pine St, Chicago, IL', '', '', 'file_json'),
            ('CUST-004', 'Alice Johnson', 'alice.johnson@email.com', '+1-555-0126', '321 Elm St, Houston, TX', '', '', 'api_rest'),
            ('CUST-005', 'Charlie Brown', 'charlie.brown@email.com', '+1-555-0127', '654 Maple Ave, Phoenix, AZ', '', '', 'file_csv')
        ]
        
        # Insert customers (ignore if already exists)
        for customer in sample_customers:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO customers 
                    (customer_id, name, email, phone, address, company, website, source_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', customer)
            except sqlite3.IntegrityError:
                pass  # Customer already exists
        
        # Sample products
        sample_products = [
            ('PROD-001', 'iPhone 15', 'Electronics', 999.99, 1, 'Latest iPhone model'),
            ('PROD-002', 'MacBook Pro', 'Electronics', 1999.99, 1, 'Professional laptop'),
            ('PROD-003', 'AirPods Pro', 'Electronics', 249.99, 1, 'Wireless earbuds with noise cancellation'),
            ('PROD-004', 'iPad Air', 'Electronics', 599.99, 1, 'Lightweight tablet'),
            ('PROD-005', 'Apple Watch', 'Electronics', 399.99, 1, 'Smartwatch with health features')
        ]
        
        # Insert products (ignore if already exists)
        for product in sample_products:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO products 
                    (product_id, name, category, price, in_stock, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', product)
            except sqlite3.IntegrityError:
                pass  # Product already exists
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info("Sample data inserted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error inserting sample data: {e}")
        return False

def get_database_info(db_path: str) -> dict:
    """
    Get database information
    
    Args:
        db_path (str): Path to database file
        
    Returns:
        dict: Database information
    """
    info = {
        'exists': False,
        'size_mb': 0,
        'tables': [],
        'record_counts': {}
    }
    
    try:
        if os.path.exists(db_path):
            info['exists'] = True
            info['size_mb'] = os.path.getsize(db_path) / (1024 * 1024)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            info['tables'] = [table[0] for table in tables]
            
            # Get record counts
            for table_name in info['tables']:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                info['record_counts'][table_name] = count
            
            conn.close()
    
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
    
    return info

def main():
    """Main function"""
    print("🗄️ Database Setup Script")
    print("=" * 50)
    
    # Get database path
    db_path = config.database.path
    print(f"Database path: {db_path}")
    
    # Check if database already exists
    db_info = get_database_info(db_path)
    if db_info['exists']:
        print(f"📊 Existing database found ({db_info['size_mb']:.2f} MB)")
        print(f"📋 Tables: {', '.join(db_info['tables'])}")
        for table, count in db_info['record_counts'].items():
            print(f"   - {table}: {count:,} records")
        
        response = input("\n❓ Database already exists. Recreate? (y/N): ").lower()
        if response != 'y':
            print("✅ Using existing database")
            return 0
    
    # Create database tables
    print("\n🔧 Creating database tables...")
    if create_database_tables(db_path):
        print("✅ Database tables created successfully")
    else:
        print("❌ Failed to create database tables")
        return 1
    
    # Insert sample data
    print("\n📊 Inserting sample data...")
    if insert_sample_data(db_path):
        print("✅ Sample data inserted successfully")
    else:
        print("❌ Failed to insert sample data")
        return 1
    
    # Show final database info
    print("\n📈 Final Database Information:")
    final_info = get_database_info(db_path)
    print(f"📊 Database size: {final_info['size_mb']:.2f} MB")
    print(f"📋 Tables created: {len(final_info['tables'])}")
    for table, count in final_info['record_counts'].items():
        print(f"   - {table}: {count:,} records")
    
    print("\n🎉 Database setup completed successfully!")
    print(f"💡 You can now run: python scripts/run_pipeline.py")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
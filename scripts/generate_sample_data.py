"""
Sample data generation script for data ingestion pipeline
Creates realistic sample CSV and JSON files for testing and demonstration
"""

import os
import sys
import json
import csv
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import config
from utils.helpers import ensure_directory_exists, write_csv_file, write_json_file
from utils.constants import SampleData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """Generate realistic sample data for testing the pipeline"""
    
    def __init__(self):
        """Initialize the sample data generator"""
        self.products = [
            {"name": "iPhone 15", "category": "Electronics", "price_range": (999, 1199)},
            {"name": "MacBook Pro", "category": "Electronics", "price_range": (1999, 2499)},
            {"name": "AirPods Pro", "category": "Electronics", "price_range": (249, 299)},
            {"name": "iPad Air", "category": "Electronics", "price_range": (599, 799)},
            {"name": "Apple Watch", "category": "Electronics", "price_range": (399, 499)},
            {"name": "Samsung Galaxy S24", "category": "Electronics", "price_range": (899, 1099)},
            {"name": "Dell XPS 13", "category": "Electronics", "price_range": (1299, 1599)},
            {"name": "Sony WH-1000XM4", "category": "Electronics", "price_range": (349, 399)},
            {"name": "Nintendo Switch", "category": "Gaming", "price_range": (299, 349)},
            {"name": "PlayStation 5", "category": "Gaming", "price_range": (499, 599)},
            {"name": "Xbox Series X", "category": "Gaming", "price_range": (499, 599)},
            {"name": "Kindle Paperwhite", "category": "Books", "price_range": (139, 189)},
            {"name": "Echo Dot", "category": "Smart Home", "price_range": (49, 79)},
            {"name": "Ring Doorbell", "category": "Smart Home", "price_range": (199, 249)},
            {"name": "Fitbit Charge 5", "category": "Fitness", "price_range": (179, 229)}
        ]
        
        self.customers = [
            {"name": "John Doe", "email": "john.doe@email.com"},
            {"name": "Jane Smith", "email": "jane.smith@email.com"},
            {"name": "Bob Wilson", "email": "bob.wilson@email.com"},
            {"name": "Alice Johnson", "email": "alice.johnson@email.com"},
            {"name": "Charlie Brown", "email": "charlie.brown@email.com"},
            {"name": "Diana Prince", "email": "diana.prince@email.com"},
            {"name": "Edward Norton", "email": "edward.norton@email.com"},
            {"name": "Fiona Green", "email": "fiona.green@email.com"},
            {"name": "George Miller", "email": "george.miller@email.com"},
            {"name": "Helen Davis", "email": "helen.davis@email.com"},
            {"name": "Ivan Petrov", "email": "ivan.petrov@email.com"},
            {"name": "Julia Roberts", "email": "julia.roberts@email.com"},
            {"name": "Kevin Hart", "email": "kevin.hart@email.com"},
            {"name": "Linda Williams", "email": "linda.williams@email.com"},
            {"name": "Michael Jordan", "email": "michael.jordan@email.com"}
        ]
        
        self.sources = ["website", "mobile_app", "store", "phone", "partner"]
        self.store_locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
        
        self.order_counter = 1
    
    def generate_order_id(self) -> str:
        """Generate a unique order ID"""
        order_id = f"ORD-2024-{self.order_counter:03d}"
        self.order_counter += 1
        return order_id
    
    def generate_random_date(self, days_back: int = 30) -> str:
        """Generate a random date within the last N days"""
        start_date = datetime.now() - timedelta(days=days_back)
        random_days = random.randint(0, days_back)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime("%Y-%m-%d")
    
    def generate_order(self) -> Dict[str, Any]:
        """Generate a single order record"""
        product = random.choice(self.products)
        customer = random.choice(self.customers)
        source = random.choice(self.sources)
        
        # Generate price with some variation
        min_price, max_price = product["price_range"]
        price = round(random.uniform(min_price, max_price), 2)
        
        # Generate quantity (mostly 1, sometimes 2-3)
        quantity = random.choices([1, 2, 3], weights=[70, 20, 10])[0]
        
        # Generate discount (sometimes)
        discount = 0.0
        if random.random() < 0.2:  # 20% chance of discount
            discount = round(random.uniform(10, 100), 2)
        
        order = {
            "order_id": self.generate_order_id(),
            "customer_name": customer["name"],
            "customer_email": customer["email"],
            "product": product["name"],
            "product_category": product["category"],
            "quantity": quantity,
            "price": price,
            "discount": discount,
            "total_amount": round((price * quantity) - discount, 2),
            "order_date": self.generate_random_date(),
            "source": source
        }
        
        # Add store location for store orders
        if source == "store":
            order["store_location"] = random.choice(self.store_locations)
        
        # Sometimes add notes
        if random.random() < 0.1:  # 10% chance
            notes = [
                "Customer requested expedited shipping",
                "Gift wrapping requested",
                "Delivery to office address",
                "Customer called to confirm order",
                "Special handling required"
            ]
            order["notes"] = random.choice(notes)
        
        return order
    
    def generate_csv_file(self, filename: str, num_records: int = 50) -> bool:
        """
        Generate CSV file with sample orders
        
        Args:
            filename (str): Output filename
            num_records (int): Number of records to generate
            
        Returns:
            bool: True if successful
        """
        try:
            # Generate orders
            orders = [self.generate_order() for _ in range(num_records)]
            
            # Convert to DataFrame and save as CSV
            import pandas as pd
            df = pd.DataFrame(orders)
            
            # Ensure output directory exists
            file_path = os.path.join(config.file.input_dir, "csv", filename)
            ensure_directory_exists(os.path.dirname(file_path))
            
            # Save CSV file
            df.to_csv(file_path, index=False)
            
            logger.info(f"✅ Generated CSV file: {file_path} ({num_records} records)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating CSV file {filename}: {e}")
            return False
    
    def generate_json_file(self, filename: str, num_records: int = 30) -> bool:
        """
        Generate JSON file with sample orders
        
        Args:
            filename (str): Output filename
            num_records (int): Number of records to generate
            
        Returns:
            bool: True if successful
        """
        try:
            # Generate orders
            orders = [self.generate_order() for _ in range(num_records)]
            
            # Create JSON structure
            json_data = {
                "app_version": "2.1.0",
                "upload_time": datetime.now().isoformat(),
                "total_orders": len(orders),
                "orders": orders
            }
            
            # Ensure output directory exists
            file_path = os.path.join(config.file.input_dir, "json", filename)
            ensure_directory_exists(os.path.dirname(file_path))
            
            # Save JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            logger.info(f"✅ Generated JSON file: {file_path} ({num_records} records)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating JSON file {filename}: {e}")
            return False
    
    def generate_corrupted_csv(self, filename: str) -> bool:
        """
        Generate a CSV file with some data quality issues for testing
        
        Args:
            filename (str): Output filename
            
        Returns:
            bool: True if successful
        """
        try:
            orders = []
            
            # Generate some good records
            for _ in range(15):
                orders.append(self.generate_order())
            
            # Add some problematic records
            problematic_orders = [
                {
                    "order_id": "",  # Missing order ID
                    "customer_name": "Test Customer",
                    "product": "Test Product",
                    "quantity": 1,
                    "price": 99.99,
                    "order_date": "2024-01-15",
                    "source": "website"
                },
                {
                    "order_id": "ORD-2024-999",
                    "customer_name": "",  # Missing customer name
                    "product": "Test Product",
                    "quantity": 1,
                    "price": 99.99,
                    "order_date": "2024-01-15",
                    "source": "website"
                },
                {
                    "order_id": "ORD-2024-998",
                    "customer_name": "Test Customer",
                    "product": "Test Product",
                    "quantity": -1,  # Negative quantity
                    "price": 99.99,
                    "order_date": "2024-01-15",
                    "source": "website"
                },
                {
                    "order_id": "ORD-2024-997",
                    "customer_name": "Test Customer",
                    "product": "Test Product",
                    "quantity": 1,
                    "price": -50.0,  # Negative price
                    "order_date": "2024-01-15",
                    "source": "website"
                },
                {
                    "order_id": "ORD-2024-996",
                    "customer_name": "Test Customer",
                    "product": "Test Product",
                    "quantity": 1,
                    "price": 99.99,
                    "order_date": "invalid-date",  # Invalid date
                    "source": "website"
                }
            ]
            
            orders.extend(problematic_orders)
            
            # Convert to DataFrame and save
            import pandas as pd
            df = pd.DataFrame(orders)
            
            file_path = os.path.join(config.file.input_dir, "csv", filename)
            ensure_directory_exists(os.path.dirname(file_path))
            
            df.to_csv(file_path, index=False)
            
            logger.info(f"✅ Generated corrupted CSV file: {file_path} ({len(orders)} records, {len(problematic_orders)} with issues)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating corrupted CSV file {filename}: {e}")
            return False
    
    def generate_large_csv(self, filename: str, num_records: int = 1000) -> bool:
        """
        Generate a large CSV file for performance testing
        
        Args:
            filename (str): Output filename
            num_records (int): Number of records to generate
            
        Returns:
            bool: True if successful
        """
        try:
            file_path = os.path.join(config.file.input_dir, "csv", filename)
            ensure_directory_exists(os.path.dirname(file_path))
            
            # Generate data in batches to avoid memory issues
            batch_size = 100
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'order_id', 'customer_name', 'customer_email', 'product', 
                    'product_category', 'quantity', 'price', 'discount', 
                    'total_amount', 'order_date', 'source', 'store_location', 'notes'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for batch_start in range(0, num_records, batch_size):
                    batch_end = min(batch_start + batch_size, num_records)
                    batch_orders = [self.generate_order() for _ in range(batch_end - batch_start)]
                    
                    for order in batch_orders:
                        # Ensure all fields are present
                        row = {field: order.get(field, '') for field in fieldnames}
                        writer.writerow(row)
            
            logger.info(f"✅ Generated large CSV file: {file_path} ({num_records} records)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating large CSV file {filename}: {e}")
            return False

def create_sample_files():
    """Create various sample files for testing"""
    generator = SampleDataGenerator()
    
    logger.info("📊 Generating sample data files...")
    
    # Generate regular CSV files
    generator.generate_csv_file("orders_daily_20240115.csv", 50)
    generator.generate_csv_file("orders_daily_20240116.csv", 45)
    generator.generate_csv_file("orders_daily_20240117.csv", 60)
    
    # Generate JSON files
    generator.generate_json_file("mobile_orders_20240115.json", 30)
    generator.generate_json_file("mobile_orders_20240116.json", 25)
    
    # Generate a corrupted file for testing error handling
    generator.generate_corrupted_csv("orders_corrupted_20240118.csv")
    
    # Generate a large file for performance testing
    generator.generate_large_csv("orders_large_dataset.csv", 1000)
    
    logger.info("✅ Sample data generation completed!")

def create_sample_config_files():
    """Create sample configuration files"""
    logger.info("⚙️ Creating sample configuration files...")
    
    try:
        # Create config directory
        config_dir = "config"
        ensure_directory_exists(config_dir)
        
        # Pipeline configuration
        pipeline_config = {
            "database": {
                "path": "data/orders.db",
                "timeout": 30,
                "check_same_thread": False
            },
            "api": {
                "base_url": "https://jsonplaceholder.typicode.com",
                "timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 5
            },
            "file": {
                "input_dir": "data/input",
                "output_dir": "data/output",
                "processed_dir": "data/input/processed",
                "archive_dir": "data/archive",
                "max_file_size_mb": 100
            },
            "pipeline": {
                "batch_size": 1000,
                "max_workers": 4,
                "enable_parallel_processing": True,
                "enable_monitoring": True,
                "log_level": "INFO"
            },
            "alerts": {
                "enable_email_alerts": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "email_from": "",
                "email_to": "",
                "email_password": ""
            }
        }
        
        config_file = os.path.join(config_dir, "pipeline_config.yaml")
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(pipeline_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Created pipeline configuration: {config_file}")
        
        # API endpoints configuration
        api_config = {
            "endpoints": {
                "orders": "/posts",
                "users": "/users",
                "comments": "/comments"
            },
            "headers": {
                "Content-Type": "application/json",
                "User-Agent": "DataIngestionPipeline/1.0"
            },
            "rate_limiting": {
                "requests_per_minute": 60,
                "burst_limit": 10
            }
        }
        
        api_config_file = os.path.join(config_dir, "api_config.yaml")
        with open(api_config_file, 'w') as f:
            yaml.dump(api_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Created API configuration: {api_config_file}")
        
    except Exception as e:
        logger.error(f"❌ Error creating configuration files: {e}")

def main():
    """Main function to generate all sample data"""
    logger.info("🚀 Starting sample data generation...")
    
    try:
        # Create sample data files
        create_sample_files()
        
        # Create sample configuration files
        create_sample_config_files()
        
        # Create directory structure
        directories = [
            "data/samples",
            "data/output/reports",
            "logs",
            "data/archive"
        ]
        
        for directory in directories:
            ensure_directory_exists(directory)
        
        logger.info("🎉 Sample data generation completed successfully!")
        logger.info("💡 You can now test the data ingestion pipeline with the generated files!")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 SAMPLE DATA GENERATION SUMMARY")
        print("="*60)
        print("Generated files:")
        print("  📁 CSV files: data/input/csv/")
        print("  📄 JSON files: data/input/json/")
        print("  ⚙️ Config files: config/")
        print("  📝 Logs directory: logs/")
        print("\nNext steps:")
        print("  1. Run: python scripts/setup_database.py")
        print("  2. Run: python scripts/run_pipeline.py")
        print("  3. Check: data/output/ for results")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Sample data generation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
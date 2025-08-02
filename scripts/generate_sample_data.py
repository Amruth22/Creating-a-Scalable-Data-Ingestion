"""
Sample data generation script
Creates sample CSV and JSON files for testing the data ingestion pipeline
"""

import os
import sys
import json
import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

try:
    from src.utils.helpers import ensure_directory_exists
except ImportError:
    def ensure_directory_exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)

def generate_sample_orders(num_orders: int = 50) -> list:
    """Generate sample order data"""
    
    # Sample data pools
    customers = [
        'John Doe', 'Jane Smith', 'Bob Wilson', 'Alice Johnson', 'Charlie Brown',
        'Diana Prince', 'Bruce Wayne', 'Clark Kent', 'Peter Parker', 'Tony Stark',
        'Natasha Romanoff', 'Steve Rogers', 'Wanda Maximoff', 'Vision Android',
        'Scott Lang', 'Hope Van Dyne', 'Carol Danvers', 'Stephen Strange',
        'T\'Challa Udaku', 'Shuri Udaku', 'Sam Wilson', 'Bucky Barnes',
        'Clint Barton', 'Laura Barton', 'Pepper Potts', 'Happy Hogan'
    ]
    
    products = [
        {'name': 'iPhone 15', 'price': 999.99, 'category': 'Electronics'},
        {'name': 'MacBook Pro', 'price': 1999.99, 'category': 'Electronics'},
        {'name': 'AirPods Pro', 'price': 249.99, 'category': 'Electronics'},
        {'name': 'iPad Air', 'price': 599.99, 'category': 'Electronics'},
        {'name': 'Apple Watch', 'price': 399.99, 'category': 'Electronics'},
        {'name': 'Samsung Galaxy S24', 'price': 899.99, 'category': 'Electronics'},
        {'name': 'Nintendo Switch', 'price': 299.99, 'category': 'Gaming'},
        {'name': 'PlayStation 5', 'price': 499.99, 'category': 'Gaming'},
        {'name': 'Xbox Series X', 'price': 499.99, 'category': 'Gaming'},
        {'name': 'Kindle Paperwhite', 'price': 139.99, 'category': 'Electronics'},
        {'name': 'Echo Dot', 'price': 49.99, 'category': 'Smart Home'},
        {'name': 'Ring Doorbell', 'price': 199.99, 'category': 'Smart Home'},
        {'name': 'Fitbit Charge 5', 'price': 179.99, 'category': 'Fitness'},
        {'name': 'Sony WH-1000XM4', 'price': 349.99, 'category': 'Electronics'},
        {'name': 'GoPro Hero 11', 'price': 399.99, 'category': 'Electronics'}
    ]
    
    sources = ['website', 'mobile_app', 'store', 'phone', 'partner']
    
    locations = [
        'New York Store', 'Los Angeles Store', 'Chicago Store', 'Houston Store',
        'Phoenix Store', 'Philadelphia Store', 'San Antonio Store', 'San Diego Store',
        'Dallas Store', 'San Jose Store', 'Austin Store', 'Jacksonville Store',
        'Fort Worth Store', 'Columbus Store', 'Charlotte Store', 'San Francisco Store',
        'Indianapolis Store', 'Seattle Store', 'Denver Store', 'Washington Store'
    ]
    
    orders = []
    
    # Generate orders
    for i in range(num_orders):
        customer = random.choice(customers)
        product = random.choice(products)
        source = random.choice(sources)
        
        # Generate order date (last 90 days)
        days_ago = random.randint(0, 90)
        order_date = datetime.now() - timedelta(days=days_ago)
        
        # Generate quantity (weighted towards 1)
        quantity = random.choices([1, 2, 3, 4, 5], weights=[60, 20, 10, 5, 5])[0]
        
        # Generate discount (80% no discount, 20% some discount)
        discount = 0.0
        if random.random() < 0.2:  # 20% chance of discount
            discount = round(random.uniform(10, 100), 2)
        
        # Calculate total
        subtotal = product['price'] * quantity
        total_amount = subtotal - discount
        
        # Generate email
        first_name = customer.split()[0].lower()
        last_name = customer.split()[-1].lower()
        domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'company.com']
        email = f"{first_name}.{last_name}@{random.choice(domains)}"
        
        order = {
            'order_id': f'ORD-{order_date.year}-{i+1:03d}',
            'customer_name': customer,
            'customer_email': email,
            'product': product['name'],
            'product_category': product['category'],
            'quantity': quantity,
            'price': product['price'],
            'discount': discount,
            'total_amount': round(total_amount, 2),
            'order_date': order_date.strftime('%Y-%m-%d'),
            'source': source,
            'store_location': random.choice(locations) if source == 'store' else '',
            'notes': random.choice(['', 'Priority order', 'Gift order', 'Express delivery', 'Customer pickup']) if random.random() < 0.3 else ''
        }
        
        orders.append(order)
    
    return orders

def create_csv_files(orders: list, output_dir: str):
    """Create sample CSV files"""
    csv_dir = os.path.join(output_dir, 'csv')
    ensure_directory_exists(csv_dir)
    
    # Split orders into multiple files
    chunk_size = len(orders) // 3 + 1
    
    for i in range(0, len(orders), chunk_size):
        chunk = orders[i:i + chunk_size]
        df = pd.DataFrame(chunk)
        
        filename = f"orders_batch_{i//chunk_size + 1}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(csv_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"Created CSV file: {filepath} ({len(chunk)} records)")

def create_json_files(orders: list, output_dir: str):
    """Create sample JSON files"""
    json_dir = os.path.join(output_dir, 'json')
    ensure_directory_exists(json_dir)
    
    # Split orders into multiple files with JSON structure
    chunk_size = len(orders) // 2 + 1
    
    for i in range(0, len(orders), chunk_size):
        chunk = orders[i:i + chunk_size]
        
        json_data = {
            'app_version': '2.1.0',
            'upload_time': datetime.now().isoformat(),
            'total_orders': len(chunk),
            'orders': chunk
        }
        
        filename = f"mobile_orders_{i//chunk_size + 1}_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = os.path.join(json_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"Created JSON file: {filepath} ({len(chunk)} records)")

def create_sample_config_files():
    """Create sample configuration files if they don't exist"""
    config_dir = "config"
    ensure_directory_exists(config_dir)
    
    # Simple database config
    db_config = {
        'database': {
            'path': 'data/orders.db',
            'timeout': 30
        }
    }
    
    db_config_path = os.path.join(config_dir, 'database_config.yaml')
    if not os.path.exists(db_config_path):
        import yaml
        with open(db_config_path, 'w') as f:
            yaml.dump(db_config, f, default_flow_style=False, indent=2)
        print(f"Created config file: {db_config_path}")
    
    # Simple API config
    api_config = {
        'api': {
            'base_url': 'https://jsonplaceholder.typicode.com',
            'timeout': 30,
            'retry_attempts': 3
        }
    }
    
    api_config_path = os.path.join(config_dir, 'api_config.yaml')
    if not os.path.exists(api_config_path):
        import yaml
        with open(api_config_path, 'w') as f:
            yaml.dump(api_config, f, default_flow_style=False, indent=2)
        print(f"Created config file: {api_config_path}")

def main():
    """Main function"""
    print("📊 Sample Data Generator")
    print("=" * 50)
    
    # Create directories
    input_dir = "data/input"
    ensure_directory_exists(input_dir)
    
    # Generate sample orders
    print("🔄 Generating sample orders...")
    orders = generate_sample_orders(75)  # Generate 75 sample orders
    print(f"✅ Generated {len(orders)} sample orders")
    
    # Create CSV files
    print("\n📄 Creating CSV files...")
    create_csv_files(orders, input_dir)
    
    # Create JSON files
    print("\n📄 Creating JSON files...")
    create_json_files(orders, input_dir)
    
    # Create sample config files
    print("\n⚙️ Creating sample configuration files...")
    create_sample_config_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 Sample Data Generation Summary")
    print("=" * 50)
    print(f"✅ Total orders generated: {len(orders)}")
    print(f"📁 CSV files created in: data/input/csv/")
    print(f"📁 JSON files created in: data/input/json/")
    print(f"⚙️ Config files created in: config/")
    
    print("\n🎉 Sample data generation completed!")
    print("💡 You can now run: python scripts/run_pipeline.py")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
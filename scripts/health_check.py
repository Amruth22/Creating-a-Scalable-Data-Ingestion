"""
Health check script
Verifies system status, dependencies, and configuration
"""

import os
import sys
import sqlite3
import requests
import logging
from datetime import datetime
from pathlib import Path

# Add src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Suppress info logs for cleaner output

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check")
    version = sys.version_info
    print(f"   Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python version is compatible")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Dependencies Check")
    
    required_packages = [
        'pandas',
        'numpy', 
        'requests',
        'sqlalchemy',
        'pyyaml',
        'jsonschema',
        'structlog',
        'tqdm',
        'click'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   💡 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_directories():
    """Check required directories"""
    print("\n📁 Directory Structure Check")
    
    required_dirs = [
        'data',
        'data/input',
        'data/input/csv',
        'data/input/json',
        'data/output',
        'logs',
        'config'
    ]
    
    all_exist = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}")
        else:
            print(f"   ❌ {directory} (missing)")
            all_exist = False
            # Create missing directory
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"   🔧 Created {directory}")
            except Exception as e:
                print(f"   ⚠️ Failed to create {directory}: {e}")
    
    return all_exist

def check_database():
    """Check database connectivity"""
    print("\n🗄️ Database Check")
    
    try:
        # Try to import config
        from src.utils.config import config
        db_path = config.database.path
    except ImportError:
        db_path = "data/orders.db"
    
    print(f"   Database path: {db_path}")
    
    if not os.path.exists(db_path):
        print("   ❌ Database file not found")
        print("   💡 Run: python scripts/setup_database.py")
        return False
    
    try:
        # Test database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        required_tables = ['orders', 'customers', 'products', 'pipeline_runs']
        missing_tables = [table for table in required_tables if table not in table_names]
        
        if missing_tables:
            print(f"   ❌ Missing tables: {', '.join(missing_tables)}")
            print("   💡 Run: python scripts/setup_database.py")
            conn.close()
            return False
        
        # Get record counts
        for table in required_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   ✅ {table}: {count:,} records")
        
        conn.close()
        
        # Check database size
        db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"   📊 Database size: {db_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return False

def check_api_connectivity():
    """Check API connectivity"""
    print("\n🌐 API Connectivity Check")
    
    try:
        # Try to import config
        from src.utils.config import config
        base_url = config.api.base_url
    except ImportError:
        base_url = "https://jsonplaceholder.typicode.com"
    
    print(f"   API URL: {base_url}")
    
    try:
        # Test API connection
        response = requests.get(f"{base_url}/posts/1", timeout=10)
        
        if response.status_code == 200:
            print(f"   ✅ API accessible (response time: {response.elapsed.total_seconds():.2f}s)")
            return True
        else:
            print(f"   ❌ API returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API (network error)")
        return False
    except requests.exceptions.Timeout:
        print("   ❌ API request timeout")
        return False
    except Exception as e:
        print(f"   ❌ API error: {e}")
        return False

def check_configuration():
    """Check configuration files"""
    print("\n⚙️ Configuration Check")
    
    config_files = [
        'config/pipeline_config.yaml',
        'config/database_config.yaml',
        'config/api_config.yaml'
    ]
    
    all_exist = True
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"   ✅ {config_file}")
        else:
            print(f"   ⚠️ {config_file} (optional, using defaults)")
    
    # Test config loading
    try:
        from src.utils.config import config
        print("   ✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False

def check_disk_space():
    """Check available disk space"""
    print("\n💾 Disk Space Check")
    
    try:
        import shutil
        
        # Check space in current directory
        total, used, free = shutil.disk_usage('.')
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_percent = (used / total) * 100
        
        print(f"   Total space: {total_gb:.1f} GB")
        print(f"   Free space: {free_gb:.1f} GB")
        print(f"   Used: {used_percent:.1f}%")
        
        if free_gb > 1.0:  # At least 1GB free
            print("   ✅ Sufficient disk space")
            return True
        else:
            print("   ⚠️ Low disk space (less than 1GB free)")
            return False
            
    except Exception as e:
        print(f"   ❌ Cannot check disk space: {e}")
        return False

def check_permissions():
    """Check file permissions"""
    print("\n🔐 Permissions Check")
    
    test_dirs = ['data', 'logs']
    all_writable = True
    
    for directory in test_dirs:
        try:
            # Test write permission
            test_file = os.path.join(directory, f'test_write_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tmp')
            
            # Create directory if it doesn't exist
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Try to write a test file
            with open(test_file, 'w') as f:
                f.write('test')
            
            # Clean up test file
            os.remove(test_file)
            
            print(f"   ✅ {directory} (writable)")
            
        except Exception as e:
            print(f"   ❌ {directory} (not writable): {e}")
            all_writable = False
    
    return all_writable

def run_sample_pipeline_test():
    """Run a minimal pipeline test"""
    print("\n🧪 Sample Pipeline Test")
    
    try:
        # Try to import and create a simple pipeline component
        from src.ingestion.file_ingestion import FileIngestion
        
        # Create file ingestion instance
        file_ingestion = FileIngestion()
        print("   ✅ File ingestion module loaded")
        
        # Test file discovery (should not fail even if no files)
        new_files = file_ingestion.discover_new_files()
        print(f"   ✅ File discovery works ({len(new_files['csv'])} CSV, {len(new_files['json'])} JSON files found)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Pipeline test failed: {e}")
        return False

def main():
    """Main health check function"""
    print("🏥 System Health Check")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Database", check_database),
        ("API Connectivity", check_api_connectivity),
        ("Configuration", check_configuration),
        ("Disk Space", check_disk_space),
        ("Permissions", check_permissions),
        ("Pipeline Test", run_sample_pipeline_test)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Health Check Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} checks passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All systems are healthy! You can run the pipeline.")
        print("💡 Next step: python scripts/run_pipeline.py")
        return 0
    else:
        print(f"\n⚠️ {total - passed} issues found. Please fix them before running the pipeline.")
        
        # Provide specific guidance
        failed_checks = [name for name, result in results if not result]
        
        if "Dependencies" in failed_checks:
            print("   📦 Install missing packages: pip install -r requirements.txt")
        
        if "Database" in failed_checks:
            print("   🗄️ Set up database: python scripts/setup_database.py")
        
        if "Directories" in failed_checks:
            print("   📁 Directories were created automatically")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
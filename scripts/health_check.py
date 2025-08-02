"""
Health check script for data ingestion pipeline
Verifies all system components are working correctly
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import config
from utils.helpers import ensure_directory_exists
from ingestion.api_ingestion import APIIngestion

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthChecker:
    """System health checker for data ingestion pipeline"""
    
    def __init__(self):
        """Initialize health checker"""
        self.checks = []
        self.passed_checks = 0
        self.failed_checks = 0
    
    def add_check_result(self, check_name: str, passed: bool, message: str = ""):
        """Add a check result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.checks.append({
            'name': check_name,
            'passed': passed,
            'message': message,
            'status': status
        })
        
        if passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1
        
        print(f"{status} {check_name}")
        if message:
            print(f"    {message}")
    
    def check_python_version(self) -> bool:
        """Check Python version"""
        try:
            version = sys.version_info
            if version.major >= 3 and version.minor >= 8:
                self.add_check_result(
                    "Python Version", 
                    True, 
                    f"Python {version.major}.{version.minor}.{version.micro}"
                )
                return True
            else:
                self.add_check_result(
                    "Python Version", 
                    False, 
                    f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+"
                )
                return False
        except Exception as e:
            self.add_check_result("Python Version", False, f"Error: {e}")
            return False
    
    def check_required_packages(self) -> bool:
        """Check if required packages are installed"""
        required_packages = [
            'pandas', 'requests', 'schedule', 'pyyaml', 'sqlite3'
        ]
        
        all_packages_ok = True
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'sqlite3':
                    import sqlite3
                else:
                    __import__(package)
                logger.debug(f"Package {package} is available")
            except ImportError:
                missing_packages.append(package)
                all_packages_ok = False
        
        if all_packages_ok:
            self.add_check_result(
                "Required Packages", 
                True, 
                f"All {len(required_packages)} packages available"
            )
        else:
            self.add_check_result(
                "Required Packages", 
                False, 
                f"Missing packages: {', '.join(missing_packages)}"
            )
        
        return all_packages_ok
    
    def check_directory_structure(self) -> bool:
        """Check if required directories exist"""
        required_directories = [
            config.file.input_dir,
            f"{config.file.input_dir}/csv",
            f"{config.file.input_dir}/json",
            config.file.output_dir,
            config.file.processed_dir,
            "logs",
            "data/samples"
        ]
        
        all_dirs_ok = True
        missing_dirs = []
        
        for directory in required_directories:
            if not os.path.exists(directory):
                try:
                    ensure_directory_exists(directory)
                    logger.debug(f"Created directory: {directory}")
                except Exception as e:
                    missing_dirs.append(f"{directory} ({e})")
                    all_dirs_ok = False
            else:
                logger.debug(f"Directory exists: {directory}")
        
        if all_dirs_ok:
            self.add_check_result(
                "Directory Structure", 
                True, 
                f"All {len(required_directories)} directories available"
            )
        else:
            self.add_check_result(
                "Directory Structure", 
                False, 
                f"Issues with directories: {', '.join(missing_dirs)}"
            )
        
        return all_dirs_ok
    
    def check_database_connection(self) -> bool:
        """Check database connection and schema"""
        try:
            db_path = config.database.path
            
            # Check if database file exists
            if not os.path.exists(db_path):
                self.add_check_result(
                    "Database Connection", 
                    False, 
                    f"Database file not found: {db_path}"
                )
                return False
            
            # Try to connect and query
            conn = sqlite3.connect(db_path, timeout=config.database.timeout)
            cursor = conn.cursor()
            
            # Check if main tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['orders', 'customers', 'products', 'pipeline_runs']
            missing_tables = [table for table in required_tables if table not in tables]
            
            if missing_tables:
                conn.close()
                self.add_check_result(
                    "Database Connection", 
                    False, 
                    f"Missing tables: {', '.join(missing_tables)}"
                )
                return False
            
            # Test a simple query
            cursor.execute("SELECT COUNT(*) FROM orders")
            order_count = cursor.fetchone()[0]
            
            conn.close()
            
            self.add_check_result(
                "Database Connection", 
                True, 
                f"Connected successfully, {order_count} orders in database"
            )
            return True
            
        except Exception as e:
            self.add_check_result("Database Connection", False, f"Error: {e}")
            return False
    
    def check_api_connectivity(self) -> bool:
        """Check API connectivity"""
        try:
            api_ingestion = APIIngestion()
            
            # Test API connection
            connection_test = api_ingestion.test_api_connection()
            
            if connection_test['success']:
                self.add_check_result(
                    "API Connectivity", 
                    True, 
                    f"Connected to {connection_test['base_url']} ({connection_test['response_time']:.2f}s)"
                )
                api_ingestion.close()
                return True
            else:
                self.add_check_result(
                    "API Connectivity", 
                    False, 
                    f"Connection failed: {connection_test['error_message']}"
                )
                api_ingestion.close()
                return False
                
        except Exception as e:
            self.add_check_result("API Connectivity", False, f"Error: {e}")
            return False
    
    def check_sample_data(self) -> bool:
        """Check if sample data is available"""
        try:
            sample_files_found = 0
            
            # Check for CSV sample files
            csv_dir = f"{config.file.input_dir}/csv"
            if os.path.exists(csv_dir):
                csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
                sample_files_found += len(csv_files)
            
            # Check for JSON sample files
            json_dir = f"{config.file.input_dir}/json"
            if os.path.exists(json_dir):
                json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
                sample_files_found += len(json_files)
            
            if sample_files_found > 0:
                self.add_check_result(
                    "Sample Data", 
                    True, 
                    f"Found {sample_files_found} sample files"
                )
                return True
            else:
                self.add_check_result(
                    "Sample Data", 
                    False, 
                    "No sample files found. Run: python scripts/generate_sample_data.py"
                )
                return False
                
        except Exception as e:
            self.add_check_result("Sample Data", False, f"Error: {e}")
            return False
    
    def check_configuration(self) -> bool:
        """Check configuration validity"""
        try:
            # Validate configuration
            validation = config.validate_config()
            
            if validation['is_valid']:
                self.add_check_result(
                    "Configuration", 
                    True, 
                    "All configuration settings are valid"
                )
                return True
            else:
                self.add_check_result(
                    "Configuration", 
                    False, 
                    f"Configuration issues: {', '.join(validation['issues'])}"
                )
                return False
                
        except Exception as e:
            self.add_check_result("Configuration", False, f"Error: {e}")
            return False
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            import shutil
            
            # Check disk space for data directory
            data_dir = "data"
            if os.path.exists(data_dir):
                total, used, free = shutil.disk_usage(data_dir)
                
                # Convert to MB
                free_mb = free // (1024 * 1024)
                total_mb = total // (1024 * 1024)
                
                # Check if we have at least 100MB free
                if free_mb >= 100:
                    self.add_check_result(
                        "Disk Space", 
                        True, 
                        f"{free_mb:,} MB free of {total_mb:,} MB total"
                    )
                    return True
                else:
                    self.add_check_result(
                        "Disk Space", 
                        False, 
                        f"Low disk space: {free_mb:,} MB free (need at least 100 MB)"
                    )
                    return False
            else:
                self.add_check_result("Disk Space", False, "Data directory not found")
                return False
                
        except Exception as e:
            self.add_check_result("Disk Space", False, f"Error: {e}")
            return False
    
    def check_permissions(self) -> bool:
        """Check file system permissions"""
        try:
            test_dirs = [
                config.file.input_dir,
                config.file.output_dir,
                "logs"
            ]
            
            permission_issues = []
            
            for test_dir in test_dirs:
                # Test write permission
                test_file = os.path.join(test_dir, f"test_write_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tmp")
                
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    logger.debug(f"Write permission OK for {test_dir}")
                except Exception as e:
                    permission_issues.append(f"{test_dir}: {e}")
            
            if not permission_issues:
                self.add_check_result(
                    "File Permissions", 
                    True, 
                    f"Write permissions OK for {len(test_dirs)} directories"
                )
                return True
            else:
                self.add_check_result(
                    "File Permissions", 
                    False, 
                    f"Permission issues: {', '.join(permission_issues)}"
                )
                return False
                
        except Exception as e:
            self.add_check_result("File Permissions", False, f"Error: {e}")
            return False
    
    def run_all_checks(self) -> bool:
        """Run all health checks"""
        print("🔍 Running system health checks...\n")
        
        # Run all checks
        checks_to_run = [
            self.check_python_version,
            self.check_required_packages,
            self.check_directory_structure,
            self.check_configuration,
            self.check_database_connection,
            self.check_api_connectivity,
            self.check_sample_data,
            self.check_disk_space,
            self.check_permissions
        ]
        
        for check_func in checks_to_run:
            try:
                check_func()
            except Exception as e:
                logger.error(f"Error running check {check_func.__name__}: {e}")
                self.add_check_result(check_func.__name__, False, f"Check error: {e}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("HEALTH CHECK SUMMARY")
        print(f"{'='*60}")
        print(f"Total checks: {len(self.checks)}")
        print(f"Passed: {self.passed_checks}")
        print(f"Failed: {self.failed_checks}")
        
        if self.failed_checks == 0:
            print("🎉 All systems are healthy!")
            print("💡 You can now run the data ingestion pipeline!")
        else:
            print(f"⚠️  {self.failed_checks} issues found. Please fix them before running the pipeline.")
            print("\nFailed checks:")
            for check in self.checks:
                if not check['passed']:
                    print(f"  ❌ {check['name']}: {check['message']}")
        
        print(f"{'='*60}")
        
        return self.failed_checks == 0
    
    def generate_health_report(self) -> str:
        """Generate detailed health report"""
        report = []
        report.append("# System Health Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## Summary")
        report.append(f"- Total checks: {len(self.checks)}")
        report.append(f"- Passed: {self.passed_checks}")
        report.append(f"- Failed: {self.failed_checks}")
        report.append(f"- Success rate: {(self.passed_checks / len(self.checks) * 100):.1f}%")
        report.append("")
        
        report.append("## Detailed Results")
        for check in self.checks:
            status_icon = "✅" if check['passed'] else "❌"
            report.append(f"### {status_icon} {check['name']}")
            if check['message']:
                report.append(f"**Details:** {check['message']}")
            report.append("")
        
        if self.failed_checks > 0:
            report.append("## Recommendations")
            for check in self.checks:
                if not check['passed']:
                    if "Missing packages" in check['message']:
                        report.append("- Install missing packages: `pip install -r requirements.txt`")
                    elif "Database" in check['name']:
                        report.append("- Initialize database: `python scripts/setup_database.py`")
                    elif "Sample Data" in check['name']:
                        report.append("- Generate sample data: `python scripts/generate_sample_data.py`")
                    elif "API" in check['name']:
                        report.append("- Check internet connection and API endpoint")
        
        return "\n".join(report)

def main():
    """Main function to run health checks"""
    print("🚀 Starting system health check...")
    
    try:
        checker = HealthChecker()
        all_healthy = checker.run_all_checks()
        
        # Generate and save report
        report = checker.generate_health_report()
        report_file = "logs/health_check_report.md"
        
        ensure_directory_exists(os.path.dirname(report_file))
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return 0 if all_healthy else 1
        
    except Exception as e:
        logger.error(f"Health check failed with error: {e}")
        print(f"❌ Health check failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
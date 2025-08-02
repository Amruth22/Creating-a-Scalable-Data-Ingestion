#!/usr/bin/env python3
"""
Test runner script for data ingestion pipeline tests
Provides comprehensive test execution with various options and reporting
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def run_command(command, description=""):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print()
    
    try:
        result = subprocess.run(command, capture_output=False, text=True, check=True)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n💥 {description} failed with error: {e}")
        return False

def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests"""
    command = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        command.append("-v")
    
    if coverage:
        command.extend([
            "--cov=src",
            "--cov-report=html:tests/coverage_html",
            "--cov-report=term-missing"
        ])
    
    return run_command(command, "Running Unit Tests")

def run_integration_tests(verbose=False):
    """Run integration tests"""
    command = ["python", "-m", "pytest", "tests/integration/"]
    
    if verbose:
        command.append("-v")
    
    return run_command(command, "Running Integration Tests")

def run_all_tests(verbose=False, coverage=False):
    """Run all tests"""
    command = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        command.append("-v")
    
    if coverage:
        command.extend([
            "--cov=src",
            "--cov-report=html:tests/coverage_html",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])
    
    return run_command(command, "Running All Tests")

def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function"""
    command = ["python", "-m", "pytest", test_path]
    
    if verbose:
        command.append("-v")
    
    return run_command(command, f"Running Specific Test: {test_path}")

def run_tests_by_marker(marker, verbose=False):
    """Run tests by marker"""
    command = ["python", "-m", "pytest", "-m", marker]
    
    if verbose:
        command.append("-v")
    
    return run_command(command, f"Running Tests with Marker: {marker}")

def run_performance_tests(verbose=False):
    """Run performance tests"""
    command = ["python", "-m", "pytest", "-m", "performance"]
    
    if verbose:
        command.append("-v")
    
    return run_command(command, "Running Performance Tests")

def run_slow_tests(verbose=False):
    """Run slow tests"""
    command = ["python", "-m", "pytest", "-m", "slow"]
    
    if verbose:
        command.append("-v")
    
    return run_command(command, "Running Slow Tests")

def generate_test_report():
    """Generate comprehensive test report"""
    print(f"\n{'='*60}")
    print("📊 Generating Test Report")
    print(f"{'='*60}")
    
    # Run tests with JUnit XML output
    command = [
        "python", "-m", "pytest", "tests/",
        "--junitxml=tests/test_report.xml",
        "--html=tests/test_report.html",
        "--self-contained-html",
        "--cov=src",
        "--cov-report=html:tests/coverage_html",
        "--cov-report=xml:tests/coverage.xml"
    ]
    
    success = run_command(command, "Generating Test Report")
    
    if success:
        print("\n📄 Test reports generated:")
        print("  - HTML Report: tests/test_report.html")
        print("  - XML Report: tests/test_report.xml")
        print("  - Coverage HTML: tests/coverage_html/index.html")
        print("  - Coverage XML: tests/coverage.xml")
    
    return success

def check_test_environment():
    """Check test environment and dependencies"""
    print(f"\n{'='*60}")
    print("🔍 Checking Test Environment")
    print(f"{'='*60}")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    else:
        print("✅ Python version is compatible")
    
    # Check required packages
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-html",
        "pandas",
        "numpy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check test directories
    test_dirs = ["tests/unit", "tests/integration", "tests/test_data"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            print(f"✅ {test_dir} directory exists")
        else:
            print(f"❌ {test_dir} directory missing")
            return False
    
    print("\n✅ Test environment is ready")
    return True

def clean_test_artifacts():
    """Clean test artifacts and temporary files"""
    print(f"\n{'='*60}")
    print("🧹 Cleaning Test Artifacts")
    print(f"{'='*60}")
    
    # Directories and files to clean
    cleanup_paths = [
        "tests/coverage_html",
        "tests/test_report.html",
        "tests/test_report.xml",
        "tests/coverage.xml",
        "tests/.pytest_cache",
        "tests/__pycache__",
        ".coverage",
        "*.pyc",
        "*/__pycache__"
    ]
    
    cleaned_count = 0
    for path_pattern in cleanup_paths:
        if "*" in path_pattern:
            # Handle glob patterns
            import glob
            for path in glob.glob(path_pattern, recursive=True):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        print(f"🗑️ Removed file: {path}")
                        cleaned_count += 1
                    elif os.path.isdir(path):
                        import shutil
                        shutil.rmtree(path)
                        print(f"🗑️ Removed directory: {path}")
                        cleaned_count += 1
                except Exception as e:
                    print(f"⚠️ Could not remove {path}: {e}")
        else:
            path = Path(path_pattern)
            try:
                if path.is_file():
                    path.unlink()
                    print(f"🗑️ Removed file: {path}")
                    cleaned_count += 1
                elif path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                    print(f"🗑️ Removed directory: {path}")
                    cleaned_count += 1
            except Exception as e:
                print(f"⚠️ Could not remove {path}: {e}")
    
    print(f"\n✅ Cleaned {cleaned_count} artifacts")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test runner for data ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py --all                    # Run all tests
  python tests/run_tests.py --unit                   # Run unit tests only
  python tests/run_tests.py --integration            # Run integration tests only
  python tests/run_tests.py --marker unit            # Run tests with 'unit' marker
  python tests/run_tests.py --specific tests/unit/test_file_ingestion.py
  python tests/run_tests.py --performance            # Run performance tests
  python tests/run_tests.py --report                 # Generate test report
  python tests/run_tests.py --check                  # Check test environment
  python tests/run_tests.py --clean                  # Clean test artifacts
        """
    )
    
    # Test execution options
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--specific', type=str, help='Run specific test file or function')
    parser.add_argument('--marker', type=str, help='Run tests with specific marker')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--slow', action='store_true', help='Run slow tests')
    
    # Options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', '-c', action='store_true', help='Generate coverage report')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive test report')
    
    # Utility options
    parser.add_argument('--check', action='store_true', help='Check test environment')
    parser.add_argument('--clean', action='store_true', help='Clean test artifacts')
    
    args = parser.parse_args()
    
    # Print banner
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           🧪 DATA PIPELINE TEST RUNNER 🧪                   ║
    ║                                                              ║
    ║              Comprehensive Testing Suite                     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Test execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = True
    
    # Handle utility commands first
    if args.check:
        success = check_test_environment()
        if not success:
            return 1
    
    if args.clean:
        clean_test_artifacts()
        return 0
    
    # Handle test execution
    if args.all:
        success = run_all_tests(args.verbose, args.coverage)
    elif args.unit:
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.integration:
        success = run_integration_tests(args.verbose)
    elif args.specific:
        success = run_specific_test(args.specific, args.verbose)
    elif args.marker:
        success = run_tests_by_marker(args.marker, args.verbose)
    elif args.performance:
        success = run_performance_tests(args.verbose)
    elif args.slow:
        success = run_slow_tests(args.verbose)
    elif args.report:
        success = generate_test_report()
    else:
        # Default: run all tests
        success = run_all_tests(args.verbose, args.coverage)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📋 TEST EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\n🎉 All tests passed successfully!")
        print("\nNext steps:")
        print("  - Review test coverage report")
        print("  - Check for any warnings in test output")
        print("  - Consider adding more tests for edge cases")
    else:
        print("\n💥 Some tests failed!")
        print("\nTroubleshooting:")
        print("  - Check the test output for specific failures")
        print("  - Review the error messages and stack traces")
        print("  - Fix failing tests and run again")
        print("  - Use --verbose flag for more detailed output")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
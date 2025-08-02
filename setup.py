#!/usr/bin/env python3
"""
Setup script for Creating a Scalable Data Ingestion Pipeline
Handles project installation, dependencies, and development environment setup
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages, Command
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess
import shutil

# Ensure Python 3.8+
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required. You are using Python {}.{}.{}".format(
        sys.version_info.major, sys.version_info.minor, sys.version_info.micro))

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Read project metadata
def read_file(filename):
    """Read file content"""
    filepath = PROJECT_ROOT / filename
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def get_version():
    """Get version from src/__init__.py or default"""
    version_file = PROJECT_ROOT / "src" / "__init__.py"
    if version_file.exists():
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    return "1.0.0"

def get_long_description():
    """Get long description from README"""
    readme_content = read_file("README.md")
    if readme_content:
        return readme_content
    return "A scalable data ingestion pipeline for processing e-commerce order data from multiple sources."

# Project metadata
NAME = "data-ingestion-pipeline"
VERSION = get_version()
DESCRIPTION = "A scalable data ingestion pipeline for e-commerce order processing"
LONG_DESCRIPTION = get_long_description()
AUTHOR = "Data Engineering Team"
AUTHOR_EMAIL = "team@company.com"
URL = "https://github.com/Amruth22/Creating-a-Scalable-Data-Ingestion"
LICENSE = "MIT"

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

# Core dependencies (production)
INSTALL_REQUIRES = [
    # Core data processing
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    
    # HTTP requests and APIs
    "requests>=2.31.0",
    "urllib3>=2.0.0",
    
    # Job scheduling
    "schedule>=1.2.0",
    "APScheduler>=3.10.0",
    
    # Database connectivity
    "sqlalchemy>=2.0.0",
    
    # Configuration management
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0",
    
    # Logging and monitoring
    "structlog>=23.0.0",
    
    # Data validation
    "jsonschema>=4.20.0",
    "cerberus>=1.3.0",
    
    # File processing
    "openpyxl>=3.1.0",  # Excel files
    "xlrd>=2.0.0",      # Excel files (older format)
    
    # Date and time utilities
    "python-dateutil>=2.8.0",
    
    # Progress bars
    "tqdm>=4.66.0",
    
    # CLI utilities
    "click>=8.1.0",
]

# Development dependencies
DEV_REQUIRES = [
    # Testing
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "pytest-html>=3.2.0",
    "pytest-timeout>=2.1.0",
    "pytest-xdist>=3.3.0",  # Parallel testing
    
    # Code quality
    "black>=23.0.0",
    "flake8>=6.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "pylint>=2.17.0",
    
    # Documentation
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
    "myst-parser>=2.0.0",
    
    # Development tools
    "pre-commit>=3.3.0",
    "bump2version>=1.0.0",
    "twine>=4.0.0",
    "wheel>=0.41.0",
]

# Database-specific dependencies
DATABASE_REQUIRES = [
    "mysql-connector-python>=8.1.0",  # MySQL
    "psycopg2-binary>=2.9.0",         # PostgreSQL
    "pymongo>=4.5.0",                 # MongoDB
    "redis>=4.6.0",                   # Redis
]

# Jupyter notebook dependencies
NOTEBOOK_REQUIRES = [
    "jupyter>=1.0.0",
    "notebook>=7.0.0",
    "ipywidgets>=8.1.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
]

# Performance optimization dependencies
PERFORMANCE_REQUIRES = [
    "numba>=0.57.0",
    "dask>=2023.8.0",
    "pyarrow>=13.0.0",
    "fastparquet>=2023.8.0",
]

# All extra dependencies
EXTRAS_REQUIRE = {
    "dev": DEV_REQUIRES,
    "database": DATABASE_REQUIRES,
    "notebooks": NOTEBOOK_REQUIRES,
    "performance": PERFORMANCE_REQUIRES,
    "all": DEV_REQUIRES + DATABASE_REQUIRES + NOTEBOOK_REQUIRES + PERFORMANCE_REQUIRES,
}

# Entry points for command-line scripts
ENTRY_POINTS = {
    "console_scripts": [
        "data-pipeline=scripts.run_pipeline:main",
        "pipeline-setup=scripts.setup_database:main",
        "pipeline-health=scripts.health_check:main",
        "generate-sample-data=scripts.generate_sample_data:main",
        "run-tests=tests.run_tests:main",
    ],
}

# Project classifiers
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Information Technology",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Database",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: System :: Monitoring",
    "Topic :: Utilities",
]

# Keywords
KEYWORDS = [
    "data-ingestion", "data-pipeline", "etl", "data-processing",
    "data-validation", "data-quality", "automation", "scheduling",
    "monitoring", "e-commerce", "orders", "analytics"
]

class PostInstallCommand(install):
    """Post-installation command to set up the environment"""
    
    def run(self):
        install.run(self)
        self.setup_environment()
    
    def setup_environment(self):
        """Set up the project environment after installation"""
        print("\n" + "="*60)
        print("🚀 Setting up Data Ingestion Pipeline environment...")
        print("="*60)
        
        try:
            # Create necessary directories
            self.create_directories()
            
            # Initialize database
            self.initialize_database()
            
            # Generate sample data
            self.generate_sample_data()
            
            # Set up configuration files
            self.setup_configuration()
            
            print("\n✅ Environment setup completed successfully!")
            print("\nNext steps:")
            print("  1. Review configuration files in config/")
            print("  2. Run: data-pipeline --help")
            print("  3. Check health: pipeline-health")
            print("  4. Run tests: run-tests --all")
            
        except Exception as e:
            print(f"\n⚠️ Environment setup encountered issues: {e}")
            print("You can manually run setup commands later.")
    
    def create_directories(self):
        """Create necessary project directories"""
        directories = [
            "data",
            "data/input",
            "data/input/csv",
            "data/input/json",
            "data/input/processed",
            "data/output",
            "data/output/reports",
            "data/output/exports",
            "data/archive",
            "data/backups",
            "logs",
            "config",
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created directory: {directory}")
    
    def initialize_database(self):
        """Initialize the database"""
        try:
            print("🗄️ Initializing database...")
            # Import and run database setup
            from scripts.setup_database import main as setup_db
            setup_db()
            print("✅ Database initialized successfully")
        except Exception as e:
            print(f"⚠️ Database initialization failed: {e}")
    
    def generate_sample_data(self):
        """Generate sample data for testing"""
        try:
            print("📊 Generating sample data...")
            from scripts.generate_sample_data import main as generate_data
            generate_data()
            print("✅ Sample data generated successfully")
        except Exception as e:
            print(f"⚠️ Sample data generation failed: {e}")
    
    def setup_configuration(self):
        """Set up configuration files"""
        try:
            print("⚙️ Setting up configuration files...")
            
            # Copy default configuration files if they don't exist
            config_files = [
                "config/pipeline_config.yaml",
                "config/database_config.yaml",
                "config/api_config.yaml",
                "config/validation_config.yaml",
                "config/monitoring_config.yaml",
            ]
            
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    print(f"✅ Configuration file exists: {config_file}")
                else:
                    print(f"⚠️ Configuration file missing: {config_file}")
            
            print("✅ Configuration setup completed")
        except Exception as e:
            print(f"⚠️ Configuration setup failed: {e}")

class PostDevelopCommand(develop):
    """Post-development installation command"""
    
    def run(self):
        develop.run(self)
        self.setup_development_environment()
    
    def setup_development_environment(self):
        """Set up development environment"""
        print("\n" + "="*60)
        print("🛠️ Setting up development environment...")
        print("="*60)
        
        try:
            # Install pre-commit hooks
            self.install_pre_commit_hooks()
            
            # Set up testing environment
            self.setup_testing_environment()
            
            print("\n✅ Development environment setup completed!")
            print("\nDevelopment commands:")
            print("  - Run tests: run-tests --all")
            print("  - Format code: black src/ tests/")
            print("  - Lint code: flake8 src/ tests/")
            print("  - Type check: mypy src/")
            
        except Exception as e:
            print(f"\n⚠️ Development setup encountered issues: {e}")
    
    def install_pre_commit_hooks(self):
        """Install pre-commit hooks"""
        try:
            print("🔗 Installing pre-commit hooks...")
            subprocess.run(["pre-commit", "install"], check=True, capture_output=True)
            print("✅ Pre-commit hooks installed")
        except subprocess.CalledProcessError:
            print("⚠️ Pre-commit hooks installation failed")
        except FileNotFoundError:
            print("⚠️ Pre-commit not found, skipping hooks installation")
    
    def setup_testing_environment(self):
        """Set up testing environment"""
        try:
            print("🧪 Setting up testing environment...")
            
            # Create test directories
            test_dirs = ["tests/test_output", "tests/coverage_html"]
            for test_dir in test_dirs:
                Path(test_dir).mkdir(parents=True, exist_ok=True)
            
            print("✅ Testing environment ready")
        except Exception as e:
            print(f"⚠️ Testing environment setup failed: {e}")

class CleanCommand(Command):
    """Custom command to clean build artifacts"""
    
    description = "Clean build artifacts and temporary files"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        """Clean build artifacts"""
        print("🧹 Cleaning build artifacts...")
        
        # Directories to clean
        clean_dirs = [
            "build",
            "dist",
            "*.egg-info",
            "__pycache__",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            ".mypy_cache",
            ".tox",
        ]
        
        # File patterns to clean
        clean_patterns = [
            "**/*.pyc",
            "**/*.pyo",
            "**/__pycache__",
            "**/.DS_Store",
        ]
        
        import glob
        
        cleaned_count = 0
        
        # Clean directories
        for pattern in clean_dirs:
            for path in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"🗑️ Removed directory: {path}")
                        cleaned_count += 1
                    elif os.path.isfile(path):
                        os.remove(path)
                        print(f"🗑️ Removed file: {path}")
                        cleaned_count += 1
                except Exception as e:
                    print(f"⚠️ Could not remove {path}: {e}")
        
        # Clean file patterns
        for pattern in clean_patterns:
            for path in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        print(f"🗑️ Removed file: {path}")
                        cleaned_count += 1
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"🗑️ Removed directory: {path}")
                        cleaned_count += 1
                except Exception as e:
                    print(f"⚠️ Could not remove {path}: {e}")
        
        print(f"\n✅ Cleaned {cleaned_count} artifacts")

class TestCommand(Command):
    """Custom command to run tests"""
    
    description = "Run the test suite"
    user_options = [
        ("coverage", "c", "Generate coverage report"),
        ("verbose", "v", "Verbose output"),
        ("unit", "u", "Run unit tests only"),
        ("integration", "i", "Run integration tests only"),
    ]
    
    def initialize_options(self):
        self.coverage = False
        self.verbose = False
        self.unit = False
        self.integration = False
    
    def finalize_options(self):
        pass
    
    def run(self):
        """Run tests"""
        print("🧪 Running test suite...")
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        if self.unit:
            cmd.append("tests/unit/")
        elif self.integration:
            cmd.append("tests/integration/")
        else:
            cmd.append("tests/")
        
        if self.verbose:
            cmd.append("-v")
        
        if self.coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=html",
                "--cov-report=term-missing"
            ])
        
        # Run tests
        try:
            subprocess.run(cmd, check=True)
            print("✅ Tests completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Tests failed with exit code {e.returncode}")
            sys.exit(e.returncode)

# Main setup configuration
def main():
    """Main setup function"""
    setup(
        # Basic project information
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        
        # Package discovery
        packages=find_packages(exclude=["tests*", "docs*"]),
        package_dir={"": "."},
        include_package_data=True,
        
        # Dependencies
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        
        # Entry points
        entry_points=ENTRY_POINTS,
        
        # Package data
        package_data={
            "": [
                "*.yaml", "*.yml", "*.json", "*.txt", "*.md",
                "config/*.yaml", "config/*.yml",
                "tests/test_data/*",
            ]
        },
        
        # Metadata
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        project_urls={
            "Bug Reports": f"{URL}/issues",
            "Source": URL,
            "Documentation": f"{URL}#readme",
        },
        
        # Custom commands
        cmdclass={
            "install": PostInstallCommand,
            "develop": PostDevelopCommand,
            "clean": CleanCommand,
            "test": TestCommand,
        },
        
        # Additional options
        zip_safe=False,
        platforms=["any"],
    )

if __name__ == "__main__":
    main()
"""
Configuration management for data ingestion pipeline
Handles settings, environment variables, and configuration files
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    path: str = "data/orders.db"
    timeout: int = 30
    check_same_thread: bool = False

@dataclass
class APIConfig:
    """API configuration settings"""
    base_url: str = "https://jsonplaceholder.typicode.com"
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 5

@dataclass
class FileConfig:
    """File processing configuration"""
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    processed_dir: str = "data/input/processed"
    archive_dir: str = "data/archive"
    max_file_size_mb: int = 100

@dataclass
class PipelineConfig:
    """Pipeline execution configuration"""
    batch_size: int = 1000
    max_workers: int = 4
    enable_parallel_processing: bool = True
    enable_monitoring: bool = True
    log_level: str = "INFO"

@dataclass
class AlertConfig:
    """Alerting configuration"""
    enable_email_alerts: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_from: str = ""
    email_to: str = ""
    email_password: str = ""

class ConfigManager:
    """Configuration manager for the data ingestion pipeline"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file (str, optional): Path to configuration file
        """
        self.config_file = config_file or "config/pipeline_config.yaml"
        self.config = self._load_config()
        
        # Initialize configuration objects
        self.database = DatabaseConfig(**self.config.get('database', {}))
        self.api = APIConfig(**self.config.get('api', {}))
        self.file = FileConfig(**self.config.get('file', {}))
        self.pipeline = PipelineConfig(**self.config.get('pipeline', {}))
        self.alerts = AlertConfig(**self.config.get('alerts', {}))
        
        # Override with environment variables
        self._load_environment_variables()
        
        # Create necessary directories
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config or {}
            else:
                logger.warning(f"Configuration file {self.config_file} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'database': {
                'path': 'data/orders.db',
                'timeout': 30,
                'check_same_thread': False
            },
            'api': {
                'base_url': 'https://jsonplaceholder.typicode.com',
                'timeout': 30,
                'retry_attempts': 3,
                'retry_delay': 5
            },
            'file': {
                'input_dir': 'data/input',
                'output_dir': 'data/output',
                'processed_dir': 'data/input/processed',
                'archive_dir': 'data/archive',
                'max_file_size_mb': 100
            },
            'pipeline': {
                'batch_size': 1000,
                'max_workers': 4,
                'enable_parallel_processing': True,
                'enable_monitoring': True,
                'log_level': 'INFO'
            },
            'alerts': {
                'enable_email_alerts': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'email_from': '',
                'email_to': '',
                'email_password': ''
            }
        }
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        # Database configuration
        if os.getenv('DB_PATH'):
            self.database.path = os.getenv('DB_PATH')
        
        # API configuration
        if os.getenv('API_BASE_URL'):
            self.api.base_url = os.getenv('API_BASE_URL')
        if os.getenv('API_TIMEOUT'):
            self.api.timeout = int(os.getenv('API_TIMEOUT'))
        
        # File configuration
        if os.getenv('INPUT_DIR'):
            self.file.input_dir = os.getenv('INPUT_DIR')
        if os.getenv('OUTPUT_DIR'):
            self.file.output_dir = os.getenv('OUTPUT_DIR')
        
        # Pipeline configuration
        if os.getenv('BATCH_SIZE'):
            self.pipeline.batch_size = int(os.getenv('BATCH_SIZE'))
        if os.getenv('MAX_WORKERS'):
            self.pipeline.max_workers = int(os.getenv('MAX_WORKERS'))
        if os.getenv('LOG_LEVEL'):
            self.pipeline.log_level = os.getenv('LOG_LEVEL')
        
        # Alert configuration
        if os.getenv('ENABLE_EMAIL_ALERTS'):
            self.alerts.enable_email_alerts = os.getenv('ENABLE_EMAIL_ALERTS').lower() == 'true'
        if os.getenv('SMTP_SERVER'):
            self.alerts.smtp_server = os.getenv('SMTP_SERVER')
        if os.getenv('EMAIL_FROM'):
            self.alerts.email_from = os.getenv('EMAIL_FROM')
        if os.getenv('EMAIL_TO'):
            self.alerts.email_to = os.getenv('EMAIL_TO')
        if os.getenv('EMAIL_PASSWORD'):
            self.alerts.email_password = os.getenv('EMAIL_PASSWORD')
        
        logger.info("Environment variables loaded")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.file.input_dir,
            f"{self.file.input_dir}/csv",
            f"{self.file.input_dir}/json",
            self.file.output_dir,
            self.file.processed_dir,
            self.file.archive_dir,
            "logs",
            "data/samples"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Created necessary directories")
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return f"sqlite:///{self.database.path}"
    
    def get_api_endpoint(self, endpoint: str) -> str:
        """Get full API endpoint URL"""
        return f"{self.api.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    def get_input_path(self, filename: str) -> str:
        """Get full path for input file"""
        return os.path.join(self.file.input_dir, filename)
    
    def get_output_path(self, filename: str) -> str:
        """Get full path for output file"""
        return os.path.join(self.file.output_dir, filename)
    
    def get_processed_path(self, filename: str) -> str:
        """Get full path for processed file"""
        return os.path.join(self.file.processed_dir, filename)
    
    def save_config(self, output_file: Optional[str] = None):
        """Save current configuration to file"""
        output_file = output_file or self.config_file
        
        config_dict = {
            'database': {
                'path': self.database.path,
                'timeout': self.database.timeout,
                'check_same_thread': self.database.check_same_thread
            },
            'api': {
                'base_url': self.api.base_url,
                'timeout': self.api.timeout,
                'retry_attempts': self.api.retry_attempts,
                'retry_delay': self.api.retry_delay
            },
            'file': {
                'input_dir': self.file.input_dir,
                'output_dir': self.file.output_dir,
                'processed_dir': self.file.processed_dir,
                'archive_dir': self.file.archive_dir,
                'max_file_size_mb': self.file.max_file_size_mb
            },
            'pipeline': {
                'batch_size': self.pipeline.batch_size,
                'max_workers': self.pipeline.max_workers,
                'enable_parallel_processing': self.pipeline.enable_parallel_processing,
                'enable_monitoring': self.pipeline.enable_monitoring,
                'log_level': self.pipeline.log_level
            },
            'alerts': {
                'enable_email_alerts': self.alerts.enable_email_alerts,
                'smtp_server': self.alerts.smtp_server,
                'smtp_port': self.alerts.smtp_port,
                'email_from': self.alerts.email_from,
                'email_to': self.alerts.email_to
                # Don't save password to file
            }
        }
        
        # Create directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {output_file}")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration settings"""
        issues = []
        
        # Validate database path
        db_dir = Path(self.database.path).parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create database directory: {e}")
        
        # Validate directories
        for dir_path in [self.file.input_dir, self.file.output_dir]:
            if not os.path.exists(dir_path):
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {dir_path}: {e}")
        
        # Validate numeric values
        if self.pipeline.batch_size <= 0:
            issues.append("Batch size must be positive")
        
        if self.pipeline.max_workers <= 0:
            issues.append("Max workers must be positive")
        
        if self.api.timeout <= 0:
            issues.append("API timeout must be positive")
        
        # Validate email configuration if enabled
        if self.alerts.enable_email_alerts:
            if not self.alerts.email_from:
                issues.append("Email from address is required when alerts are enabled")
            if not self.alerts.email_to:
                issues.append("Email to address is required when alerts are enabled")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 50)
        print("CURRENT CONFIGURATION")
        print("=" * 50)
        
        print(f"\n📊 DATABASE:")
        print(f"  Path: {self.database.path}")
        print(f"  Timeout: {self.database.timeout}s")
        
        print(f"\n🌐 API:")
        print(f"  Base URL: {self.api.base_url}")
        print(f"  Timeout: {self.api.timeout}s")
        print(f"  Retry Attempts: {self.api.retry_attempts}")
        
        print(f"\n📁 FILES:")
        print(f"  Input Directory: {self.file.input_dir}")
        print(f"  Output Directory: {self.file.output_dir}")
        print(f"  Processed Directory: {self.file.processed_dir}")
        print(f"  Max File Size: {self.file.max_file_size_mb}MB")
        
        print(f"\n🔄 PIPELINE:")
        print(f"  Batch Size: {self.pipeline.batch_size}")
        print(f"  Max Workers: {self.pipeline.max_workers}")
        print(f"  Parallel Processing: {self.pipeline.enable_parallel_processing}")
        print(f"  Monitoring: {self.pipeline.enable_monitoring}")
        print(f"  Log Level: {self.pipeline.log_level}")
        
        print(f"\n🚨 ALERTS:")
        print(f"  Email Alerts: {self.alerts.enable_email_alerts}")
        if self.alerts.enable_email_alerts:
            print(f"  SMTP Server: {self.alerts.smtp_server}:{self.alerts.smtp_port}")
            print(f"  From: {self.alerts.email_from}")
            print(f"  To: {self.alerts.email_to}")
        
        print("=" * 50)

# Global configuration instance
config = ConfigManager()

if __name__ == "__main__":
    # Test configuration
    config_manager = ConfigManager()
    
    # Print current configuration
    config_manager.print_config()
    
    # Validate configuration
    validation = config_manager.validate_config()
    if validation['is_valid']:
        print("\n✅ Configuration is valid")
    else:
        print(f"\n❌ Configuration issues: {validation['issues']}")
    
    # Save configuration
    config_manager.save_config("config/pipeline_config.yaml")
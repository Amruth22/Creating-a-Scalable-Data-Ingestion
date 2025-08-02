"""
Constants and enumerations for data ingestion pipeline
Centralized location for all constants used across the project
"""

from enum import Enum
from typing import Dict, List

# File Extensions
class FileExtension:
    CSV = '.csv'
    JSON = '.json'
    XLSX = '.xlsx'
    XLS = '.xls'
    TXT = '.txt'
    LOG = '.log'

# Data Source Types
class DataSourceType(Enum):
    FILE_CSV = "file_csv"
    FILE_JSON = "file_json"
    API_REST = "api_rest"
    DATABASE = "database"
    STREAM = "stream"

# Pipeline Stages
class PipelineStage(Enum):
    INGESTION = "ingestion"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    STORAGE = "storage"
    MONITORING = "monitoring"

# Processing Status
class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

# Data Quality Levels
class DataQualityLevel(Enum):
    EXCELLENT = "excellent"  # 95-100%
    GOOD = "good"           # 85-94%
    FAIR = "fair"           # 70-84%
    POOR = "poor"           # 50-69%
    CRITICAL = "critical"   # <50%

# Alert Types
class AlertType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# Log Levels
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Database Tables
class DatabaseTable:
    ORDERS = "orders"
    CUSTOMERS = "customers"
    PRODUCTS = "products"
    PIPELINE_RUNS = "pipeline_runs"
    DATA_QUALITY_METRICS = "data_quality_metrics"
    PROCESSING_LOGS = "processing_logs"

# API Endpoints
class APIEndpoint:
    ORDERS = "/posts"  # Using JSONPlaceholder for demo
    USERS = "/users"
    COMMENTS = "/comments"

# File Patterns
class FilePattern:
    CSV_ORDERS = "orders_*.csv"
    JSON_ORDERS = "orders_*.json"
    PROCESSED_PREFIX = "processed_"
    ARCHIVE_PREFIX = "archive_"
    ERROR_PREFIX = "error_"

# Configuration Keys
class ConfigKey:
    DATABASE_PATH = "database.path"
    API_BASE_URL = "api.base_url"
    INPUT_DIRECTORY = "file.input_dir"
    OUTPUT_DIRECTORY = "file.output_dir"
    BATCH_SIZE = "pipeline.batch_size"
    MAX_WORKERS = "pipeline.max_workers"
    LOG_LEVEL = "pipeline.log_level"

# Default Values
class DefaultValue:
    BATCH_SIZE = 1000
    MAX_WORKERS = 4
    API_TIMEOUT = 30
    DATABASE_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    MAX_FILE_SIZE_MB = 100
    LOG_RETENTION_DAYS = 30
    ARCHIVE_RETENTION_DAYS = 90

# Error Messages
class ErrorMessage:
    FILE_NOT_FOUND = "File not found: {file_path}"
    INVALID_FILE_FORMAT = "Invalid file format: {file_path}"
    DATABASE_CONNECTION_FAILED = "Failed to connect to database: {error}"
    API_REQUEST_FAILED = "API request failed: {url} - {error}"
    VALIDATION_FAILED = "Data validation failed: {errors}"
    TRANSFORMATION_FAILED = "Data transformation failed: {error}"
    INSUFFICIENT_DISK_SPACE = "Insufficient disk space: {required} MB required"
    PERMISSION_DENIED = "Permission denied: {path}"
    TIMEOUT_ERROR = "Operation timed out after {timeout} seconds"

# Success Messages
class SuccessMessage:
    FILE_PROCESSED = "Successfully processed file: {file_path}"
    DATA_VALIDATED = "Data validation completed: {records} records validated"
    DATA_TRANSFORMED = "Data transformation completed: {records} records transformed"
    DATA_STORED = "Data stored successfully: {records} records saved"
    PIPELINE_COMPLETED = "Pipeline completed successfully in {duration}"
    BACKUP_CREATED = "Backup created: {backup_path}"

# Validation Rules
class ValidationRule:
    REQUIRED_FIELDS = [
        'order_id',
        'customer_name',
        'product',
        'quantity',
        'price',
        'order_date'
    ]
    
    OPTIONAL_FIELDS = [
        'source',
        'store_location',
        'customer_email',
        'product_category',
        'discount',
        'notes'
    ]
    
    NUMERIC_FIELDS = [
        'quantity',
        'price',
        'discount'
    ]
    
    DATE_FIELDS = [
        'order_date',
        'delivery_date',
        'created_at',
        'updated_at'
    ]
    
    EMAIL_FIELDS = [
        'customer_email',
        'contact_email'
    ]

# Data Types
class DataType:
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    URL = "url"

# File Size Limits (in MB)
class FileSizeLimit:
    CSV_MAX = 50
    JSON_MAX = 25
    EXCEL_MAX = 100
    LOG_MAX = 10

# Performance Thresholds
class PerformanceThreshold:
    MAX_PROCESSING_TIME_SECONDS = 300  # 5 minutes
    MAX_MEMORY_USAGE_PERCENT = 80
    MIN_SUCCESS_RATE_PERCENT = 95
    MAX_ERROR_RATE_PERCENT = 5
    MAX_API_RESPONSE_TIME_SECONDS = 10

# Monitoring Metrics
class MetricName:
    RECORDS_PROCESSED = "records_processed"
    PROCESSING_TIME = "processing_time_seconds"
    ERROR_COUNT = "error_count"
    SUCCESS_RATE = "success_rate_percent"
    THROUGHPUT = "throughput_per_minute"
    MEMORY_USAGE = "memory_usage_mb"
    DISK_USAGE = "disk_usage_mb"
    API_RESPONSE_TIME = "api_response_time_seconds"
    DATA_QUALITY_SCORE = "data_quality_score"

# Email Templates
class EmailTemplate:
    ALERT_SUBJECT = "🚨 Data Pipeline Alert: {alert_type}"
    SUCCESS_SUBJECT = "✅ Data Pipeline Success: {pipeline_name}"
    ERROR_SUBJECT = "❌ Data Pipeline Error: {pipeline_name}"
    
    ALERT_BODY = """
    Alert Type: {alert_type}
    Pipeline: {pipeline_name}
    Timestamp: {timestamp}
    Message: {message}
    
    Details:
    {details}
    """

# Regular Expressions
class RegexPattern:
    EMAIL = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    PHONE = r'^\+?1?-?\.?\s?\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})$'
    ORDER_ID = r'^[A-Z]{3}-\d{4}-\d{3}$'  # Format: ORD-2024-001
    DATE_ISO = r'^\d{4}-\d{2}-\d{2}$'
    DATETIME_ISO = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$'

# HTTP Status Codes
class HTTPStatus:
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503

# Sample Data
class SampleData:
    ORDERS = [
        {
            "order_id": "ORD-2024-001",
            "customer_name": "John Doe",
            "product": "iPhone 15",
            "quantity": 1,
            "price": 999.99,
            "order_date": "2024-01-15",
            "source": "website"
        },
        {
            "order_id": "ORD-2024-002",
            "customer_name": "Jane Smith",
            "product": "MacBook Pro",
            "quantity": 1,
            "price": 1999.99,
            "order_date": "2024-01-15",
            "source": "store"
        },
        {
            "order_id": "ORD-2024-003",
            "customer_name": "Bob Wilson",
            "product": "AirPods Pro",
            "quantity": 2,
            "price": 249.99,
            "order_date": "2024-01-16",
            "source": "mobile_app"
        }
    ]
    
    CUSTOMERS = [
        {
            "customer_id": "CUST-001",
            "name": "John Doe",
            "email": "john.doe@email.com",
            "phone": "+1-555-0123",
            "address": "123 Main St, New York, NY"
        },
        {
            "customer_id": "CUST-002",
            "name": "Jane Smith",
            "email": "jane.smith@email.com",
            "phone": "+1-555-0124",
            "address": "456 Oak Ave, Los Angeles, CA"
        }
    ]
    
    PRODUCTS = [
        {
            "product_id": "PROD-001",
            "name": "iPhone 15",
            "category": "Electronics",
            "price": 999.99,
            "in_stock": True
        },
        {
            "product_id": "PROD-002",
            "name": "MacBook Pro",
            "category": "Electronics",
            "price": 1999.99,
            "in_stock": True
        }
    ]

# Quality Score Ranges
class QualityScore:
    EXCELLENT_MIN = 95
    GOOD_MIN = 85
    FAIR_MIN = 70
    POOR_MIN = 50
    # Below 50 is CRITICAL

# Time Formats
class TimeFormat:
    ISO_DATE = "%Y-%m-%d"
    ISO_DATETIME = "%Y-%m-%dT%H:%M:%S"
    DISPLAY_DATETIME = "%Y-%m-%d %H:%M:%S"
    FILENAME_TIMESTAMP = "%Y%m%d_%H%M%S"
    LOG_TIMESTAMP = "%Y-%m-%d %H:%M:%S.%f"

# Environment Variables
class EnvVar:
    DB_PATH = "DB_PATH"
    API_BASE_URL = "API_BASE_URL"
    INPUT_DIR = "INPUT_DIR"
    OUTPUT_DIR = "OUTPUT_DIR"
    LOG_LEVEL = "LOG_LEVEL"
    BATCH_SIZE = "BATCH_SIZE"
    MAX_WORKERS = "MAX_WORKERS"
    EMAIL_FROM = "EMAIL_FROM"
    EMAIL_TO = "EMAIL_TO"
    EMAIL_PASSWORD = "EMAIL_PASSWORD"
    SMTP_SERVER = "SMTP_SERVER"

# Directory Names
class Directory:
    INPUT = "input"
    OUTPUT = "output"
    PROCESSED = "processed"
    ARCHIVE = "archive"
    LOGS = "logs"
    CONFIG = "config"
    DATA = "data"
    SAMPLES = "samples"
    REPORTS = "reports"
    BACKUP = "backup"

if __name__ == "__main__":
    # Test constants
    print("Testing constants...")
    
    print(f"File extensions: {FileExtension.CSV}, {FileExtension.JSON}")
    print(f"Data source types: {[e.value for e in DataSourceType]}")
    print(f"Pipeline stages: {[e.value for e in PipelineStage]}")
    print(f"Required fields: {ValidationRule.REQUIRED_FIELDS}")
    print(f"Sample order: {SampleData.ORDERS[0]}")
    
    print("Constants test completed!")
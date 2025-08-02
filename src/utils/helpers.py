"""
Helper utility functions for data ingestion pipeline
Common functions used across different modules
"""

import os
import json
import csv
import hashlib
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

def get_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: MD5 hash of the file
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""

def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return 0.0

def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path (str): Path to directory
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False

def move_file(source_path: str, destination_path: str) -> bool:
    """
    Move file from source to destination
    
    Args:
        source_path (str): Source file path
        destination_path (str): Destination file path
        
    Returns:
        bool: True if file was moved successfully
    """
    try:
        # Ensure destination directory exists
        ensure_directory_exists(os.path.dirname(destination_path))
        
        # Move the file
        shutil.move(source_path, destination_path)
        logger.info(f"Moved file from {source_path} to {destination_path}")
        return True
    except Exception as e:
        logger.error(f"Error moving file from {source_path} to {destination_path}: {e}")
        return False

def copy_file(source_path: str, destination_path: str) -> bool:
    """
    Copy file from source to destination
    
    Args:
        source_path (str): Source file path
        destination_path (str): Destination file path
        
    Returns:
        bool: True if file was copied successfully
    """
    try:
        # Ensure destination directory exists
        ensure_directory_exists(os.path.dirname(destination_path))
        
        # Copy the file
        shutil.copy2(source_path, destination_path)
        logger.info(f"Copied file from {source_path} to {destination_path}")
        return True
    except Exception as e:
        logger.error(f"Error copying file from {source_path} to {destination_path}: {e}")
        return False

def delete_file(file_path: str) -> bool:
    """
    Delete a file
    
    Args:
        file_path (str): Path to file to delete
        
    Returns:
        bool: True if file was deleted successfully
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
            return True
        else:
            logger.warning(f"File not found for deletion: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False

def get_files_in_directory(directory_path: str, extension: Optional[str] = None) -> List[str]:
    """
    Get list of files in directory
    
    Args:
        directory_path (str): Path to directory
        extension (str, optional): File extension filter (e.g., '.csv')
        
    Returns:
        List[str]: List of file paths
    """
    try:
        if not os.path.exists(directory_path):
            logger.warning(f"Directory not found: {directory_path}")
            return []
        
        files = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                if extension is None or file.lower().endswith(extension.lower()):
                    files.append(file_path)
        
        return sorted(files)
    except Exception as e:
        logger.error(f"Error listing files in {directory_path}: {e}")
        return []

def read_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Read JSON file and return data
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        Dict[str, Any]: JSON data or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully read JSON file: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return None

def write_json_file(data: Dict[str, Any], file_path: str) -> bool:
    """
    Write data to JSON file
    
    Args:
        data (Dict[str, Any]): Data to write
        file_path (str): Path to output file
        
    Returns:
        bool: True if successful
    """
    try:
        # Ensure directory exists
        ensure_directory_exists(os.path.dirname(file_path))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Successfully wrote JSON file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        return False

def read_csv_file(file_path: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Read CSV file and return DataFrame
    
    Args:
        file_path (str): Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        pd.DataFrame: CSV data or None if error
    """
    try:
        data = pd.read_csv(file_path, **kwargs)
        logger.info(f"Successfully read CSV file: {file_path} ({len(data)} rows)")
        return data
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        return None

def write_csv_file(data: pd.DataFrame, file_path: str, **kwargs) -> bool:
    """
    Write DataFrame to CSV file
    
    Args:
        data (pd.DataFrame): Data to write
        file_path (str): Path to output file
        **kwargs: Additional arguments for DataFrame.to_csv
        
    Returns:
        bool: True if successful
    """
    try:
        # Ensure directory exists
        ensure_directory_exists(os.path.dirname(file_path))
        
        # Set default parameters
        csv_kwargs = {'index': False}
        csv_kwargs.update(kwargs)
        
        data.to_csv(file_path, **csv_kwargs)
        logger.info(f"Successfully wrote CSV file: {file_path} ({len(data)} rows)")
        return True
    except Exception as e:
        logger.error(f"Error writing CSV file {file_path}: {e}")
        return False

def format_timestamp(timestamp: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp as string
    
    Args:
        timestamp (datetime, optional): Timestamp to format (default: now)
        format_str (str): Format string
        
    Returns:
        str: Formatted timestamp
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime(format_str)

def parse_timestamp(timestamp_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """
    Parse timestamp string to datetime
    
    Args:
        timestamp_str (str): Timestamp string
        format_str (str): Format string
        
    Returns:
        datetime: Parsed timestamp or None if error
    """
    try:
        return datetime.strptime(timestamp_str, format_str)
    except Exception as e:
        logger.error(f"Error parsing timestamp '{timestamp_str}': {e}")
        return None

def get_age_in_days(timestamp: datetime) -> int:
    """
    Get age of timestamp in days
    
    Args:
        timestamp (datetime): Timestamp to check
        
    Returns:
        int: Age in days
    """
    return (datetime.now() - timestamp).days

def is_file_older_than(file_path: str, days: int) -> bool:
    """
    Check if file is older than specified days
    
    Args:
        file_path (str): Path to file
        days (int): Number of days
        
    Returns:
        bool: True if file is older than specified days
    """
    try:
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        return get_age_in_days(file_time) > days
    except Exception as e:
        logger.error(f"Error checking file age for {file_path}: {e}")
        return False

def cleanup_old_files(directory_path: str, days: int, extension: Optional[str] = None) -> int:
    """
    Clean up files older than specified days
    
    Args:
        directory_path (str): Directory to clean
        days (int): Files older than this will be deleted
        extension (str, optional): File extension filter
        
    Returns:
        int: Number of files deleted
    """
    deleted_count = 0
    try:
        files = get_files_in_directory(directory_path, extension)
        for file_path in files:
            if is_file_older_than(file_path, days):
                if delete_file(file_path):
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old files from {directory_path}")
        return deleted_count
    except Exception as e:
        logger.error(f"Error cleaning up old files in {directory_path}: {e}")
        return 0

def generate_unique_filename(base_name: str, extension: str, directory: str = "") -> str:
    """
    Generate unique filename by adding timestamp
    
    Args:
        base_name (str): Base filename
        extension (str): File extension (with or without dot)
        directory (str): Directory path
        
    Returns:
        str: Unique filename
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}{extension}"
    
    if directory:
        return os.path.join(directory, filename)
    return filename

def validate_email(email: str) -> bool:
    """
    Simple email validation
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if email format is valid
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted size (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def format_duration(seconds: float) -> str:
    """
    Format duration in human readable format
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"

def retry_operation(func, max_attempts: int = 3, delay: float = 1.0, *args, **kwargs):
    """
    Retry an operation with exponential backoff
    
    Args:
        func: Function to retry
        max_attempts (int): Maximum number of attempts
        delay (float): Initial delay between attempts
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        Result of function or raises last exception
    """
    import time
    
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_attempts} attempts failed. Last error: {e}")
    
    raise last_exception

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles division by zero
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float): Default value if division by zero
        
    Returns:
        float: Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default

def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to maximum length
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        suffix (str): Suffix to add if truncated
        
    Returns:
        str: Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage
    
    Returns:
        Dict[str, float]: Memory usage information
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        logger.warning("psutil not available, cannot get memory usage")
        return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}

if __name__ == "__main__":
    # Test helper functions
    print("Testing helper functions...")
    
    # Test file operations
    test_dir = "test_helpers"
    ensure_directory_exists(test_dir)
    
    # Test JSON operations
    test_data = {"test": "data", "timestamp": datetime.now()}
    json_file = os.path.join(test_dir, "test.json")
    write_json_file(test_data, json_file)
    read_data = read_json_file(json_file)
    print(f"JSON test: {read_data}")
    
    # Test timestamp formatting
    now = datetime.now()
    formatted = format_timestamp(now)
    print(f"Formatted timestamp: {formatted}")
    
    # Test file size formatting
    print(f"File size: {format_file_size(1536000)}")
    
    # Test duration formatting
    print(f"Duration: {format_duration(150.5)}")
    
    # Cleanup
    shutil.rmtree(test_dir)
    print("Helper functions test completed!")
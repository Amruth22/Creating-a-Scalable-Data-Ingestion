"""
File ingestion module for processing CSV and JSON files
Monitors directories and processes new files automatically
"""

import os
import json
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import shutil

from ..utils.config import config
from ..utils.helpers import (
    get_files_in_directory, move_file, get_file_hash, 
    get_file_size_mb, ensure_directory_exists
)
from ..utils.constants import FileExtension, ProcessingStatus, DataSourceType

# Configure logging
logger = logging.getLogger(__name__)

class FileIngestion:
    """File ingestion class for processing CSV and JSON files"""
    
    def __init__(self):
        """Initialize file ingestion"""
        self.input_dir = config.file.input_dir
        self.processed_dir = config.file.processed_dir
        self.max_file_size_mb = config.file.max_file_size_mb
        
        # Ensure directories exist
        ensure_directory_exists(f"{self.input_dir}/csv")
        ensure_directory_exists(f"{self.input_dir}/json")
        ensure_directory_exists(self.processed_dir)
        
        # Track processed files to avoid reprocessing
        self.processed_files = set()
        
        logger.info("File ingestion initialized")
    
    def discover_new_files(self) -> Dict[str, List[str]]:
        """
        Discover new files in input directories
        
        Returns:
            Dict[str, List[str]]: Dictionary with file types and their paths
        """
        try:
            new_files = {
                'csv': [],
                'json': []
            }
            
            # Find CSV files
            csv_dir = f"{self.input_dir}/csv"
            if os.path.exists(csv_dir):
                csv_files = get_files_in_directory(csv_dir, FileExtension.CSV)
                new_files['csv'] = [f for f in csv_files if f not in self.processed_files]
            
            # Find JSON files
            json_dir = f"{self.input_dir}/json"
            if os.path.exists(json_dir):
                json_files = get_files_in_directory(json_dir, FileExtension.JSON)
                new_files['json'] = [f for f in json_files if f not in self.processed_files]
            
            total_new_files = len(new_files['csv']) + len(new_files['json'])
            if total_new_files > 0:
                logger.info(f"Discovered {total_new_files} new files: {len(new_files['csv'])} CSV, {len(new_files['json'])} JSON")
            
            return new_files
            
        except Exception as e:
            logger.error(f"Error discovering new files: {e}")
            return {'csv': [], 'json': []}
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate file before processing
        
        Args:
            file_path (str): Path to file
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'file_info': {}
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                validation_result['is_valid'] = False
                validation_result['issues'].append("File does not exist")
                return validation_result
            
            # Get file information
            file_size_mb = get_file_size_mb(file_path)
            file_hash = get_file_hash(file_path)
            
            validation_result['file_info'] = {
                'size_mb': file_size_mb,
                'hash': file_hash,
                'modified_time': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }
            
            # Check file size
            if file_size_mb > self.max_file_size_mb:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"File too large: {file_size_mb:.2f}MB > {self.max_file_size_mb}MB")
            
            # Check if file is empty
            if file_size_mb == 0:
                validation_result['is_valid'] = False
                validation_result['issues'].append("File is empty")
            
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in [FileExtension.CSV, FileExtension.JSON]:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Unsupported file type: {file_ext}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def process_csv_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single CSV file
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            Dict[str, Any]: Processing results
        """
        result = {
            'success': False,
            'file_path': file_path,
            'records_count': 0,
            'data': None,
            'error_message': None,
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing CSV file: {file_path}")
            
            # Validate file first
            validation = self.validate_file(file_path)
            if not validation['is_valid']:
                result['error_message'] = f"Validation failed: {', '.join(validation['issues'])}"
                return result
            
            # Read CSV file
            try:
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252']
                data = None
                
                for encoding in encodings:
                    try:
                        data = pd.read_csv(file_path, encoding=encoding)
                        logger.info(f"Successfully read CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if data is None:
                    result['error_message'] = "Could not read CSV file with any supported encoding"
                    return result
                
            except Exception as e:
                result['error_message'] = f"Error reading CSV file: {str(e)}"
                return result
            
            # Basic data validation
            if data.empty:
                result['error_message'] = "CSV file is empty"
                return result
            
            # Add metadata columns
            data['source_file'] = os.path.basename(file_path)
            data['source_type'] = DataSourceType.FILE_CSV.value
            data['ingested_at'] = datetime.now().isoformat()
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result.update({
                'success': True,
                'records_count': len(data),
                'data': data,
                'processing_time': processing_time,
                'file_info': validation['file_info']
            })
            
            logger.info(f"Successfully processed CSV file: {file_path} ({len(data)} records in {processing_time:.2f}s)")
            
        except Exception as e:
            result['error_message'] = f"Unexpected error processing CSV file: {str(e)}"
            logger.error(f"Error processing CSV file {file_path}: {e}")
        
        return result
    
    def process_json_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            Dict[str, Any]: Processing results
        """
        result = {
            'success': False,
            'file_path': file_path,
            'records_count': 0,
            'data': None,
            'error_message': None,
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing JSON file: {file_path}")
            
            # Validate file first
            validation = self.validate_file(file_path)
            if not validation['is_valid']:
                result['error_message'] = f"Validation failed: {', '.join(validation['issues'])}"
                return result
            
            # Read JSON file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except json.JSONDecodeError as e:
                result['error_message'] = f"Invalid JSON format: {str(e)}"
                return result
            except Exception as e:
                result['error_message'] = f"Error reading JSON file: {str(e)}"
                return result
            
            # Extract orders from JSON structure
            orders_data = []
            
            if isinstance(json_data, dict):
                # Check for common JSON structures
                if 'orders' in json_data:
                    orders_data = json_data['orders']
                elif 'data' in json_data:
                    orders_data = json_data['data']
                elif 'items' in json_data:
                    orders_data = json_data['items']
                else:
                    # Assume the entire JSON is a single record
                    orders_data = [json_data]
            elif isinstance(json_data, list):
                orders_data = json_data
            else:
                result['error_message'] = "Unsupported JSON structure"
                return result
            
            if not orders_data:
                result['error_message'] = "No order data found in JSON file"
                return result
            
            # Convert to DataFrame
            try:
                data = pd.DataFrame(orders_data)
            except Exception as e:
                result['error_message'] = f"Error converting JSON to DataFrame: {str(e)}"
                return result
            
            # Add metadata columns
            data['source_file'] = os.path.basename(file_path)
            data['source_type'] = DataSourceType.FILE_JSON.value
            data['ingested_at'] = datetime.now().isoformat()
            
            # Add JSON metadata if available
            if isinstance(json_data, dict):
                if 'app_version' in json_data:
                    data['app_version'] = json_data['app_version']
                if 'upload_time' in json_data:
                    data['upload_time'] = json_data['upload_time']
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result.update({
                'success': True,
                'records_count': len(data),
                'data': data,
                'processing_time': processing_time,
                'file_info': validation['file_info']
            })
            
            logger.info(f"Successfully processed JSON file: {file_path} ({len(data)} records in {processing_time:.2f}s)")
            
        except Exception as e:
            result['error_message'] = f"Unexpected error processing JSON file: {str(e)}"
            logger.error(f"Error processing JSON file {file_path}: {e}")
        
        return result
    
    def mark_file_as_processed(self, file_path: str, success: bool = True) -> bool:
        """
        Mark file as processed by moving it to processed directory
        
        Args:
            file_path (str): Path to processed file
            success (bool): Whether processing was successful
            
        Returns:
            bool: True if file was moved successfully
        """
        try:
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if success:
                new_filename = f"processed_{timestamp}_{filename}"
                destination = os.path.join(self.processed_dir, new_filename)
            else:
                new_filename = f"error_{timestamp}_{filename}"
                error_dir = os.path.join(self.processed_dir, "errors")
                ensure_directory_exists(error_dir)
                destination = os.path.join(error_dir, new_filename)
            
            # Move the file
            if move_file(file_path, destination):
                self.processed_files.add(file_path)
                logger.info(f"Marked file as processed: {file_path} -> {destination}")
                return True
            else:
                logger.error(f"Failed to move processed file: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error marking file as processed {file_path}: {e}")
            return False
    
    def process_all_files(self) -> Dict[str, Any]:
        """
        Process all new files in input directories
        
        Returns:
            Dict[str, Any]: Processing summary
        """
        summary = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_records': 0,
            'processing_time': 0,
            'results': [],
            'errors': []
        }
        
        start_time = datetime.now()
        
        try:
            # Discover new files
            new_files = self.discover_new_files()
            total_files = len(new_files['csv']) + len(new_files['json'])
            
            if total_files == 0:
                logger.info("No new files to process")
                return summary
            
            summary['total_files'] = total_files
            logger.info(f"Starting to process {total_files} files")
            
            # Process CSV files
            for csv_file in new_files['csv']:
                result = self.process_csv_file(csv_file)
                summary['results'].append(result)
                
                if result['success']:
                    summary['successful_files'] += 1
                    summary['total_records'] += result['records_count']
                    self.mark_file_as_processed(csv_file, True)
                else:
                    summary['failed_files'] += 1
                    summary['errors'].append({
                        'file': csv_file,
                        'error': result['error_message']
                    })
                    self.mark_file_as_processed(csv_file, False)
            
            # Process JSON files
            for json_file in new_files['json']:
                result = self.process_json_file(json_file)
                summary['results'].append(result)
                
                if result['success']:
                    summary['successful_files'] += 1
                    summary['total_records'] += result['records_count']
                    self.mark_file_as_processed(json_file, True)
                else:
                    summary['failed_files'] += 1
                    summary['errors'].append({
                        'file': json_file,
                        'error': result['error_message']
                    })
                    self.mark_file_as_processed(json_file, False)
            
            # Calculate total processing time
            summary['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"File processing completed: {summary['successful_files']}/{summary['total_files']} files successful, {summary['total_records']} total records")
            
            if summary['errors']:
                logger.warning(f"Encountered {len(summary['errors'])} errors during processing")
                for error in summary['errors']:
                    logger.warning(f"  {error['file']}: {error['error']}")
            
        except Exception as e:
            logger.error(f"Error in process_all_files: {e}")
            summary['errors'].append({
                'file': 'process_all_files',
                'error': str(e)
            })
        
        return summary
    
    def get_combined_data(self) -> Optional[pd.DataFrame]:
        """
        Get combined data from all successfully processed files
        
        Returns:
            pd.DataFrame: Combined data or None if no data
        """
        try:
            # Process all files and collect successful results
            processing_summary = self.process_all_files()
            
            successful_results = [
                result for result in processing_summary['results'] 
                if result['success'] and result['data'] is not None
            ]
            
            if not successful_results:
                logger.warning("No successful file processing results found")
                return None
            
            # Combine all DataFrames
            all_dataframes = [result['data'] for result in successful_results]
            combined_data = pd.concat(all_dataframes, ignore_index=True)
            
            logger.info(f"Combined data from {len(successful_results)} files: {len(combined_data)} total records")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error combining data: {e}")
            return None

if __name__ == "__main__":
    # Test file ingestion
    ingestion = FileIngestion()
    
    # Process all files
    summary = ingestion.process_all_files()
    
    print("File Ingestion Test Results:")
    print(f"Total files: {summary['total_files']}")
    print(f"Successful: {summary['successful_files']}")
    print(f"Failed: {summary['failed_files']}")
    print(f"Total records: {summary['total_records']}")
    print(f"Processing time: {summary['processing_time']:.2f}s")
    
    if summary['errors']:
        print("\nErrors:")
        for error in summary['errors']:
            print(f"  {error['file']}: {error['error']}")
    
    # Get combined data
    combined_data = ingestion.get_combined_data()
    if combined_data is not None:
        print(f"\nCombined data shape: {combined_data.shape}")
        print(f"Columns: {list(combined_data.columns)}")
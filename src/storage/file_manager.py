"""
File management module for data storage operations
Handles file operations, archiving, and file-based storage management
"""

import os
import shutil
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from ..utils.config import config
from ..utils.helpers import (
    ensure_directory_exists, get_file_hash, get_file_size_mb,
    move_file, copy_file, delete_file, format_file_size,
    generate_unique_filename, cleanup_old_files
)

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FileOperationResult:
    """File operation result container"""
    success: bool
    operation: str
    files_affected: int
    execution_time: float
    file_paths: List[str] = None
    error_message: Optional[str] = None

class FileManager:
    """File manager for data storage and file operations"""
    
    def __init__(self):
        """Initialize file manager"""
        self.input_dir = config.file.input_dir
        self.output_dir = config.file.output_dir
        self.processed_dir = config.file.processed_dir
        self.archive_dir = config.file.archive_dir
        
        # Ensure all directories exist
        self._initialize_directories()
        
        logger.info("File manager initialized")
    
    def _initialize_directories(self):
        """Initialize required directories"""
        directories = [
            self.input_dir,
            f"{self.input_dir}/csv",
            f"{self.input_dir}/json",
            self.output_dir,
            f"{self.output_dir}/reports",
            f"{self.output_dir}/exports",
            self.processed_dir,
            f"{self.processed_dir}/errors",
            self.archive_dir,
            "logs",
            "data/backup"
        ]
        
        for directory in directories:
            ensure_directory_exists(directory)
    
    def save_dataframe_to_csv(self, data: pd.DataFrame, filename: str, 
                             directory: Optional[str] = None) -> FileOperationResult:
        """
        Save DataFrame to CSV file
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Output filename
            directory (str, optional): Output directory (default: output_dir)
            
        Returns:
            FileOperationResult: Save operation result
        """
        start_time = datetime.now()
        
        try:
            # Determine output directory
            output_dir = directory or self.output_dir
            ensure_directory_exists(output_dir)
            
            # Generate full file path
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            file_path = os.path.join(output_dir, filename)
            
            # Save DataFrame to CSV
            data.to_csv(file_path, index=False, encoding='utf-8')
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = FileOperationResult(
                success=True,
                operation='save_dataframe_to_csv',
                files_affected=1,
                execution_time=execution_time,
                file_paths=[file_path]
            )
            
            file_size = get_file_size_mb(file_path)
            logger.info(f"Saved CSV file: {file_path} ({len(data)} records, {file_size:.2f} MB)")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error saving CSV file: {e}")
            return FileOperationResult(
                success=False,
                operation='save_dataframe_to_csv',
                files_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def save_dataframe_to_json(self, data: pd.DataFrame, filename: str,
                              directory: Optional[str] = None,
                              orient: str = 'records') -> FileOperationResult:
        """
        Save DataFrame to JSON file
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Output filename
            directory (str, optional): Output directory
            orient (str): JSON orientation
            
        Returns:
            FileOperationResult: Save operation result
        """
        start_time = datetime.now()
        
        try:
            # Determine output directory
            output_dir = directory or self.output_dir
            ensure_directory_exists(output_dir)
            
            # Generate full file path
            if not filename.endswith('.json'):
                filename += '.json'
            
            file_path = os.path.join(output_dir, filename)
            
            # Save DataFrame to JSON
            data.to_json(file_path, orient=orient, date_format='iso', indent=2)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = FileOperationResult(
                success=True,
                operation='save_dataframe_to_json',
                files_affected=1,
                execution_time=execution_time,
                file_paths=[file_path]
            )
            
            file_size = get_file_size_mb(file_path)
            logger.info(f"Saved JSON file: {file_path} ({len(data)} records, {file_size:.2f} MB)")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error saving JSON file: {e}")
            return FileOperationResult(
                success=False,
                operation='save_dataframe_to_json',
                files_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def save_report(self, content: str, filename: str, 
                   format_type: str = 'txt') -> FileOperationResult:
        """
        Save report content to file
        
        Args:
            content (str): Report content
            filename (str): Output filename
            format_type (str): File format (txt, md, html)
            
        Returns:
            FileOperationResult: Save operation result
        """
        start_time = datetime.now()
        
        try:
            # Ensure reports directory exists
            reports_dir = os.path.join(self.output_dir, 'reports')
            ensure_directory_exists(reports_dir)
            
            # Add extension if not present
            if not filename.endswith(f'.{format_type}'):
                filename += f'.{format_type}'
            
            file_path = os.path.join(reports_dir, filename)
            
            # Save content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = FileOperationResult(
                success=True,
                operation='save_report',
                files_affected=1,
                execution_time=execution_time,
                file_paths=[file_path]
            )
            
            file_size = get_file_size_mb(file_path)
            logger.info(f"Saved report: {file_path} ({file_size:.2f} MB)")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error saving report: {e}")
            return FileOperationResult(
                success=False,
                operation='save_report',
                files_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def archive_files(self, file_paths: List[str], 
                     archive_subdir: Optional[str] = None) -> FileOperationResult:
        """
        Archive files to archive directory
        
        Args:
            file_paths (List[str]): List of file paths to archive
            archive_subdir (str, optional): Subdirectory in archive
            
        Returns:
            FileOperationResult: Archive operation result
        """
        start_time = datetime.now()
        
        try:
            # Determine archive directory
            if archive_subdir:
                archive_dir = os.path.join(self.archive_dir, archive_subdir)
            else:
                archive_dir = self.archive_dir
            
            ensure_directory_exists(archive_dir)
            
            archived_files = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found for archiving: {file_path}")
                    continue
                
                # Generate archive filename
                original_name = os.path.basename(file_path)
                name, ext = os.path.splitext(original_name)
                archive_filename = f"{name}_{timestamp}{ext}"
                archive_path = os.path.join(archive_dir, archive_filename)
                
                # Move file to archive
                if move_file(file_path, archive_path):
                    archived_files.append(archive_path)
                else:
                    logger.error(f"Failed to archive file: {file_path}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = FileOperationResult(
                success=len(archived_files) > 0,
                operation='archive_files',
                files_affected=len(archived_files),
                execution_time=execution_time,
                file_paths=archived_files
            )
            
            logger.info(f"Archived {len(archived_files)} files to {archive_dir}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error archiving files: {e}")
            return FileOperationResult(
                success=False,
                operation='archive_files',
                files_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def backup_files(self, file_paths: List[str], 
                    backup_subdir: Optional[str] = None) -> FileOperationResult:
        """
        Create backup copies of files
        
        Args:
            file_paths (List[str]): List of file paths to backup
            backup_subdir (str, optional): Subdirectory in backup
            
        Returns:
            FileOperationResult: Backup operation result
        """
        start_time = datetime.now()
        
        try:
            # Determine backup directory
            backup_dir = "data/backup"
            if backup_subdir:
                backup_dir = os.path.join(backup_dir, backup_subdir)
            
            ensure_directory_exists(backup_dir)
            
            backed_up_files = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found for backup: {file_path}")
                    continue
                
                # Generate backup filename
                original_name = os.path.basename(file_path)
                name, ext = os.path.splitext(original_name)
                backup_filename = f"{name}_backup_{timestamp}{ext}"
                backup_path = os.path.join(backup_dir, backup_filename)
                
                # Copy file to backup
                if copy_file(file_path, backup_path):
                    backed_up_files.append(backup_path)
                else:
                    logger.error(f"Failed to backup file: {file_path}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = FileOperationResult(
                success=len(backed_up_files) > 0,
                operation='backup_files',
                files_affected=len(backed_up_files),
                execution_time=execution_time,
                file_paths=backed_up_files
            )
            
            logger.info(f"Backed up {len(backed_up_files)} files to {backup_dir}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error backing up files: {e}")
            return FileOperationResult(
                success=False,
                operation='backup_files',
                files_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def cleanup_old_files(self, directory: str, days_old: int = 30,
                         file_pattern: Optional[str] = None) -> FileOperationResult:
        """
        Clean up old files from directory
        
        Args:
            directory (str): Directory to clean
            days_old (int): Files older than this will be deleted
            file_pattern (str, optional): File pattern to match
            
        Returns:
            FileOperationResult: Cleanup operation result
        """
        start_time = datetime.now()
        
        try:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found for cleanup: {directory}")
                return FileOperationResult(
                    success=False,
                    operation='cleanup_old_files',
                    files_affected=0,
                    execution_time=0,
                    error_message=f"Directory not found: {directory}"
                )
            
            deleted_files = []
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    # Check file pattern if specified
                    if file_pattern and not file.endswith(file_pattern):
                        continue
                    
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Check file modification time
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if file_mtime < cutoff_date:
                            if delete_file(file_path):
                                deleted_files.append(file_path)
                    except Exception as e:
                        logger.warning(f"Error checking file {file_path}: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = FileOperationResult(
                success=True,
                operation='cleanup_old_files',
                files_affected=len(deleted_files),
                execution_time=execution_time,
                file_paths=deleted_files
            )
            
            logger.info(f"Cleaned up {len(deleted_files)} old files from {directory}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error cleaning up old files: {e}")
            return FileOperationResult(
                success=False,
                operation='cleanup_old_files',
                files_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def get_directory_info(self, directory: str) -> Dict[str, Any]:
        """
        Get information about directory contents
        
        Args:
            directory (str): Directory to analyze
            
        Returns:
            Dict: Directory information
        """
        try:
            if not os.path.exists(directory):
                return {
                    'exists': False,
                    'error': f"Directory not found: {directory}"
                }
            
            info = {
                'exists': True,
                'path': directory,
                'total_files': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'file_types': {},
                'oldest_file': None,
                'newest_file': None,
                'largest_file': None
            }
            
            oldest_time = None
            newest_time = None
            largest_size = 0
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    try:
                        # File stats
                        stat = os.stat(file_path)
                        file_size = stat.st_size
                        file_mtime = datetime.fromtimestamp(stat.st_mtime)
                        
                        info['total_files'] += 1
                        info['total_size_bytes'] += file_size
                        
                        # File type
                        ext = os.path.splitext(file)[1].lower()
                        if ext:
                            info['file_types'][ext] = info['file_types'].get(ext, 0) + 1
                        
                        # Oldest file
                        if oldest_time is None or file_mtime < oldest_time:
                            oldest_time = file_mtime
                            info['oldest_file'] = {
                                'path': file_path,
                                'modified': file_mtime.isoformat(),
                                'size_bytes': file_size
                            }
                        
                        # Newest file
                        if newest_time is None or file_mtime > newest_time:
                            newest_time = file_mtime
                            info['newest_file'] = {
                                'path': file_path,
                                'modified': file_mtime.isoformat(),
                                'size_bytes': file_size
                            }
                        
                        # Largest file
                        if file_size > largest_size:
                            largest_size = file_size
                            info['largest_file'] = {
                                'path': file_path,
                                'size_bytes': file_size,
                                'size_mb': file_size / (1024 * 1024)
                            }
                    
                    except Exception as e:
                        logger.warning(f"Error analyzing file {file_path}: {e}")
            
            info['total_size_mb'] = info['total_size_bytes'] / (1024 * 1024)
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting directory info: {e}")
            return {
                'exists': False,
                'error': str(e)
            }
    
    def export_data_summary(self, data: pd.DataFrame, 
                           export_name: str) -> FileOperationResult:
        """
        Export data summary in multiple formats
        
        Args:
            data (pd.DataFrame): Data to export
            export_name (str): Base name for export files
            
        Returns:
            FileOperationResult: Export operation result
        """
        start_time = datetime.now()
        
        try:
            exports_dir = os.path.join(self.output_dir, 'exports')
            ensure_directory_exists(exports_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exported_files = []
            
            # Generate summary statistics
            summary_stats = {
                'total_records': len(data),
                'columns': list(data.columns),
                'data_types': data.dtypes.to_dict(),
                'missing_values': data.isnull().sum().to_dict(),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Add numeric summaries if available
            numeric_columns = data.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                summary_stats['numeric_summary'] = data[numeric_columns].describe().to_dict()
            
            # Export CSV
            csv_filename = f"{export_name}_data_{timestamp}.csv"
            csv_result = self.save_dataframe_to_csv(data, csv_filename, exports_dir)
            if csv_result.success:
                exported_files.extend(csv_result.file_paths)
            
            # Export JSON
            json_filename = f"{export_name}_data_{timestamp}.json"
            json_result = self.save_dataframe_to_json(data, json_filename, exports_dir)
            if json_result.success:
                exported_files.extend(json_result.file_paths)
            
            # Export summary
            summary_filename = f"{export_name}_summary_{timestamp}.json"
            summary_path = os.path.join(exports_dir, summary_filename)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            exported_files.append(summary_path)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = FileOperationResult(
                success=True,
                operation='export_data_summary',
                files_affected=len(exported_files),
                execution_time=execution_time,
                file_paths=exported_files
            )
            
            logger.info(f"Exported data summary: {len(exported_files)} files created")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error exporting data summary: {e}")
            return FileOperationResult(
                success=False,
                operation='export_data_summary',
                files_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive storage summary
        
        Returns:
            Dict: Storage summary information
        """
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'directories': {}
            }
            
            # Analyze key directories
            directories_to_analyze = [
                ('input', self.input_dir),
                ('output', self.output_dir),
                ('processed', self.processed_dir),
                ('archive', self.archive_dir),
                ('logs', 'logs')
            ]
            
            total_size_mb = 0
            total_files = 0
            
            for dir_name, dir_path in directories_to_analyze:
                dir_info = self.get_directory_info(dir_path)
                summary['directories'][dir_name] = dir_info
                
                if dir_info.get('exists', False):
                    total_size_mb += dir_info.get('total_size_mb', 0)
                    total_files += dir_info.get('total_files', 0)
            
            summary['totals'] = {
                'total_files': total_files,
                'total_size_mb': total_size_mb,
                'total_size_formatted': format_file_size(int(total_size_mb * 1024 * 1024))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting storage summary: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

if __name__ == "__main__":
    # Test file manager
    import pandas as pd
    import tempfile
    import os
    
    # Create test data
    test_data = pd.DataFrame([
        {
            'order_id': 'ORD-2024-001',
            'customer_name': 'John Doe',
            'product': 'iPhone 15',
            'quantity': 1,
            'price': 999.99,
            'order_date': '2024-01-15'
        },
        {
            'order_id': 'ORD-2024-002',
            'customer_name': 'Jane Smith',
            'product': 'MacBook Pro',
            'quantity': 1,
            'price': 1999.99,
            'order_date': '2024-01-16'
        }
    ])
    
    # Test file manager
    file_manager = FileManager()
    
    # Test save CSV
    print("Testing save CSV...")
    csv_result = file_manager.save_dataframe_to_csv(test_data, "test_orders.csv")
    print(f"CSV save result: {csv_result.success}, Files: {csv_result.files_affected}")
    
    # Test save JSON
    print("\nTesting save JSON...")
    json_result = file_manager.save_dataframe_to_json(test_data, "test_orders.json")
    print(f"JSON save result: {json_result.success}, Files: {json_result.files_affected}")
    
    # Test save report
    print("\nTesting save report...")
    report_content = "# Test Report\n\nThis is a test report.\n\n- Total orders: 2\n- Total value: $2999.98"
    report_result = file_manager.save_report(report_content, "test_report.md", "md")
    print(f"Report save result: {report_result.success}, Files: {report_result.files_affected}")
    
    # Test export data summary
    print("\nTesting export data summary...")
    export_result = file_manager.export_data_summary(test_data, "test_export")
    print(f"Export result: {export_result.success}, Files: {export_result.files_affected}")
    
    # Test storage summary
    print("\nTesting storage summary...")
    storage_summary = file_manager.get_storage_summary()
    print(f"Storage summary generated: {'error' not in storage_summary}")
    if 'totals' in storage_summary:
        totals = storage_summary['totals']
        print(f"Total files: {totals['total_files']}")
        print(f"Total size: {totals['total_size_formatted']}")
    
    print("File manager test completed!")
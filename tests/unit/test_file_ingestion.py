"""
Unit tests for file ingestion module
Tests CSV and JSON file processing, validation, and error handling
"""

import pytest
import pandas as pd
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.ingestion.file_ingestion import FileIngestion
from src.utils.constants import DataSourceType, FileExtension

@pytest.mark.unit
class TestFileIngestion:
    """Test cases for FileIngestion class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.file_ingestion = FileIngestion()
    
    def test_initialization(self):
        """Test FileIngestion initialization"""
        assert self.file_ingestion is not None
        assert hasattr(self.file_ingestion, 'input_dir')
        assert hasattr(self.file_ingestion, 'processed_dir')
        assert hasattr(self.file_ingestion, 'processed_files')
        assert isinstance(self.file_ingestion.processed_files, set)
    
    def test_discover_new_files_empty_directory(self, temp_dir):
        """Test discovering files in empty directory"""
        # Mock the input directories to use temp directory
        with patch.object(self.file_ingestion, 'input_dir', str(temp_dir)):
            # Create empty CSV and JSON directories
            (temp_dir / "csv").mkdir()
            (temp_dir / "json").mkdir()
            
            new_files = self.file_ingestion.discover_new_files()
            
            assert isinstance(new_files, dict)
            assert 'csv' in new_files
            assert 'json' in new_files
            assert len(new_files['csv']) == 0
            assert len(new_files['json']) == 0
    
    def test_discover_new_files_with_files(self, temp_dir, sample_orders_data):
        """Test discovering files with actual files present"""
        # Create test files
        csv_dir = temp_dir / "csv"
        json_dir = temp_dir / "json"
        csv_dir.mkdir()
        json_dir.mkdir()
        
        # Create test CSV file
        csv_file = csv_dir / "test_orders.csv"
        sample_orders_data.to_csv(csv_file, index=False)
        
        # Create test JSON file
        json_file = json_dir / "test_orders.json"
        with open(json_file, 'w') as f:
            json.dump({"orders": sample_orders_data.to_dict('records')}, f)
        
        # Mock the input directory
        with patch.object(self.file_ingestion, 'input_dir', str(temp_dir)):
            new_files = self.file_ingestion.discover_new_files()
            
            assert len(new_files['csv']) == 1
            assert len(new_files['json']) == 1
            assert str(csv_file) in new_files['csv']
            assert str(json_file) in new_files['json']
    
    def test_validate_file_valid(self, sample_csv_file):
        """Test file validation with valid file"""
        validation_result = self.file_ingestion.validate_file(str(sample_csv_file))
        
        assert validation_result['is_valid'] is True
        assert len(validation_result['issues']) == 0
        assert 'file_info' in validation_result
        assert 'size_mb' in validation_result['file_info']
        assert 'hash' in validation_result['file_info']
        assert 'modified_time' in validation_result['file_info']
    
    def test_validate_file_nonexistent(self):
        """Test file validation with non-existent file"""
        validation_result = self.file_ingestion.validate_file("nonexistent_file.csv")
        
        assert validation_result['is_valid'] is False
        assert "File does not exist" in validation_result['issues']
    
    def test_validate_file_empty(self, temp_dir):
        """Test file validation with empty file"""
        empty_file = temp_dir / "empty.csv"
        empty_file.touch()
        
        validation_result = self.file_ingestion.validate_file(str(empty_file))
        
        assert validation_result['is_valid'] is False
        assert "File is empty" in validation_result['issues']
    
    def test_validate_file_too_large(self, temp_dir):
        """Test file validation with file too large"""
        # Create a file and mock its size to be too large
        large_file = temp_dir / "large.csv"
        large_file.write_text("test data")
        
        # Mock max file size to be very small
        with patch.object(self.file_ingestion, 'max_file_size_mb', 0.000001):
            validation_result = self.file_ingestion.validate_file(str(large_file))
            
            assert validation_result['is_valid'] is False
            assert any("File too large" in issue for issue in validation_result['issues'])
    
    def test_process_csv_file_valid(self, sample_csv_file):
        """Test processing valid CSV file"""
        result = self.file_ingestion.process_csv_file(str(sample_csv_file))
        
        assert result['success'] is True
        assert result['records_count'] > 0
        assert result['data'] is not None
        assert isinstance(result['data'], pd.DataFrame)
        assert result['error_message'] is None
        assert result['processing_time'] > 0
        
        # Check metadata columns were added
        assert 'source_file' in result['data'].columns
        assert 'source_type' in result['data'].columns
        assert 'ingested_at' in result['data'].columns
        
        # Check source type is correct
        assert result['data']['source_type'].iloc[0] == DataSourceType.FILE_CSV.value
    
    def test_process_csv_file_invalid_encoding(self, temp_dir):
        """Test processing CSV file with encoding issues"""
        # Create a CSV file with special characters
        csv_file = temp_dir / "special_chars.csv"
        with open(csv_file, 'w', encoding='latin-1') as f:
            f.write("order_id,customer_name,product\n")
            f.write("ORD-001,José García,Café Special\n")
        
        result = self.file_ingestion.process_csv_file(str(csv_file))
        
        # Should succeed with encoding detection
        assert result['success'] is True
        assert result['records_count'] == 1
    
    def test_process_csv_file_malformed(self, temp_dir):
        """Test processing malformed CSV file"""
        # Create malformed CSV
        csv_file = temp_dir / "malformed.csv"
        with open(csv_file, 'w') as f:
            f.write("order_id,customer_name,product\n")
            f.write("ORD-001,John Doe\n")  # Missing column
            f.write("ORD-002,Jane Smith,Product,Extra Column\n")  # Extra column
        
        result = self.file_ingestion.process_csv_file(str(csv_file))
        
        # Should still succeed but may have issues
        assert result['success'] is True
        assert result['data'] is not None
    
    def test_process_json_file_valid(self, sample_json_file):
        """Test processing valid JSON file"""
        result = self.file_ingestion.process_json_file(str(sample_json_file))
        
        assert result['success'] is True
        assert result['records_count'] > 0
        assert result['data'] is not None
        assert isinstance(result['data'], pd.DataFrame)
        assert result['error_message'] is None
        
        # Check metadata columns were added
        assert 'source_file' in result['data'].columns
        assert 'source_type' in result['data'].columns
        assert 'ingested_at' in result['data'].columns
        
        # Check source type is correct
        assert result['data']['source_type'].iloc[0] == DataSourceType.FILE_JSON.value
    
    def test_process_json_file_different_structures(self, temp_dir):
        """Test processing JSON files with different structures"""
        # Test 1: JSON with 'orders' key
        json_file1 = temp_dir / "orders_structure.json"
        data1 = {"orders": [{"id": 1, "name": "Test"}]}
        with open(json_file1, 'w') as f:
            json.dump(data1, f)
        
        result1 = self.file_ingestion.process_json_file(str(json_file1))
        assert result1['success'] is True
        assert result1['records_count'] == 1
        
        # Test 2: JSON with 'data' key
        json_file2 = temp_dir / "data_structure.json"
        data2 = {"data": [{"id": 2, "name": "Test2"}]}
        with open(json_file2, 'w') as f:
            json.dump(data2, f)
        
        result2 = self.file_ingestion.process_json_file(str(json_file2))
        assert result2['success'] is True
        assert result2['records_count'] == 1
        
        # Test 3: JSON array
        json_file3 = temp_dir / "array_structure.json"
        data3 = [{"id": 3, "name": "Test3"}]
        with open(json_file3, 'w') as f:
            json.dump(data3, f)
        
        result3 = self.file_ingestion.process_json_file(str(json_file3))
        assert result3['success'] is True
        assert result3['records_count'] == 1
        
        # Test 4: Single JSON object
        json_file4 = temp_dir / "single_object.json"
        data4 = {"id": 4, "name": "Test4"}
        with open(json_file4, 'w') as f:
            json.dump(data4, f)
        
        result4 = self.file_ingestion.process_json_file(str(json_file4))
        assert result4['success'] is True
        assert result4['records_count'] == 1
    
    def test_process_json_file_invalid_json(self, temp_dir):
        """Test processing invalid JSON file"""
        json_file = temp_dir / "invalid.json"
        with open(json_file, 'w') as f:
            f.write("{ invalid json content")
        
        result = self.file_ingestion.process_json_file(str(json_file))
        
        assert result['success'] is False
        assert "Invalid JSON format" in result['error_message']
    
    def test_process_json_file_empty_data(self, temp_dir):
        """Test processing JSON file with no order data"""
        json_file = temp_dir / "empty_data.json"
        with open(json_file, 'w') as f:
            json.dump({"orders": []}, f)
        
        result = self.file_ingestion.process_json_file(str(json_file))
        
        assert result['success'] is False
        assert "No order data found" in result['error_message']
    
    def test_mark_file_as_processed_success(self, temp_dir, sample_csv_file):
        """Test marking file as processed successfully"""
        # Mock processed directory
        processed_dir = temp_dir / "processed"
        processed_dir.mkdir()
        
        with patch.object(self.file_ingestion, 'processed_dir', str(processed_dir)):
            result = self.file_ingestion.mark_file_as_processed(str(sample_csv_file), success=True)
            
            assert result is True
            assert str(sample_csv_file) in self.file_ingestion.processed_files
            
            # Check that file was moved to processed directory
            processed_files = list(processed_dir.glob("processed_*"))
            assert len(processed_files) == 1
    
    def test_mark_file_as_processed_failure(self, temp_dir, sample_csv_file):
        """Test marking file as processed with failure"""
        # Mock processed directory
        processed_dir = temp_dir / "processed"
        processed_dir.mkdir()
        
        with patch.object(self.file_ingestion, 'processed_dir', str(processed_dir)):
            result = self.file_ingestion.mark_file_as_processed(str(sample_csv_file), success=False)
            
            assert result is True
            assert str(sample_csv_file) in self.file_ingestion.processed_files
            
            # Check that file was moved to error directory
            error_dir = processed_dir / "errors"
            assert error_dir.exists()
            error_files = list(error_dir.glob("error_*"))
            assert len(error_files) == 1
    
    def test_process_all_files_no_files(self, temp_dir):
        """Test processing all files when no files are present"""
        with patch.object(self.file_ingestion, 'input_dir', str(temp_dir)):
            # Create empty directories
            (temp_dir / "csv").mkdir()
            (temp_dir / "json").mkdir()
            
            summary = self.file_ingestion.process_all_files()
            
            assert summary['total_files'] == 0
            assert summary['successful_files'] == 0
            assert summary['failed_files'] == 0
            assert summary['total_records'] == 0
            assert len(summary['results']) == 0
            assert len(summary['errors']) == 0
    
    def test_process_all_files_with_files(self, temp_dir, sample_orders_data):
        """Test processing all files with actual files"""
        # Create test files
        csv_dir = temp_dir / "csv"
        json_dir = temp_dir / "json"
        processed_dir = temp_dir / "processed"
        csv_dir.mkdir()
        json_dir.mkdir()
        processed_dir.mkdir()
        
        # Create test CSV file
        csv_file = csv_dir / "test_orders.csv"
        sample_orders_data.to_csv(csv_file, index=False)
        
        # Create test JSON file
        json_file = json_dir / "test_orders.json"
        json_data = {"orders": sample_orders_data.to_dict('records')}
        with open(json_file, 'w') as f:
            json.dump(json_data, f)
        
        # Mock directories
        with patch.object(self.file_ingestion, 'input_dir', str(temp_dir)), \
             patch.object(self.file_ingestion, 'processed_dir', str(processed_dir)):
            
            summary = self.file_ingestion.process_all_files()
            
            assert summary['total_files'] == 2
            assert summary['successful_files'] == 2
            assert summary['failed_files'] == 0
            assert summary['total_records'] > 0
            assert len(summary['results']) == 2
            assert len(summary['errors']) == 0
    
    def test_process_all_files_with_errors(self, temp_dir):
        """Test processing all files with some files causing errors"""
        # Create directories
        csv_dir = temp_dir / "csv"
        processed_dir = temp_dir / "processed"
        csv_dir.mkdir()
        processed_dir.mkdir()
        
        # Create invalid CSV file
        invalid_csv = csv_dir / "invalid.csv"
        with open(invalid_csv, 'w') as f:
            f.write("invalid csv content without proper structure")
        
        # Create empty CSV file
        empty_csv = csv_dir / "empty.csv"
        empty_csv.touch()
        
        with patch.object(self.file_ingestion, 'input_dir', str(temp_dir)), \
             patch.object(self.file_ingestion, 'processed_dir', str(processed_dir)):
            
            summary = self.file_ingestion.process_all_files()
            
            assert summary['total_files'] == 2
            assert summary['failed_files'] > 0
            assert len(summary['errors']) > 0
    
    def test_get_combined_data_success(self, temp_dir, sample_orders_data):
        """Test getting combined data from successful processing"""
        # Create test files
        csv_dir = temp_dir / "csv"
        processed_dir = temp_dir / "processed"
        csv_dir.mkdir()
        processed_dir.mkdir()
        
        # Create test CSV file
        csv_file = csv_dir / "test_orders.csv"
        sample_orders_data.to_csv(csv_file, index=False)
        
        with patch.object(self.file_ingestion, 'input_dir', str(temp_dir)), \
             patch.object(self.file_ingestion, 'processed_dir', str(processed_dir)):
            
            combined_data = self.file_ingestion.get_combined_data()
            
            assert combined_data is not None
            assert isinstance(combined_data, pd.DataFrame)
            assert len(combined_data) > 0
            
            # Check that metadata columns are present
            assert 'source_file' in combined_data.columns
            assert 'source_type' in combined_data.columns
            assert 'ingested_at' in combined_data.columns
    
    def test_get_combined_data_no_files(self, temp_dir):
        """Test getting combined data when no files are processed"""
        with patch.object(self.file_ingestion, 'input_dir', str(temp_dir)):
            # Create empty directories
            (temp_dir / "csv").mkdir()
            (temp_dir / "json").mkdir()
            
            combined_data = self.file_ingestion.get_combined_data()
            
            assert combined_data is None
    
    def test_get_combined_data_duplicate_removal(self, temp_dir, sample_orders_data):
        """Test that combined data removes duplicates based on order_id"""
        # Create directories
        csv_dir = temp_dir / "csv"
        processed_dir = temp_dir / "processed"
        csv_dir.mkdir()
        processed_dir.mkdir()
        
        # Create two CSV files with overlapping data
        csv_file1 = csv_dir / "orders1.csv"
        csv_file2 = csv_dir / "orders2.csv"
        
        sample_orders_data.to_csv(csv_file1, index=False)
        sample_orders_data.to_csv(csv_file2, index=False)  # Same data
        
        with patch.object(self.file_ingestion, 'input_dir', str(temp_dir)), \
             patch.object(self.file_ingestion, 'processed_dir', str(processed_dir)):
            
            combined_data = self.file_ingestion.get_combined_data()
            
            assert combined_data is not None
            # Should have same number of records as original data (duplicates removed)
            assert len(combined_data) == len(sample_orders_data)
    
    @patch('src.ingestion.file_ingestion.logger')
    def test_logging_calls(self, mock_logger, temp_dir, sample_csv_file):
        """Test that appropriate logging calls are made"""
        processed_dir = temp_dir / "processed"
        processed_dir.mkdir()
        
        with patch.object(self.file_ingestion, 'processed_dir', str(processed_dir)):
            # Process a file
            result = self.file_ingestion.process_csv_file(str(sample_csv_file))
            
            # Check that info logging was called
            mock_logger.info.assert_called()
            
            # Check for specific log messages
            log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("Processing CSV file" in msg for msg in log_calls)
            assert any("Successfully processed CSV file" in msg for msg in log_calls)
    
    def test_file_extension_validation(self, temp_dir):
        """Test file extension validation"""
        # Create files with different extensions
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test content")
        
        validation_result = self.file_ingestion.validate_file(str(txt_file))
        
        # Should fail validation due to unsupported extension
        assert validation_result['is_valid'] is False
        assert any("Unsupported file type" in issue for issue in validation_result['issues'])
    
    def test_concurrent_file_processing(self, temp_dir, sample_orders_data):
        """Test that processed files tracking works correctly"""
        # Create test file
        csv_dir = temp_dir / "csv"
        csv_dir.mkdir()
        csv_file = csv_dir / "test_orders.csv"
        sample_orders_data.to_csv(csv_file, index=False)
        
        # Simulate file already being processed
        self.file_ingestion.processed_files.add(str(csv_file))
        
        with patch.object(self.file_ingestion, 'input_dir', str(temp_dir)):
            new_files = self.file_ingestion.discover_new_files()
            
            # Should not include already processed file
            assert len(new_files['csv']) == 0
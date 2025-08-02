"""
Integration tests for end-to-end pipeline execution
Tests complete data flow from ingestion through storage with real components
"""

import pytest
import pandas as pd
import sqlite3
import json
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.pipeline.pipeline_manager import PipelineManager
from src.ingestion.file_ingestion import FileIngestion
from src.validation.data_validator import DataValidator
from src.transformation.data_cleaner import DataCleaner
from src.storage.database_manager import DatabaseManager

@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for complete pipeline execution"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Create temporary directories for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.processed_dir = self.temp_dir / "processed"
        
        # Create directory structure
        (self.input_dir / "csv").mkdir(parents=True)
        (self.input_dir / "json").mkdir(parents=True)
        self.output_dir.mkdir()
        self.processed_dir.mkdir()
        
        # Create temporary database
        self.db_path = self.temp_dir / "test_orders.db"
        self._setup_test_database()
        
        # Initialize pipeline manager
        self.pipeline_manager = PipelineManager("integration_test_pipeline")
    
    def teardown_method(self):
        """Cleanup after each test method"""
        # Remove temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _setup_test_database(self):
        """Setup test database with required tables"""
        conn = sqlite3.connect(str(self.db_path))
        
        # Create orders table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id VARCHAR(50) UNIQUE NOT NULL,
                customer_name VARCHAR(100) NOT NULL,
                customer_email VARCHAR(100),
                product VARCHAR(200) NOT NULL,
                product_category VARCHAR(50),
                quantity INTEGER NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                discount DECIMAL(10,2) DEFAULT 0,
                total_amount DECIMAL(10,2),
                order_date DATE NOT NULL,
                store_location VARCHAR(100),
                source VARCHAR(50),
                source_file VARCHAR(200),
                source_type VARCHAR(50),
                ingested_at TIMESTAMP,
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create pipeline_runs table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id VARCHAR(100) UNIQUE NOT NULL,
                pipeline_name VARCHAR(100) NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status VARCHAR(20) NOT NULL,
                records_processed INTEGER DEFAULT 0,
                records_failed INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create data_quality_metrics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS data_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id VARCHAR(100),
                metric_name VARCHAR(100) NOT NULL,
                metric_value DECIMAL(10,4),
                metric_type VARCHAR(50),
                source_table VARCHAR(100),
                measured_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.close()
    
    def _create_test_csv_file(self, filename: str, data: pd.DataFrame):
        """Create a test CSV file"""
        csv_path = self.input_dir / "csv" / filename
        data.to_csv(csv_path, index=False)
        return csv_path
    
    def _create_test_json_file(self, filename: str, data: dict):
        """Create a test JSON file"""
        json_path = self.input_dir / "json" / filename
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        return json_path
    
    def test_complete_pipeline_with_csv_data(self, sample_orders_data):
        """Test complete pipeline execution with CSV data"""
        # Create test CSV file
        csv_file = self._create_test_csv_file("test_orders.csv", sample_orders_data)
        
        # Configure pipeline to use test directories and database
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run complete pipeline
            result = self.pipeline_manager.run_pipeline()
            
            # Verify pipeline execution
            assert result.success is True
            assert result.total_records_processed > 0
            assert len(result.stages_completed) >= 3  # At least ingestion, transformation, storage
            assert len(result.stages_failed) == 0
            
            # Verify data was stored in database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM orders")
            order_count = cursor.fetchone()[0]
            conn.close()
            
            assert order_count > 0
            
            # Verify file was moved to processed directory
            processed_files = list(self.processed_dir.glob("processed_*"))
            assert len(processed_files) > 0
    
    def test_complete_pipeline_with_json_data(self, sample_orders_data):
        """Test complete pipeline execution with JSON data"""
        # Create test JSON file
        json_data = {
            "app_version": "2.1.0",
            "upload_time": "2024-01-15T12:00:00Z",
            "orders": sample_orders_data.to_dict('records')
        }
        json_file = self._create_test_json_file("test_orders.json", json_data)
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run pipeline
            result = self.pipeline_manager.run_pipeline()
            
            # Verify results
            assert result.success is True
            assert result.total_records_processed > 0
            
            # Verify data in database
            conn = sqlite3.connect(str(self.db_path))
            df = pd.read_sql_query("SELECT * FROM orders", conn)
            conn.close()
            
            assert len(df) > 0
            assert 'source_type' in df.columns
            assert df['source_type'].iloc[0] == 'file_json'
    
    def test_complete_pipeline_with_mixed_data(self, sample_orders_data):
        """Test complete pipeline execution with mixed CSV and JSON data"""
        # Create CSV file
        csv_data = sample_orders_data.iloc[:2]  # First 2 records
        csv_file = self._create_test_csv_file("orders1.csv", csv_data)
        
        # Create JSON file
        json_data = {
            "orders": sample_orders_data.iloc[2:].to_dict('records')  # Last record
        }
        json_file = self._create_test_json_file("orders1.json", json_data)
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run pipeline
            result = self.pipeline_manager.run_pipeline()
            
            # Verify results
            assert result.success is True
            assert result.total_records_processed == len(sample_orders_data)
            
            # Verify mixed data sources in database
            conn = sqlite3.connect(str(self.db_path))
            df = pd.read_sql_query("SELECT source_type, COUNT(*) as count FROM orders GROUP BY source_type", conn)
            conn.close()
            
            source_types = df['source_type'].tolist()
            assert 'file_csv' in source_types
            assert 'file_json' in source_types
    
    def test_pipeline_with_invalid_data(self, invalid_orders_data):
        """Test pipeline execution with invalid data"""
        # Create CSV file with invalid data
        csv_file = self._create_test_csv_file("invalid_orders.csv", invalid_orders_data)
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run pipeline
            result = self.pipeline_manager.run_pipeline()
            
            # Pipeline should complete but with warnings/errors
            # The exact behavior depends on validation configuration
            assert result is not None
            assert result.run_id is not None
            
            # Check validation results
            if 'validation' in result.stage_results:
                validation_result = result.stage_results['validation']
                assert 'quality_score' in validation_result
                # Quality score should be low due to invalid data
                assert validation_result['quality_score'] < 80
    
    def test_pipeline_with_no_data(self):
        """Test pipeline execution with no input data"""
        # Don't create any input files
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Disable API ingestion to ensure no data
            self.pipeline_manager.enable_api_ingestion = False
            
            # Run pipeline
            result = self.pipeline_manager.run_pipeline()
            
            # Pipeline should fail due to no data
            assert result.success is False
            assert result.total_records_processed == 0
            assert 'ingestion' in result.stages_failed
    
    def test_pipeline_data_transformation_flow(self, sample_orders_data):
        """Test that data transformation flow works correctly"""
        # Add some data that needs cleaning
        messy_data = sample_orders_data.copy()
        messy_data.loc[0, 'customer_name'] = '  john doe  '  # Extra whitespace
        messy_data.loc[1, 'product'] = 'macbook pro'  # Lowercase
        messy_data.loc[2, 'source'] = 'WEB'  # Uppercase
        
        # Create test file
        csv_file = self._create_test_csv_file("messy_orders.csv", messy_data)
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run pipeline
            result = self.pipeline_manager.run_pipeline()
            
            # Verify transformation occurred
            assert result.success is True
            
            # Check that data was cleaned in database
            conn = sqlite3.connect(str(self.db_path))
            df = pd.read_sql_query("SELECT customer_name, product, source FROM orders", conn)
            conn.close()
            
            # Verify cleaning occurred
            assert df.loc[0, 'customer_name'] == 'John Doe'  # Should be title case and trimmed
            assert df.loc[2, 'source'] == 'web'  # Should be lowercase
    
    def test_pipeline_duplicate_handling(self, sample_orders_data):
        """Test pipeline handling of duplicate data"""
        # Create two files with overlapping data
        csv_file1 = self._create_test_csv_file("orders1.csv", sample_orders_data)
        csv_file2 = self._create_test_csv_file("orders2.csv", sample_orders_data)  # Same data
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run pipeline
            result = self.pipeline_manager.run_pipeline()
            
            # Verify deduplication occurred
            assert result.success is True
            
            # Check database for duplicates
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM orders")
            total_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT order_id) FROM orders")
            unique_count = cursor.fetchone()[0]
            conn.close()
            
            # Should have deduplicated based on order_id
            assert total_count == unique_count
            assert total_count == len(sample_orders_data)  # Original count, not doubled
    
    def test_pipeline_error_recovery(self, sample_orders_data):
        """Test pipeline error recovery and partial processing"""
        # Create one valid file and one invalid file
        valid_csv = self._create_test_csv_file("valid_orders.csv", sample_orders_data)
        
        # Create invalid CSV file
        invalid_csv_path = self.input_dir / "csv" / "invalid_orders.csv"
        with open(invalid_csv_path, 'w') as f:
            f.write("invalid,csv,content,without,proper,headers\n")
            f.write("this,is,not,valid,csv,data\n")
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run pipeline
            result = self.pipeline_manager.run_pipeline()
            
            # Pipeline should succeed with valid data despite invalid file
            assert result.success is True
            assert result.total_records_processed > 0
            
            # Check that valid data was processed
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM orders")
            order_count = cursor.fetchone()[0]
            conn.close()
            
            assert order_count > 0
            
            # Check that both files were moved to processed directory
            processed_files = list(self.processed_dir.glob("*"))
            error_files = list((self.processed_dir / "errors").glob("*")) if (self.processed_dir / "errors").exists() else []
            
            # Should have at least one processed file and possibly one error file
            assert len(processed_files) + len(error_files) >= 2
    
    def test_pipeline_performance_with_large_dataset(self):
        """Test pipeline performance with larger dataset"""
        # Create larger dataset
        large_data = pd.DataFrame({
            'order_id': [f'ORD-2024-{i:05d}' for i in range(1, 501)],  # 500 records
            'customer_name': [f'Customer {i}' for i in range(1, 501)],
            'product': ['Test Product'] * 500,
            'quantity': [1] * 500,
            'price': [99.99] * 500,
            'order_date': ['2024-01-15'] * 500,
            'source': ['test'] * 500
        })
        
        # Create test file
        csv_file = self._create_test_csv_file("large_orders.csv", large_data)
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run pipeline and measure time
            start_time = datetime.now()
            result = self.pipeline_manager.run_pipeline()
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Verify results
            assert result.success is True
            assert result.total_records_processed == 500
            
            # Should complete in reasonable time (less than 30 seconds)
            assert execution_time < 30.0
            
            # Verify all data was stored
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM orders")
            order_count = cursor.fetchone()[0]
            conn.close()
            
            assert order_count == 500
    
    def test_pipeline_metadata_tracking(self, sample_orders_data):
        """Test that pipeline tracks metadata correctly"""
        # Create test file
        csv_file = self._create_test_csv_file("metadata_test.csv", sample_orders_data)
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run pipeline
            result = self.pipeline_manager.run_pipeline()
            
            # Verify pipeline run was recorded
            conn = sqlite3.connect(str(self.db_path))
            
            # Check pipeline_runs table
            pipeline_runs = pd.read_sql_query("SELECT * FROM pipeline_runs", conn)
            assert len(pipeline_runs) > 0
            assert pipeline_runs.iloc[0]['run_id'] == result.run_id
            assert pipeline_runs.iloc[0]['pipeline_name'] == result.pipeline_name
            
            # Check data_quality_metrics table
            metrics = pd.read_sql_query("SELECT * FROM data_quality_metrics", conn)
            if len(metrics) > 0:
                assert metrics.iloc[0]['run_id'] == result.run_id
            
            # Check orders metadata
            orders = pd.read_sql_query("SELECT * FROM orders", conn)
            assert len(orders) > 0
            assert 'source_file' in orders.columns
            assert 'source_type' in orders.columns
            assert 'ingested_at' in orders.columns
            assert 'processed_at' in orders.columns
            
            conn.close()
    
    def test_pipeline_configuration_flexibility(self, sample_orders_data):
        """Test pipeline with different stage configurations"""
        # Create test file
        csv_file = self._create_test_csv_file("config_test.csv", sample_orders_data)
        
        # Test with validation disabled
        self.pipeline_manager.enable_validation = False
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run pipeline
            result = self.pipeline_manager.run_pipeline()
            
            # Verify pipeline succeeded without validation
            assert result.success is True
            assert 'validation' not in result.stage_results
            assert 'validation' not in result.stages_completed
            
            # Data should still be processed and stored
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM orders")
            order_count = cursor.fetchone()[0]
            conn.close()
            
            assert order_count > 0
    
    def test_pipeline_report_generation(self, sample_orders_data):
        """Test that pipeline generates comprehensive reports"""
        # Create test file
        csv_file = self._create_test_csv_file("report_test.csv", sample_orders_data)
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run pipeline
            result = self.pipeline_manager.run_pipeline()
            
            # Generate report
            report = self.pipeline_manager.generate_pipeline_report(result)
            
            # Verify report content
            assert isinstance(report, str)
            assert len(report) > 0
            assert "# Data Ingestion Pipeline Execution Report" in report
            assert result.run_id in report
            assert str(result.total_records_processed) in report
            
            # Report should contain stage information
            for stage in result.stages_completed:
                assert stage in report
            
            # Should contain execution summary
            assert "Executive Summary" in report
            assert "Stage Results" in report
    
    @pytest.mark.slow
    def test_pipeline_stress_test(self):
        """Stress test with multiple files and larger dataset"""
        # Create multiple CSV files with different data
        for i in range(5):
            data = pd.DataFrame({
                'order_id': [f'ORD-{i:02d}-{j:03d}' for j in range(1, 101)],  # 100 records per file
                'customer_name': [f'Customer {i}-{j}' for j in range(1, 101)],
                'product': [f'Product {i}'] * 100,
                'quantity': [1] * 100,
                'price': [99.99 + i] * 100,
                'order_date': ['2024-01-15'] * 100,
                'source': ['test'] * 100
            })
            self._create_test_csv_file(f"stress_test_{i}.csv", data)
        
        # Configure pipeline
        with patch.object(self.pipeline_manager.file_ingestion, 'input_dir', str(self.input_dir)), \
             patch.object(self.pipeline_manager.file_ingestion, 'processed_dir', str(self.processed_dir)), \
             patch.object(self.pipeline_manager.database_manager, 'db_path', str(self.db_path)):
            
            # Run pipeline
            start_time = datetime.now()
            result = self.pipeline_manager.run_pipeline()
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Verify results
            assert result.success is True
            assert result.total_records_processed == 500  # 5 files * 100 records
            
            # Should complete in reasonable time
            assert execution_time < 60.0  # Less than 1 minute
            
            # Verify all data was stored
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM orders")
            order_count = cursor.fetchone()[0]
            conn.close()
            
            assert order_count == 500
            
            # Verify all files were processed
            processed_files = list(self.processed_dir.glob("processed_*"))
            assert len(processed_files) == 5
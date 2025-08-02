"""
Unit tests for database manager module
Tests CRUD operations, connection handling, and database performance
"""

import pytest
import pandas as pd
import sqlite3
import os
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.storage.database_manager import DatabaseManager, DatabaseResult

@pytest.mark.unit
@pytest.mark.database
class TestDatabaseManager:
    """Test cases for DatabaseManager class"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Use temporary database for testing
        self.temp_db_path = tempfile.mktemp(suffix='.db')
        self.db_manager = DatabaseManager(self.temp_db_path)
    
    def teardown_method(self):
        """Cleanup after each test method"""
        # Close any open connections and remove temp database
        if hasattr(self.db_manager, 'connection_pools'):
            self.db_manager.connection_pools.clear()
        
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except:
                pass
    
    def test_initialization(self):
        """Test DatabaseManager initialization"""
        assert self.db_manager is not None
        assert self.db_manager.db_path == self.temp_db_path
        assert hasattr(self.db_manager, 'timeout')
        assert os.path.exists(self.temp_db_path)  # Database file should be created
    
    def test_get_connection_context_manager(self):
        """Test database connection context manager"""
        with self.db_manager.get_connection() as conn:
            assert conn is not None
            # Test basic query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_get_connection_error_handling(self):
        """Test connection error handling"""
        # Test with invalid database path
        invalid_db_manager = DatabaseManager("/invalid/path/database.db")
        
        with pytest.raises(Exception):
            with invalid_db_manager.get_connection() as conn:
                pass
    
    def test_save_orders_success(self, sample_orders_data):
        """Test successful order saving"""
        result = self.db_manager.save_orders(sample_orders_data)
        
        assert isinstance(result, DatabaseResult)
        assert result.success is True
        assert result.operation == 'save_orders'
        assert result.records_affected == len(sample_orders_data)
        assert result.execution_time > 0
        assert result.error_message is None
    
    def test_save_orders_empty_dataframe(self):
        """Test saving empty DataFrame"""
        empty_df = pd.DataFrame()
        result = self.db_manager.save_orders(empty_df)
        
        assert result.success is True
        assert result.records_affected == 0
    
    def test_save_orders_batch_processing(self):
        """Test batch processing for large datasets"""
        # Create larger dataset
        large_data = pd.DataFrame({
            'order_id': [f'ORD-2024-{i:03d}' for i in range(1, 101)],
            'customer_name': [f'Customer {i}' for i in range(1, 101)],
            'product': ['Test Product'] * 100,
            'quantity': [1] * 100,
            'price': [99.99] * 100,
            'order_date': ['2024-01-15'] * 100
        })
        
        # Test with small batch size
        result = self.db_manager.save_orders(large_data, batch_size=10)
        
        assert result.success is True
        assert result.records_affected == 100
    
    def test_save_orders_duplicate_handling(self, sample_orders_data):
        """Test handling of duplicate orders (INSERT OR REPLACE)"""
        # Save orders first time
        result1 = self.db_manager.save_orders(sample_orders_data)
        assert result1.success is True
        
        # Save same orders again (should replace)
        result2 = self.db_manager.save_orders(sample_orders_data)
        assert result2.success is True
        assert result2.records_affected == len(sample_orders_data)
        
        # Verify total count is still the same (replaced, not duplicated)
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM orders")
            count = cursor.fetchone()[0]
            assert count == len(sample_orders_data)
    
    def test_get_orders_all(self, sample_orders_data):
        """Test retrieving all orders"""
        # First save some orders
        self.db_manager.save_orders(sample_orders_data)
        
        # Retrieve all orders
        result = self.db_manager.get_orders()
        
        assert isinstance(result, DatabaseResult)
        assert result.success is True
        assert result.operation == 'get_orders'
        assert result.records_affected == len(sample_orders_data)
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == len(sample_orders_data)
    
    def test_get_orders_with_filters(self, sample_orders_data):
        """Test retrieving orders with filters"""
        # Save orders
        self.db_manager.save_orders(sample_orders_data)
        
        # Test single filter
        filters = {'source': 'website'}
        result = self.db_manager.get_orders(filters=filters)
        
        assert result.success is True
        assert result.data is not None
        if len(result.data) > 0:
            assert all(result.data['source'] == 'website')
        
        # Test multiple filters
        filters = {'source': 'store', 'product': 'MacBook Pro'}
        result = self.db_manager.get_orders(filters=filters)
        
        assert result.success is True
        assert result.data is not None
    
    def test_get_orders_with_limit_offset(self, sample_orders_data):
        """Test retrieving orders with limit and offset"""
        # Save orders
        self.db_manager.save_orders(sample_orders_data)
        
        # Test with limit
        result = self.db_manager.get_orders(limit=2)
        assert result.success is True
        assert len(result.data) <= 2
        
        # Test with limit and offset
        result = self.db_manager.get_orders(limit=1, offset=1)
        assert result.success is True
        assert len(result.data) <= 1
    
    def test_get_orders_with_list_filter(self, sample_orders_data):
        """Test retrieving orders with list-based filters (IN clause)"""
        # Save orders
        self.db_manager.save_orders(sample_orders_data)
        
        # Test IN filter
        filters = {'source': ['website', 'store']}
        result = self.db_manager.get_orders(filters=filters)
        
        assert result.success is True
        assert result.data is not None
        if len(result.data) > 0:
            assert all(source in ['website', 'store'] for source in result.data['source'])
    
    def test_update_orders_success(self, sample_orders_data):
        """Test successful order updates"""
        # Save orders first
        self.db_manager.save_orders(sample_orders_data)
        
        # Update orders
        updates = {'source': 'updated_source'}
        filters = {'product': 'iPhone 15'}
        
        result = self.db_manager.update_orders(updates, filters)
        
        assert isinstance(result, DatabaseResult)
        assert result.success is True
        assert result.operation == 'update_orders'
        assert result.records_affected >= 0
        assert result.execution_time > 0
        
        # Verify update worked
        get_result = self.db_manager.get_orders(filters={'source': 'updated_source'})
        assert get_result.success is True
        if len(get_result.data) > 0:
            assert all(get_result.data['source'] == 'updated_source')
    
    def test_update_orders_no_matches(self, sample_orders_data):
        """Test updating orders with no matching records"""
        # Save orders first
        self.db_manager.save_orders(sample_orders_data)
        
        # Try to update non-existent records
        updates = {'source': 'updated_source'}
        filters = {'product': 'Non-existent Product'}
        
        result = self.db_manager.update_orders(updates, filters)
        
        assert result.success is True
        assert result.records_affected == 0
    
    def test_delete_orders_success(self, sample_orders_data):
        """Test successful order deletion"""
        # Save orders first
        self.db_manager.save_orders(sample_orders_data)
        
        # Delete specific orders
        filters = {'product': 'iPhone 15'}
        result = self.db_manager.delete_orders(filters)
        
        assert isinstance(result, DatabaseResult)
        assert result.success is True
        assert result.operation == 'delete_orders'
        assert result.records_affected >= 0
        assert result.execution_time > 0
        
        # Verify deletion worked
        get_result = self.db_manager.get_orders(filters=filters)
        assert get_result.success is True
        assert len(get_result.data) == 0
    
    def test_delete_orders_no_matches(self, sample_orders_data):
        """Test deleting orders with no matching records"""
        # Save orders first
        self.db_manager.save_orders(sample_orders_data)
        
        # Try to delete non-existent records
        filters = {'product': 'Non-existent Product'}
        result = self.db_manager.delete_orders(filters)
        
        assert result.success is True
        assert result.records_affected == 0
    
    def test_save_pipeline_run(self):
        """Test saving pipeline run information"""
        run_data = {
            'run_id': 'TEST-RUN-001',
            'pipeline_name': 'test_pipeline',
            'start_time': '2024-01-15T10:00:00',
            'end_time': '2024-01-15T10:30:00',
            'status': 'completed',
            'records_processed': 100,
            'records_failed': 5,
            'error_message': None
        }
        
        result = self.db_manager.save_pipeline_run(run_data)
        
        assert isinstance(result, DatabaseResult)
        assert result.success is True
        assert result.operation == 'save_pipeline_run'
        assert result.records_affected == 1
    
    def test_save_data_quality_metrics(self):
        """Test saving data quality metrics"""
        metrics_data = [
            {
                'run_id': 'TEST-RUN-001',
                'metric_name': 'data_quality_score',
                'metric_value': 95.5,
                'metric_type': 'percentage',
                'source_table': 'orders',
                'measured_at': '2024-01-15T10:30:00'
            },
            {
                'run_id': 'TEST-RUN-001',
                'metric_name': 'records_processed',
                'metric_value': 100,
                'metric_type': 'count',
                'source_table': 'orders',
                'measured_at': '2024-01-15T10:30:00'
            }
        ]
        
        result = self.db_manager.save_data_quality_metrics(metrics_data)
        
        assert result.success is True
        assert result.operation == 'save_data_quality_metrics'
        assert result.records_affected == 2
    
    def test_get_pipeline_runs(self):
        """Test retrieving pipeline runs"""
        # Save a pipeline run first
        run_data = {
            'run_id': 'TEST-RUN-001',
            'pipeline_name': 'test_pipeline',
            'start_time': '2024-01-15T10:00:00',
            'end_time': '2024-01-15T10:30:00',
            'status': 'completed',
            'records_processed': 100,
            'records_failed': 5
        }
        self.db_manager.save_pipeline_run(run_data)
        
        # Retrieve pipeline runs
        result = self.db_manager.get_pipeline_runs(limit=10)
        
        assert result.success is True
        assert result.operation == 'get_pipeline_runs'
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) >= 1
    
    def test_get_database_stats(self):
        """Test retrieving database statistics"""
        result = self.db_manager.get_database_stats()
        
        assert isinstance(result, DatabaseResult)
        assert result.success is True
        assert result.operation == 'get_database_stats'
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 1
        
        # Check that stats contain expected fields
        stats = result.data.iloc[0]
        assert 'orders_count' in stats.index
        assert 'database_size_bytes' in stats.index
        assert 'database_size_mb' in stats.index
    
    def test_execute_custom_query_select(self, sample_orders_data):
        """Test executing custom SELECT query"""
        # Save some data first
        self.db_manager.save_orders(sample_orders_data)
        
        # Execute custom query
        query = "SELECT COUNT(*) as order_count FROM orders"
        result = self.db_manager.execute_custom_query(query)
        
        assert result.success is True
        assert result.operation == 'execute_custom_query'
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert 'order_count' in result.data.columns
        assert result.data.iloc[0]['order_count'] == len(sample_orders_data)
    
    def test_execute_custom_query_with_parameters(self, sample_orders_data):
        """Test executing custom query with parameters"""
        # Save some data first
        self.db_manager.save_orders(sample_orders_data)
        
        # Execute parameterized query
        query = "SELECT * FROM orders WHERE source = ?"
        params = ['website']
        result = self.db_manager.execute_custom_query(query, params)
        
        assert result.success is True
        assert result.data is not None
        if len(result.data) > 0:
            assert all(result.data['source'] == 'website')
    
    def test_execute_custom_query_update(self, sample_orders_data):
        """Test executing custom UPDATE query"""
        # Save some data first
        self.db_manager.save_orders(sample_orders_data)
        
        # Execute update query
        query = "UPDATE orders SET source = 'updated' WHERE product = 'iPhone 15'"
        result = self.db_manager.execute_custom_query(query)
        
        assert result.success is True
        assert result.operation == 'execute_custom_query'
        assert result.records_affected >= 0
        assert result.data is None  # UPDATE queries don't return data
    
    def test_backup_database(self, temp_dir):
        """Test database backup functionality"""
        backup_path = temp_dir / "backup.db"
        
        result = self.db_manager.backup_database(str(backup_path))
        
        assert isinstance(result, DatabaseResult)
        assert result.success is True
        assert result.operation == 'backup_database'
        assert result.records_affected == 1
        assert backup_path.exists()
        assert backup_path.stat().st_size > 0
    
    def test_optimize_database(self):
        """Test database optimization"""
        result = self.db_manager.optimize_database()
        
        assert isinstance(result, DatabaseResult)
        assert result.success is True
        assert result.operation == 'optimize_database'
        assert result.records_affected == 1
        assert result.execution_time > 0
    
    def test_prepare_orders_data(self, sample_orders_data):
        """Test order data preparation for database insertion"""
        prepared_data = self.db_manager._prepare_orders_data(sample_orders_data)
        
        assert isinstance(prepared_data, list)
        assert len(prepared_data) == len(sample_orders_data)
        
        # Check that each tuple has the expected number of fields
        for order_tuple in prepared_data:
            assert isinstance(order_tuple, tuple)
            assert len(order_tuple) == 15  # Expected number of fields
    
    def test_connection_timeout(self):
        """Test connection timeout handling"""
        # Create database manager with very short timeout
        short_timeout_manager = DatabaseManager(self.temp_db_path)
        short_timeout_manager.timeout = 0.001  # Very short timeout
        
        # This should still work for simple operations
        with short_timeout_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_database_error_handling(self):
        """Test database error handling"""
        # Try to execute invalid SQL
        result = self.db_manager.execute_custom_query("INVALID SQL QUERY")
        
        assert result.success is False
        assert result.error_message is not None
        assert "INVALID SQL QUERY" not in result.error_message  # Should be sanitized
    
    def test_transaction_rollback_on_error(self):
        """Test that transactions are rolled back on error"""
        # This test would require more complex setup to force a transaction error
        # For now, we'll test that the context manager handles exceptions properly
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test_table (id INTEGER)")
                # Force an error
                cursor.execute("INVALID SQL")
        except:
            pass  # Expected to fail
        
        # Verify that the table wasn't created due to rollback
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
            result = cursor.fetchone()
            # Table should not exist due to rollback
            assert result is None
    
    def test_concurrent_access(self, sample_orders_data):
        """Test concurrent database access"""
        import threading
        import time
        
        results = []
        errors = []
        
        def save_orders_worker(worker_id):
            try:
                # Modify data slightly for each worker
                worker_data = sample_orders_data.copy()
                worker_data['order_id'] = worker_data['order_id'] + f'-W{worker_id}'
                
                result = self.db_manager.save_orders(worker_data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=save_orders_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 3
        assert all(result.success for result in results)
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset"""
        # Create a larger dataset
        large_data = pd.DataFrame({
            'order_id': [f'ORD-2024-{i:05d}' for i in range(1, 1001)],
            'customer_name': [f'Customer {i}' for i in range(1, 1001)],
            'product': ['Test Product'] * 1000,
            'quantity': [1] * 1000,
            'price': [99.99] * 1000,
            'order_date': ['2024-01-15'] * 1000
        })
        
        start_time = datetime.now()
        result = self.db_manager.save_orders(large_data)
        end_time = datetime.now()
        
        assert result.success is True
        assert result.records_affected == 1000
        
        # Should complete in reasonable time (less than 5 seconds)
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 5.0
    
    @patch('src.storage.database_manager.logger')
    def test_logging_calls(self, mock_logger, sample_orders_data):
        """Test that appropriate logging calls are made"""
        result = self.db_manager.save_orders(sample_orders_data)
        
        # Check that info logging was called
        mock_logger.info.assert_called()
        
        # Check for specific log messages
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Saving" in msg and "orders" in msg for msg in log_calls)
        assert any("Successfully saved" in msg for msg in log_calls)
    
    def test_database_initialization_missing_tables(self):
        """Test database initialization when tables are missing"""
        # Create a fresh database manager
        fresh_db_path = tempfile.mktemp(suffix='.db')
        fresh_manager = DatabaseManager(fresh_db_path)
        
        # The manager should handle missing tables gracefully
        assert fresh_manager is not None
        
        # Cleanup
        if os.path.exists(fresh_db_path):
            os.remove(fresh_db_path)
    
    def test_sql_injection_protection(self, sample_orders_data):
        """Test protection against SQL injection"""
        # Save some data first
        self.db_manager.save_orders(sample_orders_data)
        
        # Try SQL injection in filter
        malicious_filter = {"order_id": "'; DROP TABLE orders; --"}
        result = self.db_manager.get_orders(filters=malicious_filter)
        
        # Should not cause an error and should not drop the table
        assert result.success is True
        
        # Verify table still exists
        count_result = self.db_manager.execute_custom_query("SELECT COUNT(*) FROM orders")
        assert count_result.success is True
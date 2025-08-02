"""
Database management module for data storage operations
Handles SQLite database operations, CRUD operations, and data persistence
"""

import os
import sqlite3
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import json

from ..utils.config import config
from ..utils.helpers import ensure_directory_exists, safe_divide
from ..utils.constants import DatabaseTable, ProcessingStatus

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DatabaseResult:
    """Database operation result container"""
    success: bool
    operation: str
    records_affected: int
    execution_time: float
    error_message: Optional[str] = None
    data: Optional[pd.DataFrame] = None

class DatabaseManager:
    """Comprehensive database manager for data storage operations"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            db_path (str, optional): Path to database file
        """
        self.db_path = db_path or config.database.path
        self.timeout = config.database.timeout
        
        # Ensure database directory exists
        ensure_directory_exists(os.path.dirname(self.db_path))
        
        # Initialize database if it doesn't exist
        self._initialize_database()
        
        logger.info(f"Database manager initialized: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _initialize_database(self):
        """Initialize database with required tables if they don't exist"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}
                
                required_tables = {
                    DatabaseTable.ORDERS,
                    DatabaseTable.CUSTOMERS,
                    DatabaseTable.PRODUCTS,
                    DatabaseTable.PIPELINE_RUNS,
                    DatabaseTable.DATA_QUALITY_METRICS,
                    DatabaseTable.PROCESSING_LOGS
                }
                
                missing_tables = required_tables - existing_tables
                if missing_tables:
                    logger.warning(f"Missing database tables: {missing_tables}")
                    logger.info("Please run 'python scripts/setup_database.py' to initialize the database")
                
        except Exception as e:
            logger.error(f"Error checking database initialization: {e}")
    
    def save_orders(self, data: pd.DataFrame, batch_size: int = 1000) -> DatabaseResult:
        """
        Save order data to database
        
        Args:
            data (pd.DataFrame): Order data to save
            batch_size (int): Batch size for bulk operations
            
        Returns:
            DatabaseResult: Operation result
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Saving {len(data)} orders to database")
            
            with self.get_connection() as conn:
                # Prepare data for insertion
                orders_data = self._prepare_orders_data(data)
                
                # Insert in batches
                total_inserted = 0
                for i in range(0, len(orders_data), batch_size):
                    batch = orders_data[i:i + batch_size]
                    
                    # Insert batch
                    cursor = conn.cursor()
                    cursor.executemany('''
                        INSERT OR REPLACE INTO orders (
                            order_id, customer_name, product, quantity, price, order_date,
                            source, store_location, customer_email, product_category,
                            discount, total_amount, processed_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', batch)
                    
                    total_inserted += cursor.rowcount
                    conn.commit()
                    
                    logger.debug(f"Inserted batch {i//batch_size + 1}: {len(batch)} records")
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = DatabaseResult(
                    success=True,
                    operation='save_orders',
                    records_affected=total_inserted,
                    execution_time=execution_time
                )
                
                logger.info(f"Successfully saved {total_inserted} orders ({execution_time:.2f}s)")
                return result
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error saving orders: {e}")
            return DatabaseResult(
                success=False,
                operation='save_orders',
                records_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _prepare_orders_data(self, data: pd.DataFrame) -> List[Tuple]:
        """Prepare order data for database insertion"""
        prepared_data = []
        current_time = datetime.now().isoformat()
        
        for _, row in data.iterrows():
            order_tuple = (
                row.get('order_id', ''),
                row.get('customer_name', ''),
                row.get('product', ''),
                row.get('quantity', 1),
                row.get('price', 0.0),
                row.get('order_date', current_time),
                row.get('source', 'unknown'),
                row.get('store_location', None),
                row.get('customer_email', None),
                row.get('product_category', 'Electronics'),
                row.get('discount', 0.0),
                row.get('total_amount', 0.0),
                row.get('processed_at', current_time),
                current_time,  # created_at
                current_time   # updated_at
            )
            prepared_data.append(order_tuple)
        
        return prepared_data
    
    def get_orders(self, filters: Optional[Dict[str, Any]] = None, 
                   limit: Optional[int] = None, 
                   offset: Optional[int] = None) -> DatabaseResult:
        """
        Retrieve orders from database
        
        Args:
            filters (Dict, optional): Filter conditions
            limit (int, optional): Maximum number of records
            offset (int, optional): Number of records to skip
            
        Returns:
            DatabaseResult: Query result with data
        """
        start_time = datetime.now()
        
        try:
            with self.get_connection() as conn:
                # Build query
                query = "SELECT * FROM orders"
                params = []
                
                # Add filters
                if filters:
                    conditions = []
                    for field, value in filters.items():
                        if isinstance(value, list):
                            placeholders = ','.join(['?' for _ in value])
                            conditions.append(f"{field} IN ({placeholders})")
                            params.extend(value)
                        else:
                            conditions.append(f"{field} = ?")
                            params.append(value)
                    
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                # Add ordering
                query += " ORDER BY created_at DESC"
                
                # Add limit and offset
                if limit:
                    query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"
                
                # Execute query
                df = pd.read_sql_query(query, conn, params=params)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = DatabaseResult(
                    success=True,
                    operation='get_orders',
                    records_affected=len(df),
                    execution_time=execution_time,
                    data=df
                )
                
                logger.info(f"Retrieved {len(df)} orders ({execution_time:.2f}s)")
                return result
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error retrieving orders: {e}")
            return DatabaseResult(
                success=False,
                operation='get_orders',
                records_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def save_pipeline_run(self, run_data: Dict[str, Any]) -> DatabaseResult:
        """
        Save pipeline run information
        
        Args:
            run_data (Dict): Pipeline run data
            
        Returns:
            DatabaseResult: Save result
        """
        start_time = datetime.now()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO pipeline_runs (
                        run_id, pipeline_name, start_time, end_time, status,
                        records_processed, records_failed, error_message, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_data.get('run_id'),
                    run_data.get('pipeline_name'),
                    run_data.get('start_time'),
                    run_data.get('end_time'),
                    run_data.get('status'),
                    run_data.get('records_processed', 0),
                    run_data.get('records_failed', 0),
                    run_data.get('error_message'),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = DatabaseResult(
                    success=True,
                    operation='save_pipeline_run',
                    records_affected=cursor.rowcount,
                    execution_time=execution_time
                )
                
                logger.info(f"Saved pipeline run: {run_data.get('run_id')}")
                return result
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error saving pipeline run: {e}")
            return DatabaseResult(
                success=False,
                operation='save_pipeline_run',
                records_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def save_data_quality_metrics(self, metrics_data: List[Dict[str, Any]]) -> DatabaseResult:
        """
        Save data quality metrics
        
        Args:
            metrics_data (List[Dict]): List of metrics to save
            
        Returns:
            DatabaseResult: Save result
        """
        start_time = datetime.now()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare data for insertion
                metrics_tuples = []
                for metric in metrics_data:
                    metrics_tuples.append((
                        metric.get('run_id'),
                        metric.get('metric_name'),
                        metric.get('metric_value'),
                        metric.get('metric_type'),
                        metric.get('source_table'),
                        metric.get('measured_at', datetime.now().isoformat()),
                        datetime.now().isoformat()  # created_at
                    ))
                
                cursor.executemany('''
                    INSERT INTO data_quality_metrics (
                        run_id, metric_name, metric_value, metric_type,
                        source_table, measured_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', metrics_tuples)
                
                conn.commit()
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = DatabaseResult(
                    success=True,
                    operation='save_data_quality_metrics',
                    records_affected=len(metrics_tuples),
                    execution_time=execution_time
                )
                
                logger.info(f"Saved {len(metrics_tuples)} data quality metrics")
                return result
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error saving data quality metrics: {e}")
            return DatabaseResult(
                success=False,
                operation='save_data_quality_metrics',
                records_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def get_pipeline_runs(self, limit: int = 50) -> DatabaseResult:
        """
        Get recent pipeline runs
        
        Args:
            limit (int): Maximum number of runs to retrieve
            
        Returns:
            DatabaseResult: Query result
        """
        start_time = datetime.now()
        
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM pipeline_runs 
                    ORDER BY start_time DESC 
                    LIMIT ?
                """
                
                df = pd.read_sql_query(query, conn, params=[limit])
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = DatabaseResult(
                    success=True,
                    operation='get_pipeline_runs',
                    records_affected=len(df),
                    execution_time=execution_time,
                    data=df
                )
                
                logger.info(f"Retrieved {len(df)} pipeline runs")
                return result
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error retrieving pipeline runs: {e}")
            return DatabaseResult(
                success=False,
                operation='get_pipeline_runs',
                records_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def get_database_stats(self) -> DatabaseResult:
        """
        Get database statistics
        
        Returns:
            DatabaseResult: Statistics data
        """
        start_time = datetime.now()
        
        try:
            with self.get_connection() as conn:
                stats = {}
                
                # Table counts
                tables = [
                    DatabaseTable.ORDERS,
                    DatabaseTable.CUSTOMERS,
                    DatabaseTable.PRODUCTS,
                    DatabaseTable.PIPELINE_RUNS,
                    DatabaseTable.DATA_QUALITY_METRICS,
                    DatabaseTable.PROCESSING_LOGS
                ]
                
                for table in tables:
                    try:
                        cursor = conn.cursor()
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        stats[f"{table}_count"] = count
                    except sqlite3.OperationalError:
                        stats[f"{table}_count"] = 0
                
                # Database size
                cursor = conn.cursor()
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                stats['database_size_bytes'] = db_size
                stats['database_size_mb'] = db_size / (1024 * 1024)
                
                # Recent activity
                try:
                    cursor.execute("""
                        SELECT COUNT(*) FROM orders 
                        WHERE created_at >= datetime('now', '-24 hours')
                    """)
                    stats['orders_last_24h'] = cursor.fetchone()[0]
                except:
                    stats['orders_last_24h'] = 0
                
                # Convert to DataFrame for consistency
                stats_df = pd.DataFrame([stats])
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = DatabaseResult(
                    success=True,
                    operation='get_database_stats',
                    records_affected=1,
                    execution_time=execution_time,
                    data=stats_df
                )
                
                logger.info("Retrieved database statistics")
                return result
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error retrieving database stats: {e}")
            return DatabaseResult(
                success=False,
                operation='get_database_stats',
                records_affected=0,
                execution_time=execution_time,
                error_message=str(e)
            )

if __name__ == "__main__":
    # Test database manager
    import pandas as pd
    import os
    
    # Create test data
    test_data = pd.DataFrame([
        {
            'order_id': 'ORD-2024-001',
            'customer_name': 'John Doe',
            'product': 'iPhone 15',
            'quantity': 1,
            'price': 999.99,
            'order_date': '2024-01-15',
            'source': 'website',
            'customer_email': 'john@example.com',
            'discount': 0.0,
            'total_amount': 999.99
        }
    ])
    
    # Test database manager
    db_manager = DatabaseManager("test_orders.db")
    
    # Test save orders
    print("Testing save orders...")
    save_result = db_manager.save_orders(test_data)
    print(f"Save result: {save_result.success}, Records: {save_result.records_affected}")
    
    # Test get orders
    print("\nTesting get orders...")
    get_result = db_manager.get_orders()
    print(f"Get result: {get_result.success}, Records: {get_result.records_affected}")
    
    # Test database stats
    print("\nTesting database stats...")
    stats_result = db_manager.get_database_stats()
    print(f"Stats result: {stats_result.success}")
    
    # Cleanup test database
    try:
        os.remove("test_orders.db")
        print("\nTest database cleaned up")
    except:
        pass
    
    print("Database manager test completed!")
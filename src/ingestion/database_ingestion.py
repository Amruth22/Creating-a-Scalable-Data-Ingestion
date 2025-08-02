"""
Database ingestion module for extracting data from external databases
Supports multiple database types with connection pooling and incremental loading
"""

import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os
from contextlib import contextmanager
import hashlib
import time

from ..utils.config import config
from ..utils.helpers import safe_divide, ensure_directory_exists
from ..utils.constants import DataSourceType, ProcessingStatus

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLSERVER = "sqlserver"
    ORACLE = "oracle"

class IncrementalMode(Enum):
    """Incremental loading modes"""
    FULL = "full"  # Full table reload
    TIMESTAMP = "timestamp"  # Based on timestamp column
    ID = "id"  # Based on ID column
    HASH = "hash"  # Based on data hash
    CUSTOM = "custom"  # Custom SQL condition

@dataclass
class DatabaseConnection:
    """Database connection configuration"""
    name: str
    db_type: DatabaseType
    host: str
    port: int
    database: str
    username: str
    password: str
    connection_string: Optional[str] = None
    pool_size: int = 5
    timeout: int = 30
    ssl_enabled: bool = False
    additional_params: Dict[str, Any] = None

@dataclass
class DataSource:
    """Database data source configuration"""
    source_id: str
    name: str
    connection_name: str
    query: str
    incremental_mode: IncrementalMode
    incremental_column: Optional[str] = None
    batch_size: int = 1000
    enabled: bool = True
    schedule_interval: int = 3600  # seconds
    last_extracted: Optional[str] = None
    last_value: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class ExtractionResult:
    """Database extraction result container"""
    success: bool
    source_id: str
    records_extracted: int
    extraction_time: float
    data: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    last_value: Optional[str] = None
    metadata: Dict[str, Any] = None

class DatabaseIngestion:
    """Comprehensive database ingestion with multi-database support"""
    
    def __init__(self):
        """Initialize database ingestion"""
        self.connections: Dict[str, DatabaseConnection] = {}
        self.data_sources: Dict[str, DataSource] = {}
        self.connection_pools: Dict[str, Any] = {}
        
        # Configuration files
        self.connections_file = "data/database_connections.json"
        self.sources_file = "data/database_sources.json"
        self.state_file = "data/ingestion_state.json"
        
        # Ensure data directory exists
        ensure_directory_exists("data")
        
        # Load configurations
        self._load_connections()
        self._load_data_sources()
        self._load_state()
        
        logger.info("Database ingestion initialized")
    
    def add_connection(self, connection: DatabaseConnection) -> bool:
        """
        Add a database connection configuration
        
        Args:
            connection (DatabaseConnection): Database connection config
            
        Returns:
            bool: True if added successfully
        """
        try:
            # Validate connection
            if self._test_connection(connection):
                self.connections[connection.name] = connection
                self._save_connections()
                logger.info(f"Added database connection: {connection.name}")
                return True
            else:
                logger.error(f"Failed to validate connection: {connection.name}")
                return False
        
        except Exception as e:
            logger.error(f"Error adding connection {connection.name}: {e}")
            return False
    
    def add_sqlite_connection(self, name: str, database_path: str, **kwargs) -> bool:
        """
        Add SQLite database connection
        
        Args:
            name (str): Connection name
            database_path (str): Path to SQLite database
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if added successfully
        """
        connection = DatabaseConnection(
            name=name,
            db_type=DatabaseType.SQLITE,
            host="localhost",
            port=0,
            database=database_path,
            username="",
            password="",
            **kwargs
        )
        return self.add_connection(connection)
    
    def add_mysql_connection(self, name: str, host: str, database: str, 
                           username: str, password: str, port: int = 3306, **kwargs) -> bool:
        """
        Add MySQL database connection
        
        Args:
            name (str): Connection name
            host (str): Database host
            database (str): Database name
            username (str): Username
            password (str): Password
            port (int): Port number
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if added successfully
        """
        connection = DatabaseConnection(
            name=name,
            db_type=DatabaseType.MYSQL,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            **kwargs
        )
        return self.add_connection(connection)
    
    def add_postgresql_connection(self, name: str, host: str, database: str,
                                username: str, password: str, port: int = 5432, **kwargs) -> bool:
        """
        Add PostgreSQL database connection
        
        Args:
            name (str): Connection name
            host (str): Database host
            database (str): Database name
            username (str): Username
            password (str): Password
            port (int): Port number
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if added successfully
        """
        connection = DatabaseConnection(
            name=name,
            db_type=DatabaseType.POSTGRESQL,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            **kwargs
        )
        return self.add_connection(connection)
    
    def add_data_source(self, source: DataSource) -> bool:
        """
        Add a data source configuration
        
        Args:
            source (DataSource): Data source configuration
            
        Returns:
            bool: True if added successfully
        """
        try:
            # Validate that connection exists
            if source.connection_name not in self.connections:
                logger.error(f"Connection not found: {source.connection_name}")
                return False
            
            # Validate query
            if not self._validate_query(source):
                logger.error(f"Invalid query for source: {source.source_id}")
                return False
            
            self.data_sources[source.source_id] = source
            self._save_data_sources()
            logger.info(f"Added data source: {source.source_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding data source {source.source_id}: {e}")
            return False
    
    def create_orders_source(self, source_id: str, connection_name: str,
                           table_name: str = "orders", 
                           incremental_column: str = "created_at",
                           **kwargs) -> bool:
        """
        Create a standard orders data source
        
        Args:
            source_id (str): Unique source identifier
            connection_name (str): Database connection name
            table_name (str): Orders table name
            incremental_column (str): Column for incremental loading
            **kwargs: Additional source parameters
            
        Returns:
            bool: True if created successfully
        """
        # Standard orders query
        query = f"""
        SELECT 
            order_id,
            customer_name,
            customer_email,
            product,
            product_category,
            quantity,
            price,
            discount,
            total_amount,
            order_date,
            store_location,
            source,
            created_at,
            updated_at
        FROM {table_name}
        """
        
        # Add incremental condition placeholder
        if incremental_column:
            query += f" WHERE {incremental_column} > {{last_value}}"
        
        query += f" ORDER BY {incremental_column or 'order_id'}"
        
        source = DataSource(
            source_id=source_id,
            name=f"Orders from {table_name}",
            connection_name=connection_name,
            query=query,
            incremental_mode=IncrementalMode.TIMESTAMP if incremental_column else IncrementalMode.FULL,
            incremental_column=incremental_column,
            **kwargs
        )
        
        return self.add_data_source(source)
    
    def create_customers_source(self, source_id: str, connection_name: str,
                              table_name: str = "customers", **kwargs) -> bool:
        """
        Create a standard customers data source
        
        Args:
            source_id (str): Unique source identifier
            connection_name (str): Database connection name
            table_name (str): Customers table name
            **kwargs: Additional source parameters
            
        Returns:
            bool: True if created successfully
        """
        query = f"""
        SELECT 
            customer_id,
            customer_name,
            email,
            phone,
            address,
            city,
            state,
            country,
            postal_code,
            registration_date,
            last_order_date,
            total_orders,
            total_spent,
            customer_segment,
            created_at,
            updated_at
        FROM {table_name}
        ORDER BY customer_id
        """
        
        source = DataSource(
            source_id=source_id,
            name=f"Customers from {table_name}",
            connection_name=connection_name,
            query=query,
            incremental_mode=IncrementalMode.FULL,  # Usually full reload for customers
            **kwargs
        )
        
        return self.add_data_source(source)
    
    def extract_data(self, source_id: str, limit: Optional[int] = None) -> ExtractionResult:
        """
        Extract data from a specific source
        
        Args:
            source_id (str): Data source identifier
            limit (int, optional): Maximum number of records to extract
            
        Returns:
            ExtractionResult: Extraction results
        """
        start_time = datetime.now()
        
        try:
            # Get data source
            source = self.data_sources.get(source_id)
            if not source:
                return ExtractionResult(
                    success=False,
                    source_id=source_id,
                    records_extracted=0,
                    extraction_time=0,
                    error_message=f"Data source not found: {source_id}"
                )
            
            if not source.enabled:
                return ExtractionResult(
                    success=False,
                    source_id=source_id,
                    records_extracted=0,
                    extraction_time=0,
                    error_message=f"Data source disabled: {source_id}"
                )
            
            logger.info(f"🔄 Extracting data from source: {source.name}")
            
            # Get connection
            connection = self.connections.get(source.connection_name)
            if not connection:
                return ExtractionResult(
                    success=False,
                    source_id=source_id,
                    records_extracted=0,
                    extraction_time=0,
                    error_message=f"Connection not found: {source.connection_name}"
                )
            
            # Build query with incremental logic
            query = self._build_incremental_query(source)
            
            # Add limit if specified
            if limit:
                query += f" LIMIT {limit}"
            
            logger.debug(f"Executing query: {query}")
            
            # Execute query
            with self._get_connection(connection) as conn:
                data = pd.read_sql_query(query, conn)
            
            # Add metadata columns
            data['source_id'] = source_id
            data['source_type'] = DataSourceType.DATABASE.value
            data['extracted_at'] = datetime.now().isoformat()
            data['connection_name'] = source.connection_name
            
            # Calculate new last_value for incremental loading
            new_last_value = self._calculate_last_value(source, data)
            
            # Update extraction state
            source.last_extracted = datetime.now().isoformat()
            if new_last_value:
                source.last_value = str(new_last_value)
            
            self._save_state()
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            result = ExtractionResult(
                success=True,
                source_id=source_id,
                records_extracted=len(data),
                extraction_time=extraction_time,
                data=data,
                last_value=str(new_last_value) if new_last_value else None,
                metadata={
                    'query': query,
                    'connection': source.connection_name,
                    'incremental_mode': source.incremental_mode.value
                }
            )
            
            logger.info(f"✅ Extracted {len(data)} records from {source.name} ({extraction_time:.2f}s)")
            
            return result
        
        except Exception as e:
            extraction_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Error extracting from {source_id}: {e}")
            
            return ExtractionResult(
                success=False,
                source_id=source_id,
                records_extracted=0,
                extraction_time=extraction_time,
                error_message=str(e)
            )
    
    def extract_all_sources(self, enabled_only: bool = True) -> Dict[str, ExtractionResult]:
        """
        Extract data from all configured sources
        
        Args:
            enabled_only (bool): Only extract from enabled sources
            
        Returns:
            Dict[str, ExtractionResult]: Results for each source
        """
        results = {}
        sources_to_extract = [
            source for source in self.data_sources.values()
            if not enabled_only or source.enabled
        ]
        
        logger.info(f"🔄 Extracting data from {len(sources_to_extract)} sources")
        
        for source in sources_to_extract:
            try:
                result = self.extract_data(source.source_id)
                results[source.source_id] = result
                
                # Small delay between extractions to avoid overwhelming the database
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error extracting from {source.source_id}: {e}")
                results[source.source_id] = ExtractionResult(
                    success=False,
                    source_id=source.source_id,
                    records_extracted=0,
                    extraction_time=0,
                    error_message=str(e)
                )
        
        # Summary
        total_records = sum(r.records_extracted for r in results.values())
        successful_sources = sum(1 for r in results.values() if r.success)
        
        logger.info(f"📊 Extraction summary: {successful_sources}/{len(sources_to_extract)} sources successful, {total_records:,} total records")
        
        return results
    
    def get_combined_data(self, source_ids: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Get combined data from multiple sources
        
        Args:
            source_ids (List[str], optional): Specific sources to combine
            
        Returns:
            pd.DataFrame: Combined data or None if no data
        """
        try:
            # Extract data from specified sources or all sources
            if source_ids:
                results = {sid: self.extract_data(sid) for sid in source_ids}
            else:
                results = self.extract_all_sources()
            
            # Collect successful extractions
            successful_data = [
                result.data for result in results.values()
                if result.success and result.data is not None and not result.data.empty
            ]
            
            if not successful_data:
                logger.warning("No successful data extractions found")
                return None
            
            # Combine all DataFrames
            combined_data = pd.concat(successful_data, ignore_index=True)
            
            # Remove duplicates if order_id exists
            if 'order_id' in combined_data.columns:
                before_dedup = len(combined_data)
                combined_data = combined_data.drop_duplicates(subset=['order_id'], keep='first')
                after_dedup = len(combined_data)
                if before_dedup != after_dedup:
                    logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
            
            logger.info(f"Combined data from {len(successful_data)} sources: {len(combined_data)} total records")
            
            return combined_data
        
        except Exception as e:
            logger.error(f"Error combining data: {e}")
            return None
    
    def _build_incremental_query(self, source: DataSource) -> str:
        """Build query with incremental loading logic"""
        query = source.query
        
        if source.incremental_mode == IncrementalMode.FULL:
            return query
        
        # Replace placeholder with actual last value
        if "{last_value}" in query and source.last_value:
            if source.incremental_mode == IncrementalMode.TIMESTAMP:
                # For timestamp columns, use proper date formatting
                last_value = f"'{source.last_value}'"
            elif source.incremental_mode == IncrementalMode.ID:
                # For ID columns, use numeric value
                last_value = source.last_value
            else:
                last_value = f"'{source.last_value}'"
            
            query = query.replace("{last_value}", last_value)
        elif "{last_value}" in query:
            # First time extraction - use a default value
            if source.incremental_mode == IncrementalMode.TIMESTAMP:
                # Start from 30 days ago
                default_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
                query = query.replace("{last_value}", f"'{default_date}'")
            elif source.incremental_mode == IncrementalMode.ID:
                query = query.replace("{last_value}", "0")
            else:
                query = query.replace("{last_value}", "''")
        
        return query
    
    def _calculate_last_value(self, source: DataSource, data: pd.DataFrame) -> Optional[Any]:
        """Calculate the last value for incremental loading"""
        if data.empty or source.incremental_mode == IncrementalMode.FULL:
            return None
        
        if not source.incremental_column or source.incremental_column not in data.columns:
            return None
        
        # Get the maximum value from the incremental column
        max_value = data[source.incremental_column].max()
        
        # Convert to string for storage
        if pd.isna(max_value):
            return None
        
        if source.incremental_mode == IncrementalMode.TIMESTAMP:
            if isinstance(max_value, str):
                return max_value
            else:
                return max_value.isoformat() if hasattr(max_value, 'isoformat') else str(max_value)
        
        return max_value
    
    def _test_connection(self, connection: DatabaseConnection) -> bool:
        """Test database connection"""
        try:
            with self._get_connection(connection) as conn:
                if connection.db_type == DatabaseType.SQLITE:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                else:
                    # For other databases, use pandas to test
                    pd.read_sql_query("SELECT 1 as test", conn)
            
            logger.info(f"✅ Connection test successful: {connection.name}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Connection test failed for {connection.name}: {e}")
            return False
    
    def _validate_query(self, source: DataSource) -> bool:
        """Validate data source query"""
        try:
            connection = self.connections.get(source.connection_name)
            if not connection:
                return False
            
            # Test query with LIMIT 1
            test_query = f"SELECT * FROM ({source.query.replace('{last_value}', '0')}) AS test_query LIMIT 1"
            
            with self._get_connection(connection) as conn:
                pd.read_sql_query(test_query, conn)
            
            logger.info(f"✅ Query validation successful: {source.source_id}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Query validation failed for {source.source_id}: {e}")
            return False
    
    @contextmanager
    def _get_connection(self, connection: DatabaseConnection):
        """Get database connection with proper cleanup"""
        conn = None
        try:
            if connection.db_type == DatabaseType.SQLITE:
                conn = sqlite3.connect(
                    connection.database,
                    timeout=connection.timeout,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
            
            elif connection.db_type == DatabaseType.MYSQL:
                try:
                    import mysql.connector
                    conn = mysql.connector.connect(
                        host=connection.host,
                        port=connection.port,
                        database=connection.database,
                        user=connection.username,
                        password=connection.password,
                        connection_timeout=connection.timeout
                    )
                except ImportError:
                    raise ImportError("mysql-connector-python package required for MySQL connections")
            
            elif connection.db_type == DatabaseType.POSTGRESQL:
                try:
                    import psycopg2
                    conn = psycopg2.connect(
                        host=connection.host,
                        port=connection.port,
                        database=connection.database,
                        user=connection.username,
                        password=connection.password,
                        connect_timeout=connection.timeout
                    )
                except ImportError:
                    raise ImportError("psycopg2 package required for PostgreSQL connections")
            
            else:
                raise ValueError(f"Unsupported database type: {connection.db_type}")
            
            yield conn
        
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        
        finally:
            if conn:
                conn.close()
    
    def _load_connections(self):
        """Load database connections from file"""
        try:
            if os.path.exists(self.connections_file):
                with open(self.connections_file, 'r') as f:
                    connections_data = json.load(f)
                
                for conn_data in connections_data:
                    connection = DatabaseConnection(
                        name=conn_data['name'],
                        db_type=DatabaseType(conn_data['db_type']),
                        host=conn_data['host'],
                        port=conn_data['port'],
                        database=conn_data['database'],
                        username=conn_data['username'],
                        password=conn_data['password'],
                        connection_string=conn_data.get('connection_string'),
                        pool_size=conn_data.get('pool_size', 5),
                        timeout=conn_data.get('timeout', 30),
                        ssl_enabled=conn_data.get('ssl_enabled', False),
                        additional_params=conn_data.get('additional_params', {})
                    )
                    self.connections[connection.name] = connection
                
                logger.info(f"Loaded {len(self.connections)} database connections")
        
        except Exception as e:
            logger.error(f"Error loading connections: {e}")
    
    def _save_connections(self):
        """Save database connections to file"""
        try:
            connections_data = []
            for connection in self.connections.values():
                connections_data.append({
                    'name': connection.name,
                    'db_type': connection.db_type.value,
                    'host': connection.host,
                    'port': connection.port,
                    'database': connection.database,
                    'username': connection.username,
                    'password': connection.password,  # In production, encrypt this
                    'connection_string': connection.connection_string,
                    'pool_size': connection.pool_size,
                    'timeout': connection.timeout,
                    'ssl_enabled': connection.ssl_enabled,
                    'additional_params': connection.additional_params
                })
            
            with open(self.connections_file, 'w') as f:
                json.dump(connections_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving connections: {e}")
    
    def _load_data_sources(self):
        """Load data sources from file"""
        try:
            if os.path.exists(self.sources_file):
                with open(self.sources_file, 'r') as f:
                    sources_data = json.load(f)
                
                for source_data in sources_data:
                    source = DataSource(
                        source_id=source_data['source_id'],
                        name=source_data['name'],
                        connection_name=source_data['connection_name'],
                        query=source_data['query'],
                        incremental_mode=IncrementalMode(source_data['incremental_mode']),
                        incremental_column=source_data.get('incremental_column'),
                        batch_size=source_data.get('batch_size', 1000),
                        enabled=source_data.get('enabled', True),
                        schedule_interval=source_data.get('schedule_interval', 3600),
                        last_extracted=source_data.get('last_extracted'),
                        last_value=source_data.get('last_value'),
                        metadata=source_data.get('metadata', {})
                    )
                    self.data_sources[source.source_id] = source
                
                logger.info(f"Loaded {len(self.data_sources)} data sources")
        
        except Exception as e:
            logger.error(f"Error loading data sources: {e}")
    
    def _save_data_sources(self):
        """Save data sources to file"""
        try:
            sources_data = []
            for source in self.data_sources.values():
                sources_data.append({
                    'source_id': source.source_id,
                    'name': source.name,
                    'connection_name': source.connection_name,
                    'query': source.query,
                    'incremental_mode': source.incremental_mode.value,
                    'incremental_column': source.incremental_column,
                    'batch_size': source.batch_size,
                    'enabled': source.enabled,
                    'schedule_interval': source.schedule_interval,
                    'last_extracted': source.last_extracted,
                    'last_value': source.last_value,
                    'metadata': source.metadata
                })
            
            with open(self.sources_file, 'w') as f:
                json.dump(sources_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving data sources: {e}")
    
    def _load_state(self):
        """Load ingestion state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Update data sources with saved state
                for source_id, state in state_data.items():
                    if source_id in self.data_sources:
                        source = self.data_sources[source_id]
                        source.last_extracted = state.get('last_extracted')
                        source.last_value = state.get('last_value')
                
                logger.info("Loaded ingestion state")
        
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save ingestion state to file"""
        try:
            state_data = {}
            for source in self.data_sources.values():
                state_data[source.source_id] = {
                    'last_extracted': source.last_extracted,
                    'last_value': source.last_value
                }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def get_ingestion_status(self) -> Dict[str, Any]:
        """Get database ingestion status"""
        total_connections = len(self.connections)
        total_sources = len(self.data_sources)
        enabled_sources = len([s for s in self.data_sources.values() if s.enabled])
        
        # Test connections
        healthy_connections = 0
        for connection in self.connections.values():
            if self._test_connection(connection):
                healthy_connections += 1
        
        # Recent extractions
        recent_extractions = 0
        cutoff_time = datetime.now() - timedelta(hours=24)
        for source in self.data_sources.values():
            if source.last_extracted:
                try:
                    last_extracted = datetime.fromisoformat(source.last_extracted)
                    if last_extracted > cutoff_time:
                        recent_extractions += 1
                except:
                    pass
        
        return {
            'total_connections': total_connections,
            'healthy_connections': healthy_connections,
            'total_sources': total_sources,
            'enabled_sources': enabled_sources,
            'recent_extractions_24h': recent_extractions,
            'connection_health_rate': safe_divide(healthy_connections, total_connections, 0) * 100,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_ingestion_report(self) -> str:
        """Generate comprehensive database ingestion report"""
        status = self.get_ingestion_status()
        
        report = []
        report.append("# Database Ingestion Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Total Connections**: {status['total_connections']}")
        report.append(f"- **Healthy Connections**: {status['healthy_connections']}")
        report.append(f"- **Connection Health**: {status['connection_health_rate']:.1f}%")
        report.append(f"- **Total Sources**: {status['total_sources']}")
        report.append(f"- **Enabled Sources**: {status['enabled_sources']}")
        report.append(f"- **Recent Extractions (24h)**: {status['recent_extractions_24h']}")
        report.append("")
        
        # Connections
        if self.connections:
            report.append("## Database Connections")
            for connection in self.connections.values():
                health_icon = "🟢" if self._test_connection(connection) else "🔴"
                report.append(f"### {health_icon} {connection.name}")
                report.append(f"- **Type**: {connection.db_type.value}")
                report.append(f"- **Host**: {connection.host}:{connection.port}")
                report.append(f"- **Database**: {connection.database}")
                report.append("")
        
        # Data Sources
        if self.data_sources:
            report.append("## Data Sources")
            for source in self.data_sources.values():
                status_icon = "✅" if source.enabled else "❌"
                report.append(f"### {status_icon} {source.name}")
                report.append(f"- **ID**: {source.source_id}")
                report.append(f"- **Connection**: {source.connection_name}")
                report.append(f"- **Incremental Mode**: {source.incremental_mode.value}")
                if source.incremental_column:
                    report.append(f"- **Incremental Column**: {source.incremental_column}")
                if source.last_extracted:
                    report.append(f"- **Last Extracted**: {source.last_extracted}")
                if source.last_value:
                    report.append(f"- **Last Value**: {source.last_value}")
                report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test database ingestion
    print("Testing Database Ingestion...")
    
    # Initialize ingestion
    db_ingestion = DatabaseIngestion()
    
    # Add SQLite connection (using the main orders database)
    success = db_ingestion.add_sqlite_connection(
        name="main_orders_db",
        database_path="data/orders.db"
    )
    print(f"Added SQLite connection: {success}")
    
    # Create orders data source
    success = db_ingestion.create_orders_source(
        source_id="main_orders",
        connection_name="main_orders_db",
        table_name="orders",
        incremental_column="created_at"
    )
    print(f"Created orders source: {success}")
    
    # Test extraction
    result = db_ingestion.extract_data("main_orders", limit=10)
    print(f"Extraction result: Success={result.success}, Records={result.records_extracted}")
    
    if result.data is not None:
        print(f"Sample data columns: {list(result.data.columns)}")
    
    # Get status
    status = db_ingestion.get_ingestion_status()
    print(f"Ingestion status: {status['total_connections']} connections, {status['total_sources']} sources")
    
    # Generate report
    report = db_ingestion.generate_ingestion_report()
    print(f"Generated report ({len(report)} characters)")
    
    print("Database ingestion test completed!")
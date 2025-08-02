"""
Comprehensive metrics collection and analysis system for data pipelines
Provides performance monitoring, data quality tracking, and system health metrics
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
import sqlite3
from contextlib import contextmanager

from ..utils.helpers import ensure_directory_exists
from ..utils.constants import MetricName
from .logger import get_logger

@dataclass
class MetricPoint:
    """Individual metric data point"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, timer

@dataclass
class MetricSummary:
    """Statistical summary of metric values"""
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    std_dev: float
    percentile_95: float
    percentile_99: float

class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self, db_path: str = "data/metrics.db", 
                 retention_days: int = 30,
                 collection_interval: int = 60):
        self.db_path = db_path
        self.retention_days = retention_days
        self.collection_interval = collection_interval
        self.logger = get_logger("MetricsCollector")
        
        # In-memory metrics storage
        self.metrics_buffer = deque(maxlen=10000)
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)
        
        # System metrics collection
        self.system_metrics_enabled = True
        self.system_metrics_thread = None
        self.stop_collection = threading.Event()
        
        # Initialize database
        self._init_database()
        
        # Start system metrics collection
        self.start_system_metrics_collection()
        
        self.logger.info("Metrics collector initialized", 
                        db_path=db_path, 
                        retention_days=retention_days)
    
    def _init_database(self):
        """Initialize metrics database"""
        ensure_directory_exists(os.path.dirname(self.db_path))
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tags TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_name_timestamp (name, timestamp),
                    INDEX idx_timestamp (timestamp)
                )
            ''')
            
            # Metric summaries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metric_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    time_window TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    sum REAL NOT NULL,
                    min REAL NOT NULL,
                    max REAL NOT NULL,
                    mean REAL NOT NULL,
                    median REAL NOT NULL,
                    std_dev REAL NOT NULL,
                    percentile_95 REAL NOT NULL,
                    percentile_99 REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, time_window, timestamp)
                )
            ''')
            
            conn.commit()
    
    def record_metric(self, name: str, value: Union[int, float], 
                     metric_type: str = "gauge", tags: Dict[str, str] = None):
        """Record a metric value"""
        metric_point = MetricPoint(
            name=name,
            value=float(value),
            timestamp=datetime.now(),
            tags=tags or {},
            metric_type=metric_type
        )
        
        # Add to buffer
        self.metrics_buffer.append(metric_point)
        
        # Update in-memory storage based on type
        if metric_type == "counter":
            self.counters[name] += value
        elif metric_type == "gauge":
            self.gauges[name] = value
        elif metric_type == "histogram":
            self.histograms[name].append(value)
            # Keep only recent values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
        elif metric_type == "timer":
            self.timers[name].append(value)
            # Keep only recent values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
        
        # Persist to database periodically
        if len(self.metrics_buffer) >= 100:
            self._flush_metrics()
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        self.record_metric(name, value, "counter", tags)
    
    def set_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Set a gauge metric value"""
        self.record_metric(name, value, "gauge", tags)
    
    def record_histogram(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Record a histogram value"""
        self.record_metric(name, value, "histogram", tags)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer duration in seconds"""
        self.record_metric(name, duration, "timer", tags)
    
    @contextmanager
    def timer(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timer(name, duration, tags)
    
    def _flush_metrics(self):
        """Flush metrics buffer to database"""
        if not self.metrics_buffer:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                metrics_data = []
                while self.metrics_buffer:
                    metric = self.metrics_buffer.popleft()
                    metrics_data.append((
                        metric.name,
                        metric.value,
                        metric.metric_type,
                        metric.timestamp.isoformat(),
                        json.dumps(metric.tags) if metric.tags else None
                    ))
                
                cursor.executemany('''
                    INSERT INTO metrics (name, value, metric_type, timestamp, tags)
                    VALUES (?, ?, ?, ?, ?)
                ''', metrics_data)
                
                conn.commit()
                
                self.logger.debug(f"Flushed {len(metrics_data)} metrics to database")
                
        except Exception as e:
            self.logger.error(f"Error flushing metrics: {e}")
    
    def get_metric_summary(self, name: str, hours_back: int = 24) -> Optional[MetricSummary]:
        """Get statistical summary of a metric"""
        cutoff_time = (datetime.now() - timedelta(hours=hours_back)).isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT value FROM metrics 
                    WHERE name = ? AND timestamp >= ?
                    ORDER BY timestamp
                ''', (name, cutoff_time))
                
                values = [row[0] for row in cursor.fetchall()]
                
                if not values:
                    return None
                
                return self._calculate_summary(values)
                
        except Exception as e:
            self.logger.error(f"Error getting metric summary for {name}: {e}")
            return None
    
    def _calculate_summary(self, values: List[float]) -> MetricSummary:
        """Calculate statistical summary of values"""
        if not values:
            return None
        
        sorted_values = sorted(values)
        count = len(values)
        
        return MetricSummary(
            count=count,
            sum=sum(values),
            min=min(values),
            max=max(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if count > 1 else 0.0,
            percentile_95=sorted_values[int(0.95 * count)] if count > 0 else 0.0,
            percentile_99=sorted_values[int(0.99 * count)] if count > 0 else 0.0
        )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {name: self._calculate_summary(values) 
                          for name, values in self.histograms.items() if values},
            'timers': {name: self._calculate_summary(values) 
                      for name, values in self.timers.items() if values}
        }
    
    def start_system_metrics_collection(self):
        """Start collecting system metrics"""
        if self.system_metrics_thread is None or not self.system_metrics_thread.is_alive():
            self.stop_collection.clear()
            self.system_metrics_thread = threading.Thread(
                target=self._collect_system_metrics,
                daemon=True
            )
            self.system_metrics_thread.start()
            self.logger.info("Started system metrics collection")
    
    def stop_system_metrics_collection(self):
        """Stop collecting system metrics"""
        self.stop_collection.set()
        if self.system_metrics_thread:
            self.system_metrics_thread.join(timeout=5)
        self.logger.info("Stopped system metrics collection")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        while not self.stop_collection.wait(self.collection_interval):
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.set_gauge("system.cpu.usage_percent", cpu_percent)
                
                cpu_count = psutil.cpu_count()
                self.set_gauge("system.cpu.count", cpu_count)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.set_gauge("system.memory.total_bytes", memory.total)
                self.set_gauge("system.memory.available_bytes", memory.available)
                self.set_gauge("system.memory.used_bytes", memory.used)
                self.set_gauge("system.memory.usage_percent", memory.percent)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.set_gauge("system.disk.total_bytes", disk.total)
                self.set_gauge("system.disk.used_bytes", disk.used)
                self.set_gauge("system.disk.free_bytes", disk.free)
                self.set_gauge("system.disk.usage_percent", (disk.used / disk.total) * 100)
                
                # Network metrics
                network = psutil.net_io_counters()
                self.increment_counter("system.network.bytes_sent", network.bytes_sent)
                self.increment_counter("system.network.bytes_recv", network.bytes_recv)
                self.increment_counter("system.network.packets_sent", network.packets_sent)
                self.increment_counter("system.network.packets_recv", network.packets_recv)
                
                # Process metrics
                process = psutil.Process()
                self.set_gauge("process.cpu.usage_percent", process.cpu_percent())
                self.set_gauge("process.memory.rss_bytes", process.memory_info().rss)
                self.set_gauge("process.memory.vms_bytes", process.memory_info().vms)
                self.set_gauge("process.threads.count", process.num_threads())
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
    
    def cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period"""
        cutoff_time = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old metrics
                cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_time,))
                deleted_metrics = cursor.rowcount
                
                # Delete old summaries
                cursor.execute('DELETE FROM metric_summaries WHERE timestamp < ?', (cutoff_time,))
                deleted_summaries = cursor.rowcount
                
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_metrics} old metrics and {deleted_summaries} summaries")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics: {e}")
    
    def generate_hourly_summaries(self):
        """Generate hourly metric summaries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get unique metric names
                cursor.execute('SELECT DISTINCT name FROM metrics')
                metric_names = [row[0] for row in cursor.fetchall()]
                
                # Generate summaries for each metric
                for metric_name in metric_names:
                    self._generate_metric_summary(cursor, metric_name, 'hourly')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error generating hourly summaries: {e}")
    
    def _generate_metric_summary(self, cursor, metric_name: str, window: str):
        """Generate summary for a specific metric and time window"""
        if window == 'hourly':
            # Get data for the last complete hour
            end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(hours=1)
        else:
            return  # Only hourly summaries for now
        
        cursor.execute('''
            SELECT value FROM metrics 
            WHERE name = ? AND timestamp >= ? AND timestamp < ?
        ''', (metric_name, start_time.isoformat(), end_time.isoformat()))
        
        values = [row[0] for row in cursor.fetchall()]
        
        if not values:
            return
        
        summary = self._calculate_summary(values)
        
        # Insert summary
        cursor.execute('''
            INSERT OR REPLACE INTO metric_summaries 
            (name, time_window, count, sum, min, max, mean, median, std_dev, 
             percentile_95, percentile_99, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric_name, window, summary.count, summary.sum, summary.min, summary.max,
            summary.mean, summary.median, summary.std_dev, summary.percentile_95,
            summary.percentile_99, start_time.isoformat()
        ))


class PipelineMetrics:
    """Pipeline-specific metrics collection"""
    
    def __init__(self, pipeline_id: str, metrics_collector: MetricsCollector):
        self.pipeline_id = pipeline_id
        self.metrics = metrics_collector
        self.stage_timers = {}
        self.pipeline_start_time = None
    
    def start_pipeline(self):
        """Mark pipeline start"""
        self.pipeline_start_time = time.time()
        self.metrics.increment_counter("pipeline.starts", tags={"pipeline_id": self.pipeline_id})
    
    def end_pipeline(self, success: bool):
        """Mark pipeline end"""
        if self.pipeline_start_time:
            duration = time.time() - self.pipeline_start_time
            self.metrics.record_timer("pipeline.execution_time", duration, 
                                    tags={"pipeline_id": self.pipeline_id, "success": str(success)})
        
        if success:
            self.metrics.increment_counter("pipeline.successes", tags={"pipeline_id": self.pipeline_id})
        else:
            self.metrics.increment_counter("pipeline.failures", tags={"pipeline_id": self.pipeline_id})
    
    def start_stage(self, stage: str):
        """Start timing a pipeline stage"""
        self.stage_timers[stage] = time.time()
    
    def end_stage(self, stage: str, records_processed: int = 0):
        """End timing a pipeline stage"""
        if stage in self.stage_timers:
            duration = time.time() - self.stage_timers[stage]
            self.metrics.record_timer(f"pipeline.stage.{stage}.execution_time", duration,
                                    tags={"pipeline_id": self.pipeline_id})
            
            if records_processed > 0:
                self.metrics.record_histogram(f"pipeline.stage.{stage}.records_processed", 
                                            records_processed,
                                            tags={"pipeline_id": self.pipeline_id})
                
                # Calculate throughput
                throughput = records_processed / duration if duration > 0 else 0
                self.metrics.record_histogram(f"pipeline.stage.{stage}.throughput", throughput,
                                            tags={"pipeline_id": self.pipeline_id})
            
            del self.stage_timers[stage]
    
    def record_data_quality(self, total_records: int, valid_records: int, quality_score: float):
        """Record data quality metrics"""
        self.metrics.record_histogram("pipeline.data_quality.total_records", total_records,
                                     tags={"pipeline_id": self.pipeline_id})
        self.metrics.record_histogram("pipeline.data_quality.valid_records", valid_records,
                                     tags={"pipeline_id": self.pipeline_id})
        self.metrics.record_histogram("pipeline.data_quality.score", quality_score,
                                     tags={"pipeline_id": self.pipeline_id})
        
        # Record quality level
        if quality_score >= 95:
            quality_level = "excellent"
        elif quality_score >= 85:
            quality_level = "good"
        elif quality_score >= 70:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        self.metrics.increment_counter(f"pipeline.data_quality.level.{quality_level}",
                                     tags={"pipeline_id": self.pipeline_id})
    
    def record_error(self, stage: str, error_type: str):
        """Record an error occurrence"""
        self.metrics.increment_counter("pipeline.errors", 
                                     tags={"pipeline_id": self.pipeline_id, 
                                          "stage": stage, "error_type": error_type})
    
    def record_data_source(self, source: str, records: int):
        """Record data source metrics"""
        self.metrics.record_histogram(f"pipeline.data_source.{source}.records", records,
                                     tags={"pipeline_id": self.pipeline_id})
        self.metrics.increment_counter(f"pipeline.data_source.{source}.ingestions",
                                     tags={"pipeline_id": self.pipeline_id})


class MetricsReporter:
    """Generate metrics reports and dashboards"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = get_logger("MetricsReporter")
    
    def generate_pipeline_report(self, pipeline_id: str = None, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive pipeline metrics report"""
        cutoff_time = (datetime.now() - timedelta(hours=hours_back)).isoformat()
        
        try:
            with sqlite3.connect(self.metrics.db_path) as conn:
                cursor = conn.cursor()
                
                # Base query filter
                base_filter = "timestamp >= ?"
                params = [cutoff_time]
                
                if pipeline_id:
                    base_filter += " AND tags LIKE ?"
                    params.append(f'%"pipeline_id": "{pipeline_id}"%')
                
                report = {
                    'pipeline_id': pipeline_id,
                    'time_period_hours': hours_back,
                    'generated_at': datetime.now().isoformat()
                }
                
                # Pipeline execution metrics
                cursor.execute(f'''
                    SELECT COUNT(*) as executions,
                           AVG(value) as avg_duration,
                           MIN(value) as min_duration,
                           MAX(value) as max_duration
                    FROM metrics 
                    WHERE name = 'pipeline.execution_time' AND {base_filter}
                ''', params)
                
                exec_result = cursor.fetchone()
                if exec_result and exec_result[0] > 0:
                    report['execution_metrics'] = {
                        'total_executions': exec_result[0],
                        'avg_duration_seconds': round(exec_result[1], 2),
                        'min_duration_seconds': round(exec_result[2], 2),
                        'max_duration_seconds': round(exec_result[3], 2)
                    }
                
                # Success/failure rates
                cursor.execute(f'''
                    SELECT name, COUNT(*) as count
                    FROM metrics 
                    WHERE name IN ('pipeline.successes', 'pipeline.failures') AND {base_filter}
                    GROUP BY name
                ''', params)
                
                success_failure = dict(cursor.fetchall())
                total_completions = sum(success_failure.values())
                
                if total_completions > 0:
                    success_rate = (success_failure.get('pipeline.successes', 0) / total_completions) * 100
                    report['success_metrics'] = {
                        'total_completions': total_completions,
                        'successes': success_failure.get('pipeline.successes', 0),
                        'failures': success_failure.get('pipeline.failures', 0),
                        'success_rate_percent': round(success_rate, 1)
                    }
                
                # Data quality metrics
                cursor.execute(f'''
                    SELECT AVG(value) as avg_quality,
                           MIN(value) as min_quality,
                           MAX(value) as max_quality,
                           COUNT(*) as measurements
                    FROM metrics 
                    WHERE name = 'pipeline.data_quality.score' AND {base_filter}
                ''', params)
                
                quality_result = cursor.fetchone()
                if quality_result and quality_result[3] > 0:
                    report['quality_metrics'] = {
                        'avg_quality_score': round(quality_result[0], 1),
                        'min_quality_score': round(quality_result[1], 1),
                        'max_quality_score': round(quality_result[2], 1),
                        'quality_measurements': quality_result[3]
                    }
                
                # Stage performance
                cursor.execute(f'''
                    SELECT name, AVG(value) as avg_time, COUNT(*) as executions
                    FROM metrics 
                    WHERE name LIKE 'pipeline.stage.%.execution_time' AND {base_filter}
                    GROUP BY name
                    ORDER BY avg_time DESC
                ''', params)
                
                stage_results = cursor.fetchall()
                if stage_results:
                    report['stage_performance'] = {}
                    for name, avg_time, executions in stage_results:
                        stage_name = name.split('.')[2]  # Extract stage name
                        report['stage_performance'][stage_name] = {
                            'avg_execution_time_seconds': round(avg_time, 3),
                            'executions': executions
                        }
                
                # Error metrics
                cursor.execute(f'''
                    SELECT SUM(value) as total_errors
                    FROM metrics 
                    WHERE name = 'pipeline.errors' AND {base_filter}
                ''', params)
                
                error_result = cursor.fetchone()
                if error_result and error_result[0]:
                    report['error_metrics'] = {
                        'total_errors': int(error_result[0])
                    }
                
                return report
                
        except Exception as e:
            self.logger.error(f"Error generating pipeline report: {e}")
            return {'error': str(e)}
    
    def generate_system_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate system metrics report"""
        report = {
            'time_period_hours': hours_back,
            'generated_at': datetime.now().isoformat()
        }
        
        # Get current system metrics
        current_metrics = self.metrics.get_current_metrics()
        
        system_gauges = {k: v for k, v in current_metrics['gauges'].items() 
                        if k.startswith('system.')}
        
        if system_gauges:
            report['current_system_metrics'] = {
                'cpu_usage_percent': system_gauges.get('system.cpu.usage_percent', 0),
                'memory_usage_percent': system_gauges.get('system.memory.usage_percent', 0),
                'disk_usage_percent': system_gauges.get('system.disk.usage_percent', 0),
                'memory_used_gb': system_gauges.get('system.memory.used_bytes', 0) / (1024**3),
                'disk_used_gb': system_gauges.get('system.disk.used_bytes', 0) / (1024**3)
            }
        
        # Get historical summaries
        cpu_summary = self.metrics.get_metric_summary('system.cpu.usage_percent', hours_back)
        memory_summary = self.metrics.get_metric_summary('system.memory.usage_percent', hours_back)
        
        if cpu_summary:
            report['cpu_summary'] = {
                'avg_usage_percent': round(cpu_summary.mean, 1),
                'max_usage_percent': round(cpu_summary.max, 1),
                'min_usage_percent': round(cpu_summary.min, 1)
            }
        
        if memory_summary:
            report['memory_summary'] = {
                'avg_usage_percent': round(memory_summary.mean, 1),
                'max_usage_percent': round(memory_summary.max, 1),
                'min_usage_percent': round(memory_summary.min, 1)
            }
        
        return report


# Global metrics collector instance
_global_metrics_collector = None

def get_metrics_collector(db_path: str = "data/metrics.db") -> MetricsCollector:
    """Get or create global metrics collector"""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector(db_path)
    return _global_metrics_collector

def get_pipeline_metrics(pipeline_id: str, 
                        metrics_collector: MetricsCollector = None) -> PipelineMetrics:
    """Get pipeline-specific metrics collector"""
    if metrics_collector is None:
        metrics_collector = get_metrics_collector()
    return PipelineMetrics(pipeline_id, metrics_collector)


if __name__ == "__main__":
    # Test the metrics system
    print("Testing Metrics Collection System...")
    
    # Test basic metrics
    metrics = get_metrics_collector("test_metrics.db")
    
    # Test different metric types
    metrics.increment_counter("test.counter", 5)
    metrics.set_gauge("test.gauge", 42.5)
    metrics.record_histogram("test.histogram", 100)
    metrics.record_timer("test.timer", 1.5)
    
    # Test timer context manager
    with metrics.timer("test.operation"):
        time.sleep(0.1)
    
    # Test pipeline metrics
    pipeline_metrics = get_pipeline_metrics("TEST-PIPELINE-001", metrics)
    
    pipeline_metrics.start_pipeline()
    pipeline_metrics.start_stage("ingestion")
    time.sleep(0.05)
    pipeline_metrics.end_stage("ingestion", records_processed=100)
    pipeline_metrics.record_data_quality(100, 95, 95.0)
    pipeline_metrics.end_pipeline(success=True)
    
    # Flush metrics
    metrics._flush_metrics()
    
    # Test reporting
    reporter = MetricsReporter(metrics)
    pipeline_report = reporter.generate_pipeline_report("TEST-PIPELINE-001")
    system_report = reporter.generate_system_report()
    
    print("📊 Pipeline Report:")
    print(json.dumps(pipeline_report, indent=2))
    
    print("\n🖥️ System Report:")
    print(json.dumps(system_report, indent=2))
    
    # Cleanup
    metrics.stop_system_metrics_collection()
    
    print("✅ Metrics system test completed!")
    
    # Clean up test database
    import os
    try:
        os.remove("test_metrics.db")
    except:
        pass
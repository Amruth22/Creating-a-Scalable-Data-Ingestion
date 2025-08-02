"""
Structured logging system for data ingestion pipeline
Provides comprehensive logging with multiple handlers, formatters, and log levels
"""

import os
import sys
import logging
import logging.handlers
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import traceback

from ..utils.helpers import ensure_directory_exists
from ..utils.constants import LogLevel

class StructuredLogger:
    """Enhanced structured logger with multiple output formats and handlers"""
    
    def __init__(self, name: str = "DataPipeline", log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir
        self.logger = None
        self.handlers = {}
        
        # Ensure log directory exists
        ensure_directory_exists(self.log_dir)
        
        # Initialize logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with multiple handlers and formatters"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Console handler with colored output
        self._add_console_handler()
        
        # File handler for general logs
        self._add_file_handler()
        
        # Rotating file handler for error logs
        self._add_error_handler()
        
        # JSON structured log handler
        self._add_json_handler()
        
        # Performance metrics handler
        self._add_metrics_handler()
    
    def _add_console_handler(self):
        """Add console handler with colored output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Colored formatter for console
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
        self.handlers['console'] = console_handler
    
    def _add_file_handler(self):
        """Add rotating file handler for general logs"""
        log_file = os.path.join(self.log_dir, f"{self.name.lower()}.log")
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed formatter for file logs
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
        self.handlers['file'] = file_handler
    
    def _add_error_handler(self):
        """Add dedicated handler for error logs"""
        error_file = os.path.join(self.log_dir, f"{self.name.lower()}_errors.log")
        
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        # Detailed error formatter with stack traces
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d\n'
            'Function: %(funcName)s\n'
            'Message: %(message)s\n'
            '%(exc_info)s\n' + '-'*80,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        
        self.logger.addHandler(error_handler)
        self.handlers['error'] = error_handler
    
    def _add_json_handler(self):
        """Add JSON structured log handler"""
        json_file = os.path.join(self.log_dir, f"{self.name.lower()}_structured.jsonl")
        
        json_handler = logging.handlers.RotatingFileHandler(
            json_file,
            maxBytes=100*1024*1024,  # 100MB
            backupCount=5,
            encoding='utf-8'
        )
        json_handler.setLevel(logging.INFO)
        
        # JSON formatter
        json_formatter = JSONFormatter()
        json_handler.setFormatter(json_formatter)
        
        self.logger.addHandler(json_handler)
        self.handlers['json'] = json_handler
    
    def _add_metrics_handler(self):
        """Add handler for performance metrics"""
        metrics_file = os.path.join(self.log_dir, f"{self.name.lower()}_metrics.log")
        
        metrics_handler = logging.handlers.RotatingFileHandler(
            metrics_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        metrics_handler.setLevel(logging.INFO)
        
        # Metrics formatter
        metrics_formatter = logging.Formatter(
            '%(asctime)s - METRIC - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        metrics_handler.setFormatter(metrics_formatter)
        
        # Add filter to only log metrics
        metrics_handler.addFilter(MetricsFilter())
        
        self.logger.addHandler(metrics_handler)
        self.handlers['metrics'] = metrics_handler
    
    def log_structured(self, level: str, message: str, **kwargs):
        """Log structured message with additional context"""
        extra_data = {
            'timestamp': datetime.now().isoformat(),
            'logger_name': self.name,
            **kwargs
        }
        
        # Create structured message
        structured_msg = f"{message} | {json.dumps(extra_data, default=str)}"
        
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, structured_msg, extra=extra_data)
    
    def log_pipeline_event(self, event_type: str, pipeline_id: str, 
                          stage: str = None, **kwargs):
        """Log pipeline-specific events"""
        self.log_structured(
            'INFO',
            f"Pipeline Event: {event_type}",
            event_type=event_type,
            pipeline_id=pipeline_id,
            stage=stage,
            **kwargs
        )
    
    def log_performance_metric(self, metric_name: str, metric_value: Union[int, float], 
                              metric_type: str, **kwargs):
        """Log performance metrics"""
        metric_data = {
            'metric_name': metric_name,
            'metric_value': metric_value,
            'metric_type': metric_type,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        # Use special marker for metrics filtering
        self.logger.info(f"METRIC: {json.dumps(metric_data, default=str)}", 
                        extra={'is_metric': True})
    
    def log_data_quality(self, pipeline_id: str, total_records: int, 
                        valid_records: int, quality_score: float, **kwargs):
        """Log data quality metrics"""
        self.log_structured(
            'INFO',
            f"Data Quality Assessment",
            pipeline_id=pipeline_id,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=total_records - valid_records,
            quality_score=quality_score,
            **kwargs
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context and stack trace"""
        error_context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': traceback.format_exc(),
            'context': context or {}
        }
        
        self.log_structured(
            'ERROR',
            f"Error occurred: {str(error)}",
            **error_context
        )
    
    def log_stage_performance(self, pipeline_id: str, stage: str, 
                             execution_time: float, records_processed: int = 0, **kwargs):
        """Log stage performance metrics"""
        self.log_performance_metric(
            f"{stage}_execution_time",
            execution_time,
            "duration",
            pipeline_id=pipeline_id,
            stage=stage,
            records_processed=records_processed,
            **kwargs
        )
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.log_structured('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.log_structured('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.log_structured('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log_structured('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.log_structured('CRITICAL', message, **kwargs)
    
    def set_level(self, level: str):
        """Set logging level"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
    
    def add_context(self, **kwargs) -> 'LoggerContext':
        """Add context to all subsequent log messages"""
        return LoggerContext(self, **kwargs)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                              'filename', 'module', 'lineno', 'funcName', 'created',
                              'msecs', 'relativeCreated', 'thread', 'threadName',
                              'processName', 'process', 'getMessage', 'exc_info',
                              'exc_text', 'stack_info']:
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class MetricsFilter(logging.Filter):
    """Filter to only allow metrics logs"""
    
    def filter(self, record):
        return hasattr(record, 'is_metric') and record.is_metric


class LoggerContext:
    """Context manager for adding context to log messages"""
    
    def __init__(self, logger: StructuredLogger, **context):
        self.logger = logger
        self.context = context
        self.original_log_methods = {}
    
    def __enter__(self):
        # Store original methods
        self.original_log_methods = {
            'debug': self.logger.debug,
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical,
            'log_structured': self.logger.log_structured
        }
        
        # Replace methods with context-aware versions
        self.logger.debug = lambda msg, **kwargs: self.original_log_methods['debug'](msg, **{**self.context, **kwargs})
        self.logger.info = lambda msg, **kwargs: self.original_log_methods['info'](msg, **{**self.context, **kwargs})
        self.logger.warning = lambda msg, **kwargs: self.original_log_methods['warning'](msg, **{**self.context, **kwargs})
        self.logger.error = lambda msg, **kwargs: self.original_log_methods['error'](msg, **{**self.context, **kwargs})
        self.logger.critical = lambda msg, **kwargs: self.original_log_methods['critical'](msg, **{**self.context, **kwargs})
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original methods
        for method_name, method in self.original_log_methods.items():
            setattr(self.logger, method_name, method)


class PipelineLogger:
    """Specialized logger for pipeline operations"""
    
    def __init__(self, pipeline_id: str, log_dir: str = "logs"):
        self.pipeline_id = pipeline_id
        self.logger = StructuredLogger(f"Pipeline-{pipeline_id}", log_dir)
        self.stage_timers = {}
    
    def start_stage(self, stage: str):
        """Start timing a pipeline stage"""
        self.stage_timers[stage] = datetime.now()
        self.logger.log_pipeline_event(
            'stage_started',
            self.pipeline_id,
            stage=stage
        )
    
    def end_stage(self, stage: str, records_processed: int = 0, **kwargs):
        """End timing a pipeline stage"""
        if stage in self.stage_timers:
            execution_time = (datetime.now() - self.stage_timers[stage]).total_seconds()
            
            self.logger.log_stage_performance(
                self.pipeline_id,
                stage,
                execution_time,
                records_processed,
                **kwargs
            )
            
            self.logger.log_pipeline_event(
                'stage_completed',
                self.pipeline_id,
                stage=stage,
                execution_time=execution_time,
                records_processed=records_processed
            )
            
            del self.stage_timers[stage]
        else:
            self.logger.warning(f"Stage {stage} was not started or already ended")
    
    def log_pipeline_start(self, **kwargs):
        """Log pipeline start"""
        self.logger.log_pipeline_event(
            'pipeline_started',
            self.pipeline_id,
            **kwargs
        )
    
    def log_pipeline_end(self, success: bool, **kwargs):
        """Log pipeline completion"""
        event_type = 'pipeline_completed' if success else 'pipeline_failed'
        self.logger.log_pipeline_event(
            event_type,
            self.pipeline_id,
            success=success,
            **kwargs
        )
    
    def log_data_ingestion(self, source: str, records: int, **kwargs):
        """Log data ingestion event"""
        self.logger.log_pipeline_event(
            'data_ingested',
            self.pipeline_id,
            source=source,
            records_ingested=records,
            **kwargs
        )
    
    def log_validation_result(self, total_records: int, valid_records: int, 
                            quality_score: float, **kwargs):
        """Log validation results"""
        self.logger.log_data_quality(
            self.pipeline_id,
            total_records,
            valid_records,
            quality_score,
            **kwargs
        )
    
    def log_transformation(self, operation: str, records_before: int, 
                          records_after: int, **kwargs):
        """Log transformation operation"""
        self.logger.log_pipeline_event(
            'data_transformed',
            self.pipeline_id,
            operation=operation,
            records_before=records_before,
            records_after=records_after,
            **kwargs
        )
    
    def log_storage(self, destination: str, records_stored: int, **kwargs):
        """Log data storage event"""
        self.logger.log_pipeline_event(
            'data_stored',
            self.pipeline_id,
            destination=destination,
            records_stored=records_stored,
            **kwargs
        )


# Global logger instance
_global_logger = None

def get_logger(name: str = "DataPipeline", log_dir: str = "logs") -> StructuredLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger(name, log_dir)
    return _global_logger

def get_pipeline_logger(pipeline_id: str, log_dir: str = "logs") -> PipelineLogger:
    """Get pipeline-specific logger"""
    return PipelineLogger(pipeline_id, log_dir)


if __name__ == "__main__":
    # Test the logging system
    print("Testing Structured Logger...")
    
    # Test basic logger
    logger = get_logger("TestPipeline")
    
    logger.info("Testing basic logging", component="test", version="1.0")
    logger.warning("This is a warning", alert_level="medium")
    logger.error("This is an error", error_code="E001")
    
    # Test performance logging
    logger.log_performance_metric("execution_time", 45.2, "duration", stage="ingestion")
    logger.log_performance_metric("throughput", 150.5, "rate", unit="records/second")
    
    # Test pipeline logger
    pipeline_logger = get_pipeline_logger("TEST-PIPELINE-001")
    
    pipeline_logger.log_pipeline_start(data_sources=["csv", "api"])
    pipeline_logger.start_stage("ingestion")
    
    # Simulate some work
    import time
    time.sleep(0.1)
    
    pipeline_logger.end_stage("ingestion", records_processed=100)
    pipeline_logger.log_validation_result(100, 95, 95.0)
    pipeline_logger.log_pipeline_end(True, total_records=95)
    
    # Test context logging
    with logger.add_context(pipeline_id="TEST-001", user="admin"):
        logger.info("This message has context")
        logger.warning("This warning also has context")
    
    # Test error logging with context
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        logger.log_error_with_context(e, {"operation": "test", "data_size": 1000})
    
    print("✅ Logging system test completed!")
    print(f"📁 Log files created in: logs/")
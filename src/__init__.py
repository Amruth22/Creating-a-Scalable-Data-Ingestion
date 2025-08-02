"""
Data Ingestion Pipeline Package
A scalable data ingestion pipeline for processing e-commerce order data from multiple sources.
"""

__version__ = "1.0.0"
__author__ = "Data Engineering Team"
__email__ = "team@company.com"
__description__ = "A scalable data ingestion pipeline for e-commerce order processing"
__url__ = "https://github.com/Amruth22/Creating-a-Scalable-Data-Ingestion"

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__url__",
]

# Import main components for easy access
try:
    from .pipeline.pipeline_manager import PipelineManager
    from .ingestion.file_ingestion import FileIngestion
    from .ingestion.api_ingestion import APIIngestion
    from .validation.data_validator import DataValidator
    from .transformation.data_cleaner import DataCleaner
    from .storage.database_manager import DatabaseManager
    
    __all__.extend([
        "PipelineManager",
        "FileIngestion", 
        "APIIngestion",
        "DataValidator",
        "DataCleaner",
        "DatabaseManager",
    ])
    
except ImportError:
    # Handle import errors gracefully during setup
    pass
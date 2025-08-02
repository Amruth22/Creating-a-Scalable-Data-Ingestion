"""
Pipeline management module for orchestrating the complete data ingestion pipeline
Coordinates all pipeline stages: ingestion, validation, transformation, and storage
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid

from ..ingestion.file_ingestion import FileIngestion
from ..ingestion.api_ingestion import APIIngestion
from ..validation.data_validator import DataValidator
from ..validation.schema_validator import SchemaValidator
from ..transformation.data_cleaner import DataCleaner
from ..transformation.data_enricher import DataEnricher
from ..transformation.data_standardizer import DataStandardizer
from ..storage.database_manager import DatabaseManager
from ..storage.file_manager import FileManager
from ..utils.config import config
from ..utils.helpers import safe_divide, format_duration
from ..utils.constants import PipelineStage, ProcessingStatus

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Pipeline execution result container"""
    success: bool
    run_id: str
    pipeline_name: str
    start_time: str
    end_time: str
    execution_time: float
    total_records_processed: int
    total_records_failed: int
    stages_completed: List[str]
    stages_failed: List[str]
    stage_results: Dict[str, Any]
    error_message: Optional[str] = None

class PipelineManager:
    """Comprehensive pipeline manager for data ingestion orchestration"""
    
    def __init__(self, pipeline_name: str = "data_ingestion_pipeline"):
        """
        Initialize pipeline manager
        
        Args:
            pipeline_name (str): Name of the pipeline
        """
        self.pipeline_name = pipeline_name
        self.run_id = None
        
        # Initialize components
        self.file_ingestion = FileIngestion()
        self.api_ingestion = APIIngestion()
        self.data_validator = DataValidator()
        self.schema_validator = SchemaValidator()
        self.data_cleaner = DataCleaner()
        self.data_enricher = DataEnricher()
        self.data_standardizer = DataStandardizer()
        self.database_manager = DatabaseManager()
        self.file_manager = FileManager()
        
        # Pipeline configuration
        self.enable_file_ingestion = True
        self.enable_api_ingestion = True
        self.enable_validation = True
        self.enable_cleaning = True
        self.enable_enrichment = True
        self.enable_standardization = True
        self.enable_database_storage = True
        self.enable_file_export = True
        
        logger.info(f"Pipeline manager initialized: {pipeline_name}")
    
    def run_pipeline(self, **kwargs) -> PipelineResult:
        """
        Execute the complete data ingestion pipeline
        
        Args:
            **kwargs: Pipeline configuration options
            
        Returns:
            PipelineResult: Complete pipeline execution result
        """
        # Generate unique run ID
        self.run_id = f"RUN-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{str(uuid.uuid4())[:8]}"
        start_time = datetime.now()
        
        logger.info(f"🚀 Starting pipeline execution: {self.run_id}")
        
        # Initialize result tracking
        result = PipelineResult(
            success=False,
            run_id=self.run_id,
            pipeline_name=self.pipeline_name,
            start_time=start_time.isoformat(),
            end_time="",
            execution_time=0.0,
            total_records_processed=0,
            total_records_failed=0,
            stages_completed=[],
            stages_failed=[],
            stage_results={}
        )
        
        try:
            # Save pipeline run start
            self._save_pipeline_run_start(start_time)
            
            # Stage 1: Data Ingestion
            ingestion_result = self._execute_ingestion_stage()
            result.stage_results['ingestion'] = ingestion_result
            
            if not ingestion_result['success']:
                result.stages_failed.append(PipelineStage.INGESTION.value)
                raise Exception(f"Ingestion stage failed: {ingestion_result.get('error', 'Unknown error')}")
            
            result.stages_completed.append(PipelineStage.INGESTION.value)
            combined_data = ingestion_result['data']
            
            if combined_data is None or combined_data.empty:
                raise Exception("No data collected from ingestion stage")
            
            logger.info(f"✅ Ingestion completed: {len(combined_data)} records collected")
            
            # Stage 2: Data Validation
            if self.enable_validation:
                validation_result = self._execute_validation_stage(combined_data)
                result.stage_results['validation'] = validation_result
                
                if not validation_result['success']:
                    result.stages_failed.append(PipelineStage.VALIDATION.value)
                    # Continue with warnings but don't fail pipeline
                    logger.warning(f"Validation stage completed with issues: {validation_result.get('error', 'Unknown error')}")
                else:
                    result.stages_completed.append(PipelineStage.VALIDATION.value)
                    logger.info(f"✅ Validation completed: {validation_result['quality_score']:.1f}% quality score")
            
            # Stage 3: Data Transformation
            transformation_result = self._execute_transformation_stage(combined_data)
            result.stage_results['transformation'] = transformation_result
            
            if not transformation_result['success']:
                result.stages_failed.append(PipelineStage.TRANSFORMATION.value)
                raise Exception(f"Transformation stage failed: {transformation_result.get('error', 'Unknown error')}")
            
            result.stages_completed.append(PipelineStage.TRANSFORMATION.value)
            processed_data = transformation_result['data']
            
            logger.info(f"✅ Transformation completed: {len(processed_data)} records processed")
            
            # Stage 4: Data Storage
            storage_result = self._execute_storage_stage(processed_data)
            result.stage_results['storage'] = storage_result
            
            if not storage_result['success']:
                result.stages_failed.append(PipelineStage.STORAGE.value)
                raise Exception(f"Storage stage failed: {storage_result.get('error', 'Unknown error')}")
            
            result.stages_completed.append(PipelineStage.STORAGE.value)
            
            logger.info(f"✅ Storage completed: {storage_result['records_stored']} records stored")
            
            # Calculate final metrics
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result.success = True
            result.end_time = end_time.isoformat()
            result.execution_time = execution_time
            result.total_records_processed = len(processed_data)
            result.total_records_failed = ingestion_result.get('failed_records', 0)
            
            # Save pipeline run completion
            self._save_pipeline_run_completion(result)
            
            logger.info(f"🎉 Pipeline execution completed successfully: {self.run_id}")
            logger.info(f"📊 Summary: {result.total_records_processed} records processed in {format_duration(execution_time)}")
            
            return result
            
        except Exception as e:
            # Handle pipeline failure
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result.success = False
            result.end_time = end_time.isoformat()
            result.execution_time = execution_time
            result.error_message = str(e)
            
            # Save failed pipeline run
            self._save_pipeline_run_completion(result)
            
            logger.error(f"❌ Pipeline execution failed: {self.run_id} - {str(e)}")
            
            return result
    
    def _execute_ingestion_stage(self) -> Dict[str, Any]:
        """Execute data ingestion stage"""
        logger.info("📥 Executing ingestion stage...")
        
        try:
            all_data = []
            ingestion_summary = {
                'file_ingestion': {'enabled': False, 'records': 0, 'success': False},
                'api_ingestion': {'enabled': False, 'records': 0, 'success': False}
            }
            
            # File ingestion
            if self.enable_file_ingestion:
                logger.info("📁 Processing files...")
                file_result = self.file_ingestion.process_all_files()
                ingestion_summary['file_ingestion']['enabled'] = True
                ingestion_summary['file_ingestion']['success'] = file_result['successful_files'] > 0
                
                if file_result['successful_files'] > 0:
                    # Collect data from successful file processing results BEFORE files are moved
                    successful_results = [
                        result for result in file_result['results'] 
                        if result['success'] and result.get('data') is not None
                    ]
                    
                    if successful_results:
                        # Combine all DataFrames from successful results
                        file_dataframes = [result['data'] for result in successful_results]
                        file_data = pd.concat(file_dataframes, ignore_index=True)
                        
                        # Remove duplicates based on order_id if present
                        if 'order_id' in file_data.columns:
                            file_data = file_data.drop_duplicates(subset=['order_id'], keep='first')
                        
                        all_data.append(file_data)
                        ingestion_summary['file_ingestion']['records'] = len(file_data)
                        logger.info(f"📁 File ingestion: {len(file_data)} records")
            
            # API ingestion
            if self.enable_api_ingestion:
                logger.info("🌐 Processing API data...")
                api_result = self.api_ingestion.fetch_orders_from_api(limit=100)
                ingestion_summary['api_ingestion']['enabled'] = True
                ingestion_summary['api_ingestion']['success'] = api_result['success']
                
                if api_result['success'] and api_result['data'] is not None:
                    all_data.append(api_result['data'])
                    ingestion_summary['api_ingestion']['records'] = len(api_result['data'])
                    logger.info(f"🌐 API ingestion: {len(api_result['data'])} records")
            
            # Combine all data
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                # Remove duplicates based on order_id if present
                if 'order_id' in combined_data.columns:
                    combined_data = combined_data.drop_duplicates(subset=['order_id'], keep='first')
            else:
                combined_data = pd.DataFrame()
            
            total_records = len(combined_data)
            success = total_records > 0
            
            return {
                'success': success,
                'data': combined_data,
                'total_records': total_records,
                'summary': ingestion_summary,
                'error': None if success else "No data collected from any source"
            }
            
        except Exception as e:
            logger.error(f"Error in ingestion stage: {e}")
            return {
                'success': False,
                'data': None,
                'total_records': 0,
                'summary': {},
                'error': str(e)
            }
    
    def _execute_validation_stage(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute data validation stage"""
        logger.info("🔍 Executing validation stage...")
        
        try:
            # Schema validation
            schema_result = self.schema_validator.validate_schema(data, 'orders')
            
            # Data validation
            data_result = self.data_validator.validate_orders(data)
            
            # Combine results
            overall_success = schema_result.is_valid and data_result.quality_score >= 50  # Minimum 50% quality
            
            return {
                'success': overall_success,
                'schema_validation': {
                    'is_valid': schema_result.is_valid,
                    'valid_fields': schema_result.valid_fields,
                    'total_fields': schema_result.total_fields,
                    'errors': len(schema_result.errors),
                    'warnings': len(schema_result.warnings)
                },
                'data_validation': {
                    'quality_score': data_result.quality_score,
                    'quality_level': data_result.quality_level,
                    'valid_records': data_result.valid_records,
                    'invalid_records': data_result.invalid_records,
                    'errors': len(data_result.errors),
                    'warnings': len(data_result.warnings)
                },
                'quality_score': data_result.quality_score,
                'error': None if overall_success else f"Validation failed: Schema valid: {schema_result.is_valid}, Quality: {data_result.quality_score:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"Error in validation stage: {e}")
            return {
                'success': False,
                'schema_validation': {},
                'data_validation': {},
                'quality_score': 0.0,
                'error': str(e)
            }
    
    def _execute_transformation_stage(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute data transformation stage"""
        logger.info("🧹 Executing transformation stage...")
        
        try:
            current_data = data.copy()
            transformation_summary = {}
            
            # Data cleaning
            if self.enable_cleaning:
                logger.info("🧽 Cleaning data...")
                cleaning_result = self.data_cleaner.clean_orders(current_data)
                transformation_summary['cleaning'] = {
                    'success': cleaning_result.success,
                    'original_records': cleaning_result.original_records,
                    'cleaned_records': cleaning_result.cleaned_records,
                    'removed_records': cleaning_result.removed_records,
                    'retention_rate': safe_divide(cleaning_result.cleaned_records, cleaning_result.original_records, 0) * 100
                }
                
                if cleaning_result.success and cleaning_result.data is not None:
                    current_data = cleaning_result.data
                    logger.info(f"🧽 Cleaning: {cleaning_result.cleaned_records}/{cleaning_result.original_records} records retained")
                else:
                    raise Exception(f"Data cleaning failed: {cleaning_result.errors}")
            
            # Data standardization
            if self.enable_standardization:
                logger.info("📏 Standardizing data...")
                standardization_result = self.data_standardizer.standardize_orders(current_data)
                transformation_summary['standardization'] = {
                    'success': standardization_result.success,
                    'records_processed': standardization_result.records_processed,
                    'fields_standardized': standardization_result.fields_standardized
                }
                
                if standardization_result.success and standardization_result.data is not None:
                    current_data = standardization_result.data
                    logger.info(f"📏 Standardization: {standardization_result.fields_standardized} fields standardized")
                else:
                    logger.warning(f"Data standardization issues: {standardization_result.errors}")
            
            # Data enrichment
            if self.enable_enrichment:
                logger.info("➕ Enriching data...")
                enrichment_result = self.data_enricher.enrich_orders(current_data)
                transformation_summary['enrichment'] = {
                    'success': enrichment_result.success,
                    'original_fields': enrichment_result.original_fields,
                    'enriched_fields': enrichment_result.enriched_fields,
                    'added_fields': enrichment_result.added_fields
                }
                
                if enrichment_result.success and enrichment_result.data is not None:
                    current_data = enrichment_result.data
                    logger.info(f"➕ Enrichment: {enrichment_result.added_fields} fields added")
                else:
                    logger.warning(f"Data enrichment issues: {enrichment_result.errors}")
            
            return {
                'success': True,
                'data': current_data,
                'final_records': len(current_data),
                'final_fields': len(current_data.columns),
                'transformation_summary': transformation_summary,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error in transformation stage: {e}")
            return {
                'success': False,
                'data': None,
                'final_records': 0,
                'final_fields': 0,
                'transformation_summary': {},
                'error': str(e)
            }
    
    def _execute_storage_stage(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute data storage stage"""
        logger.info("💾 Executing storage stage...")
        
        try:
            storage_summary = {}
            total_stored = 0
            
            # Database storage
            if self.enable_database_storage:
                logger.info("🗄️ Storing to database...")
                db_result = self.database_manager.save_orders(data)
                storage_summary['database'] = {
                    'success': db_result.success,
                    'records_affected': db_result.records_affected,
                    'execution_time': db_result.execution_time
                }
                
                if db_result.success:
                    total_stored += db_result.records_affected
                    logger.info(f"🗄️ Database storage: {db_result.records_affected} records saved")
                else:
                    raise Exception(f"Database storage failed: {db_result.error_message}")
            
            # File export
            if self.enable_file_export:
                logger.info("📁 Exporting to files...")
                export_name = f"processed_orders_{self.run_id}"
                export_result = self.file_manager.export_data_summary(data, export_name)
                storage_summary['file_export'] = {
                    'success': export_result.success,
                    'files_created': export_result.files_affected,
                    'execution_time': export_result.execution_time
                }
                
                if export_result.success:
                    logger.info(f"📁 File export: {export_result.files_affected} files created")
                else:
                    logger.warning(f"File export issues: {export_result.error_message}")
            
            return {
                'success': True,
                'records_stored': total_stored,
                'storage_summary': storage_summary,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error in storage stage: {e}")
            return {
                'success': False,
                'records_stored': 0,
                'storage_summary': {},
                'error': str(e)
            }
    
    def _save_pipeline_run_start(self, start_time: datetime):
        """Save pipeline run start information"""
        try:
            run_data = {
                'run_id': self.run_id,
                'pipeline_name': self.pipeline_name,
                'start_time': start_time.isoformat(),
                'end_time': None,
                'status': ProcessingStatus.PROCESSING.value,
                'records_processed': 0,
                'records_failed': 0,
                'error_message': None
            }
            
            self.database_manager.save_pipeline_run(run_data)
            
        except Exception as e:
            logger.warning(f"Failed to save pipeline run start: {e}")
    
    def _save_pipeline_run_completion(self, result: PipelineResult):
        """Save pipeline run completion information"""
        try:
            run_data = {
                'run_id': result.run_id,
                'pipeline_name': result.pipeline_name,
                'start_time': result.start_time,
                'end_time': result.end_time,
                'status': ProcessingStatus.COMPLETED.value if result.success else ProcessingStatus.FAILED.value,
                'records_processed': result.total_records_processed,
                'records_failed': result.total_records_failed,
                'error_message': result.error_message
            }
            
            self.database_manager.save_pipeline_run(run_data)
            
            # Save data quality metrics if available
            if result.success and 'validation' in result.stage_results:
                validation_result = result.stage_results['validation']
                metrics_data = [
                    {
                        'run_id': result.run_id,
                        'metric_name': 'data_quality_score',
                        'metric_value': validation_result.get('quality_score', 0),
                        'metric_type': 'percentage',
                        'source_table': 'orders',
                        'measured_at': result.end_time
                    },
                    {
                        'run_id': result.run_id,
                        'metric_name': 'records_processed',
                        'metric_value': result.total_records_processed,
                        'metric_type': 'count',
                        'source_table': 'orders',
                        'measured_at': result.end_time
                    },
                    {
                        'run_id': result.run_id,
                        'metric_name': 'execution_time_seconds',
                        'metric_value': result.execution_time,
                        'metric_type': 'duration',
                        'source_table': 'pipeline_runs',
                        'measured_at': result.end_time
                    }
                ]
                
                self.database_manager.save_data_quality_metrics(metrics_data)
            
        except Exception as e:
            logger.warning(f"Failed to save pipeline run completion: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and recent runs"""
        try:
            # Get recent pipeline runs
            runs_result = self.database_manager.get_pipeline_runs(limit=10)
            
            # Get database stats
            stats_result = self.database_manager.get_database_stats()
            
            # Get storage summary
            storage_summary = self.file_manager.get_storage_summary()
            
            status = {
                'pipeline_name': self.pipeline_name,
                'current_run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'recent_runs': runs_result.data.to_dict('records') if runs_result.success else [],
                'database_stats': stats_result.data.iloc[0].to_dict() if stats_result.success else {},
                'storage_summary': storage_summary,
                'configuration': {
                    'file_ingestion_enabled': self.enable_file_ingestion,
                    'api_ingestion_enabled': self.enable_api_ingestion,
                    'validation_enabled': self.enable_validation,
                    'cleaning_enabled': self.enable_cleaning,
                    'enrichment_enabled': self.enable_enrichment,
                    'standardization_enabled': self.enable_standardization,
                    'database_storage_enabled': self.enable_database_storage,
                    'file_export_enabled': self.enable_file_export
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_pipeline_report(self, result: PipelineResult) -> str:
        """Generate comprehensive pipeline execution report"""
        report = []
        report.append("# Data Ingestion Pipeline Execution Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"- **Pipeline**: {result.pipeline_name}")
        report.append(f"- **Run ID**: {result.run_id}")
        report.append(f"- **Status**: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        report.append(f"- **Execution Time**: {format_duration(result.execution_time)}")
        report.append(f"- **Records Processed**: {result.total_records_processed:,}")
        report.append(f"- **Records Failed**: {result.total_records_failed:,}")
        report.append(f"- **Success Rate**: {safe_divide(result.total_records_processed - result.total_records_failed, result.total_records_processed, 0) * 100:.1f}%")
        report.append("")
        
        # Stage Results
        report.append("## Stage Results")
        
        for stage_name, stage_result in result.stage_results.items():
            status_icon = "✅" if stage_result.get('success', False) else "❌"
            report.append(f"### {status_icon} {stage_name.title()} Stage")
            
            if stage_name == 'ingestion':
                report.append(f"- **Total Records**: {stage_result.get('total_records', 0):,}")
                summary = stage_result.get('summary', {})
                for source, details in summary.items():
                    if details.get('enabled', False):
                        status = "✅" if details.get('success', False) else "❌"
                        report.append(f"- **{source.replace('_', ' ').title()}**: {status} ({details.get('records', 0):,} records)")
            
            elif stage_name == 'validation':
                report.append(f"- **Quality Score**: {stage_result.get('quality_score', 0):.1f}%")
                data_val = stage_result.get('data_validation', {})
                report.append(f"- **Valid Records**: {data_val.get('valid_records', 0):,}")
                report.append(f"- **Invalid Records**: {data_val.get('invalid_records', 0):,}")
            
            elif stage_name == 'transformation':
                report.append(f"- **Final Records**: {stage_result.get('final_records', 0):,}")
                report.append(f"- **Final Fields**: {stage_result.get('final_fields', 0)}")
                trans_summary = stage_result.get('transformation_summary', {})
                for trans_type, details in trans_summary.items():
                    if details.get('success', False):
                        report.append(f"- **{trans_type.title()}**: ✅ Completed")
            
            elif stage_name == 'storage':
                report.append(f"- **Records Stored**: {stage_result.get('records_stored', 0):,}")
                storage_summary = stage_result.get('storage_summary', {})
                for storage_type, details in storage_summary.items():
                    if details.get('success', False):
                        report.append(f"- **{storage_type.replace('_', ' ').title()}**: ✅ Completed")
            
            if stage_result.get('error'):
                report.append(f"- **Error**: {stage_result['error']}")
            
            report.append("")
        
        # Stages Summary
        report.append("## Stages Summary")
        report.append(f"- **Completed Stages**: {', '.join(result.stages_completed) if result.stages_completed else 'None'}")
        report.append(f"- **Failed Stages**: {', '.join(result.stages_failed) if result.stages_failed else 'None'}")
        report.append("")
        
        # Error Details
        if result.error_message:
            report.append("## Error Details")
            report.append(f"```")
            report.append(result.error_message)
            report.append(f"```")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if result.success:
            report.append("- ✅ Pipeline executed successfully")
            report.append("- Consider scheduling regular pipeline runs")
            report.append("- Monitor data quality trends over time")
        else:
            report.append("- ❌ Pipeline execution failed")
            report.append("- Review error details and fix issues")
            report.append("- Check data source availability")
            report.append("- Verify system resources and permissions")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test pipeline manager
    print("Testing Pipeline Manager...")
    
    # Initialize pipeline manager
    pipeline = PipelineManager("test_pipeline")
    
    # Run pipeline
    print("🚀 Running test pipeline...")
    result = pipeline.run_pipeline()
    
    # Print results
    print(f"\nPipeline Result:")
    print(f"Success: {result.success}")
    print(f"Run ID: {result.run_id}")
    print(f"Execution Time: {format_duration(result.execution_time)}")
    print(f"Records Processed: {result.total_records_processed}")
    print(f"Stages Completed: {', '.join(result.stages_completed)}")
    
    if result.stages_failed:
        print(f"Stages Failed: {', '.join(result.stages_failed)}")
    
    if result.error_message:
        print(f"Error: {result.error_message}")
    
    # Generate report
    report = pipeline.generate_pipeline_report(result)
    print(f"\nGenerated pipeline report ({len(report)} characters)")
    
    # Get pipeline status
    status = pipeline.get_pipeline_status()
    print(f"\nPipeline status retrieved: {'error' not in status}")
    
    print("Pipeline manager test completed!")
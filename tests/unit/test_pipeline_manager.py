"""
Unit tests for pipeline manager module
Tests pipeline orchestration, stage coordination, error handling, and reporting
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call
from datetime import datetime

from src.pipeline.pipeline_manager import PipelineManager, PipelineResult
from src.utils.constants import PipelineStage, ProcessingStatus

@pytest.mark.unit
class TestPipelineManager:
    """Test cases for PipelineManager class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.pipeline_manager = PipelineManager("test_pipeline")
    
    def test_initialization(self):
        """Test PipelineManager initialization"""
        assert self.pipeline_manager is not None
        assert self.pipeline_manager.pipeline_name == "test_pipeline"
        assert self.pipeline_manager.run_id is None
        
        # Check that all components are initialized
        assert hasattr(self.pipeline_manager, 'file_ingestion')
        assert hasattr(self.pipeline_manager, 'api_ingestion')
        assert hasattr(self.pipeline_manager, 'data_validator')
        assert hasattr(self.pipeline_manager, 'data_cleaner')
        assert hasattr(self.pipeline_manager, 'database_manager')
        
        # Check default stage enablement
        assert self.pipeline_manager.enable_file_ingestion is True
        assert self.pipeline_manager.enable_api_ingestion is True
        assert self.pipeline_manager.enable_validation is True
        assert self.pipeline_manager.enable_cleaning is True
        assert self.pipeline_manager.enable_database_storage is True
    
    def test_stage_configuration(self):
        """Test pipeline stage configuration"""
        # Test disabling stages
        self.pipeline_manager.enable_file_ingestion = False
        self.pipeline_manager.enable_api_ingestion = False
        self.pipeline_manager.enable_validation = False
        
        assert self.pipeline_manager.enable_file_ingestion is False
        assert self.pipeline_manager.enable_api_ingestion is False
        assert self.pipeline_manager.enable_validation is False
    
    @patch('src.pipeline.pipeline_manager.FileIngestion')
    @patch('src.pipeline.pipeline_manager.APIIngestion')
    def test_execute_ingestion_stage_success(self, mock_api_ingestion, mock_file_ingestion, sample_orders_data):
        """Test successful ingestion stage execution"""
        # Mock file ingestion
        mock_file_instance = MagicMock()
        mock_file_instance.process_all_files.return_value = {
            'successful_files': 2,
            'failed_files': 0,
            'total_records': 100
        }
        mock_file_instance.get_combined_data.return_value = sample_orders_data
        mock_file_ingestion.return_value = mock_file_instance
        
        # Mock API ingestion
        mock_api_instance = MagicMock()
        mock_api_instance.fetch_orders_from_api.return_value = {
            'success': True,
            'data': sample_orders_data
        }
        mock_api_ingestion.return_value = mock_api_instance
        
        # Create new pipeline manager with mocked components
        pipeline = PipelineManager("test_pipeline")
        pipeline.file_ingestion = mock_file_instance
        pipeline.api_ingestion = mock_api_instance
        
        result = pipeline._execute_ingestion_stage()
        
        assert result['success'] is True
        assert result['data'] is not None
        assert isinstance(result['data'], pd.DataFrame)
        assert result['total_records'] > 0
        assert 'summary' in result
    
    @patch('src.pipeline.pipeline_manager.FileIngestion')
    def test_execute_ingestion_stage_no_data(self, mock_file_ingestion):
        """Test ingestion stage with no data collected"""
        # Mock file ingestion to return no data
        mock_file_instance = MagicMock()
        mock_file_instance.process_all_files.return_value = {
            'successful_files': 0,
            'failed_files': 0,
            'total_records': 0
        }
        mock_file_instance.get_combined_data.return_value = None
        mock_file_ingestion.return_value = mock_file_instance
        
        # Mock API ingestion to return no data
        mock_api_instance = MagicMock()
        mock_api_instance.fetch_orders_from_api.return_value = {
            'success': False,
            'data': None
        }
        
        pipeline = PipelineManager("test_pipeline")
        pipeline.file_ingestion = mock_file_instance
        pipeline.api_ingestion = mock_api_instance
        
        result = pipeline._execute_ingestion_stage()
        
        assert result['success'] is False
        assert result['total_records'] == 0
        assert 'No data collected' in result['error']
    
    def test_execute_ingestion_stage_disabled_sources(self, sample_orders_data):
        """Test ingestion stage with disabled sources"""
        # Disable file ingestion
        self.pipeline_manager.enable_file_ingestion = False
        
        # Mock API ingestion to return data
        mock_api_instance = MagicMock()
        mock_api_instance.fetch_orders_from_api.return_value = {
            'success': True,
            'data': sample_orders_data
        }
        self.pipeline_manager.api_ingestion = mock_api_instance
        
        result = self.pipeline_manager._execute_ingestion_stage()
        
        assert result['success'] is True
        assert result['summary']['file_ingestion']['enabled'] is False
        assert result['summary']['api_ingestion']['enabled'] is True
    
    def test_execute_validation_stage_success(self, sample_orders_data):
        """Test successful validation stage execution"""
        # Mock schema validator
        mock_schema_result = MagicMock()
        mock_schema_result.is_valid = True
        mock_schema_result.valid_fields = 10
        mock_schema_result.total_fields = 10
        mock_schema_result.errors = []
        mock_schema_result.warnings = []
        
        # Mock data validator
        mock_data_result = MagicMock()
        mock_data_result.quality_score = 95.0
        mock_data_result.quality_level = 'excellent'
        mock_data_result.valid_records = 3
        mock_data_result.invalid_records = 0
        mock_data_result.errors = []
        mock_data_result.warnings = []
        
        self.pipeline_manager.schema_validator.validate_schema = MagicMock(return_value=mock_schema_result)
        self.pipeline_manager.data_validator.validate_orders = MagicMock(return_value=mock_data_result)
        
        result = self.pipeline_manager._execute_validation_stage(sample_orders_data)
        
        assert result['success'] is True
        assert result['quality_score'] == 95.0
        assert 'schema_validation' in result
        assert 'data_validation' in result
    
    def test_execute_validation_stage_low_quality(self, sample_orders_data):
        """Test validation stage with low quality data"""
        # Mock schema validator
        mock_schema_result = MagicMock()
        mock_schema_result.is_valid = True
        mock_schema_result.valid_fields = 8
        mock_schema_result.total_fields = 10
        mock_schema_result.errors = []
        mock_schema_result.warnings = []
        
        # Mock data validator with low quality score
        mock_data_result = MagicMock()
        mock_data_result.quality_score = 30.0  # Below threshold
        mock_data_result.quality_level = 'critical'
        mock_data_result.valid_records = 1
        mock_data_result.invalid_records = 2
        mock_data_result.errors = [{'type': 'validation_error', 'message': 'Test error'}]
        mock_data_result.warnings = []
        
        self.pipeline_manager.schema_validator.validate_schema = MagicMock(return_value=mock_schema_result)
        self.pipeline_manager.data_validator.validate_orders = MagicMock(return_value=mock_data_result)
        
        result = self.pipeline_manager._execute_validation_stage(sample_orders_data)
        
        assert result['success'] is False  # Should fail due to low quality
        assert result['quality_score'] == 30.0
    
    def test_execute_transformation_stage_success(self, sample_orders_data):
        """Test successful transformation stage execution"""
        # Mock data cleaner
        mock_cleaning_result = MagicMock()
        mock_cleaning_result.success = True
        mock_cleaning_result.data = sample_orders_data
        mock_cleaning_result.original_records = 3
        mock_cleaning_result.cleaned_records = 3
        mock_cleaning_result.removed_records = 0
        mock_cleaning_result.errors = []
        
        # Mock data standardizer
        mock_standardization_result = MagicMock()
        mock_standardization_result.success = True
        mock_standardization_result.data = sample_orders_data
        mock_standardization_result.records_processed = 3
        mock_standardization_result.fields_standardized = 5
        mock_standardization_result.errors = []
        
        # Mock data enricher
        mock_enrichment_result = MagicMock()
        mock_enrichment_result.success = True
        mock_enrichment_result.data = sample_orders_data
        mock_enrichment_result.original_fields = 10
        mock_enrichment_result.enriched_fields = 15
        mock_enrichment_result.added_fields = 5
        mock_enrichment_result.errors = []
        
        self.pipeline_manager.data_cleaner.clean_orders = MagicMock(return_value=mock_cleaning_result)
        self.pipeline_manager.data_standardizer.standardize_orders = MagicMock(return_value=mock_standardization_result)
        self.pipeline_manager.data_enricher.enrich_orders = MagicMock(return_value=mock_enrichment_result)
        
        result = self.pipeline_manager._execute_transformation_stage(sample_orders_data)
        
        assert result['success'] is True
        assert result['data'] is not None
        assert result['final_records'] == len(sample_orders_data)
        assert 'transformation_summary' in result
        assert 'cleaning' in result['transformation_summary']
        assert 'standardization' in result['transformation_summary']
        assert 'enrichment' in result['transformation_summary']
    
    def test_execute_transformation_stage_cleaning_failure(self, sample_orders_data):
        """Test transformation stage with cleaning failure"""
        # Mock data cleaner to fail
        mock_cleaning_result = MagicMock()
        mock_cleaning_result.success = False
        mock_cleaning_result.data = None
        mock_cleaning_result.errors = ['Cleaning failed']
        
        self.pipeline_manager.data_cleaner.clean_orders = MagicMock(return_value=mock_cleaning_result)
        
        result = self.pipeline_manager._execute_transformation_stage(sample_orders_data)
        
        assert result['success'] is False
        assert 'Cleaning failed' in result['error']
    
    def test_execute_storage_stage_success(self, sample_orders_data):
        """Test successful storage stage execution"""
        # Mock database manager
        mock_db_result = MagicMock()
        mock_db_result.success = True
        mock_db_result.records_affected = len(sample_orders_data)
        mock_db_result.execution_time = 1.5
        
        # Mock file manager
        mock_file_result = MagicMock()
        mock_file_result.success = True
        mock_file_result.files_affected = 2
        mock_file_result.execution_time = 0.5
        
        self.pipeline_manager.database_manager.save_orders = MagicMock(return_value=mock_db_result)
        self.pipeline_manager.file_manager.export_data_summary = MagicMock(return_value=mock_file_result)
        
        result = self.pipeline_manager._execute_storage_stage(sample_orders_data)
        
        assert result['success'] is True
        assert result['records_stored'] == len(sample_orders_data)
        assert 'storage_summary' in result
        assert 'database' in result['storage_summary']
        assert 'file_export' in result['storage_summary']
    
    def test_execute_storage_stage_database_failure(self, sample_orders_data):
        """Test storage stage with database failure"""
        # Mock database manager to fail
        mock_db_result = MagicMock()
        mock_db_result.success = False
        mock_db_result.error_message = "Database connection failed"
        
        self.pipeline_manager.database_manager.save_orders = MagicMock(return_value=mock_db_result)
        
        result = self.pipeline_manager._execute_storage_stage(sample_orders_data)
        
        assert result['success'] is False
        assert 'Database connection failed' in result['error']
    
    def test_execute_storage_stage_disabled_database(self, sample_orders_data):
        """Test storage stage with database storage disabled"""
        # Disable database storage
        self.pipeline_manager.enable_database_storage = False
        
        # Mock file manager
        mock_file_result = MagicMock()
        mock_file_result.success = True
        mock_file_result.files_affected = 2
        mock_file_result.execution_time = 0.5
        
        self.pipeline_manager.file_manager.export_data_summary = MagicMock(return_value=mock_file_result)
        
        result = self.pipeline_manager._execute_storage_stage(sample_orders_data)
        
        assert result['success'] is True
        assert result['records_stored'] == 0  # No database storage
        assert 'database' not in result['storage_summary']
        assert 'file_export' in result['storage_summary']
    
    @patch('src.pipeline.pipeline_manager.PipelineManager._execute_ingestion_stage')
    @patch('src.pipeline.pipeline_manager.PipelineManager._execute_validation_stage')
    @patch('src.pipeline.pipeline_manager.PipelineManager._execute_transformation_stage')
    @patch('src.pipeline.pipeline_manager.PipelineManager._execute_storage_stage')
    def test_run_pipeline_success(self, mock_storage, mock_transformation, mock_validation, mock_ingestion, sample_orders_data):
        """Test successful complete pipeline execution"""
        # Mock all stages to succeed
        mock_ingestion.return_value = {
            'success': True,
            'data': sample_orders_data,
            'total_records': len(sample_orders_data),
            'summary': {'file_ingestion': {'enabled': True, 'records': 3, 'success': True}}
        }
        
        mock_validation.return_value = {
            'success': True,
            'quality_score': 95.0,
            'schema_validation': {'is_valid': True},
            'data_validation': {'quality_score': 95.0}
        }
        
        mock_transformation.return_value = {
            'success': True,
            'data': sample_orders_data,
            'final_records': len(sample_orders_data),
            'transformation_summary': {'cleaning': {'success': True}}
        }
        
        mock_storage.return_value = {
            'success': True,
            'records_stored': len(sample_orders_data),
            'storage_summary': {'database': {'success': True}}
        }
        
        result = self.pipeline_manager.run_pipeline()
        
        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert result.run_id is not None
        assert result.pipeline_name == "test_pipeline"
        assert result.total_records_processed == len(sample_orders_data)
        assert result.total_records_failed == 0
        assert len(result.stages_completed) == 4  # All stages completed
        assert len(result.stages_failed) == 0
        assert result.execution_time > 0
        assert result.error_message is None
    
    @patch('src.pipeline.pipeline_manager.PipelineManager._execute_ingestion_stage')
    def test_run_pipeline_ingestion_failure(self, mock_ingestion):
        """Test pipeline execution with ingestion failure"""
        # Mock ingestion to fail
        mock_ingestion.return_value = {
            'success': False,
            'data': None,
            'total_records': 0,
            'error': 'No data sources available'
        }
        
        result = self.pipeline_manager.run_pipeline()
        
        assert isinstance(result, PipelineResult)
        assert result.success is False
        assert PipelineStage.INGESTION.value in result.stages_failed
        assert result.total_records_processed == 0
        assert result.error_message is not None
        assert 'Ingestion stage failed' in result.error_message
    
    @patch('src.pipeline.pipeline_manager.PipelineManager._execute_ingestion_stage')
    @patch('src.pipeline.pipeline_manager.PipelineManager._execute_validation_stage')
    def test_run_pipeline_validation_warning(self, mock_validation, mock_ingestion, sample_orders_data):
        """Test pipeline execution with validation warnings (should continue)"""
        # Mock ingestion to succeed
        mock_ingestion.return_value = {
            'success': True,
            'data': sample_orders_data,
            'total_records': len(sample_orders_data),
            'summary': {}
        }
        
        # Mock validation to have issues but not fail completely
        mock_validation.return_value = {
            'success': False,  # Has issues
            'quality_score': 75.0,  # But not critical
            'error': 'Some validation warnings'
        }
        
        # Mock other stages
        with patch.object(self.pipeline_manager, '_execute_transformation_stage') as mock_transform, \
             patch.object(self.pipeline_manager, '_execute_storage_stage') as mock_store:
            
            mock_transform.return_value = {
                'success': True,
                'data': sample_orders_data,
                'final_records': len(sample_orders_data)
            }
            
            mock_store.return_value = {
                'success': True,
                'records_stored': len(sample_orders_data)
            }
            
            result = self.pipeline_manager.run_pipeline()
            
            # Pipeline should continue despite validation warnings
            assert result.success is True
            assert PipelineStage.VALIDATION.value not in result.stages_failed
    
    def test_run_pipeline_disabled_validation(self, sample_orders_data):
        """Test pipeline execution with validation disabled"""
        # Disable validation
        self.pipeline_manager.enable_validation = False
        
        # Mock other stages
        with patch.object(self.pipeline_manager, '_execute_ingestion_stage') as mock_ingestion, \
             patch.object(self.pipeline_manager, '_execute_transformation_stage') as mock_transform, \
             patch.object(self.pipeline_manager, '_execute_storage_stage') as mock_store:
            
            mock_ingestion.return_value = {
                'success': True,
                'data': sample_orders_data,
                'total_records': len(sample_orders_data),
                'summary': {}
            }
            
            mock_transform.return_value = {
                'success': True,
                'data': sample_orders_data,
                'final_records': len(sample_orders_data)
            }
            
            mock_store.return_value = {
                'success': True,
                'records_stored': len(sample_orders_data)
            }
            
            result = self.pipeline_manager.run_pipeline()
            
            assert result.success is True
            assert PipelineStage.VALIDATION.value not in result.stages_completed
            assert 'validation' not in result.stage_results
    
    def test_save_pipeline_run_start(self):
        """Test saving pipeline run start information"""
        start_time = datetime.now()
        
        # Mock database manager
        mock_db_result = MagicMock()
        mock_db_result.success = True
        self.pipeline_manager.database_manager.save_pipeline_run = MagicMock(return_value=mock_db_result)
        
        self.pipeline_manager.run_id = "TEST-RUN-001"
        self.pipeline_manager._save_pipeline_run_start(start_time)
        
        # Verify database manager was called
        self.pipeline_manager.database_manager.save_pipeline_run.assert_called_once()
        call_args = self.pipeline_manager.database_manager.save_pipeline_run.call_args[0][0]
        assert call_args['run_id'] == "TEST-RUN-001"
        assert call_args['status'] == ProcessingStatus.PROCESSING.value
    
    def test_save_pipeline_run_completion(self, sample_orders_data):
        """Test saving pipeline run completion information"""
        # Create a mock pipeline result
        result = PipelineResult(
            success=True,
            run_id="TEST-RUN-001",
            pipeline_name="test_pipeline",
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:30:00",
            execution_time=1800.0,
            total_records_processed=len(sample_orders_data),
            total_records_failed=0,
            stages_completed=[PipelineStage.INGESTION.value],
            stages_failed=[],
            stage_results={'validation': {'quality_score': 95.0}},
            error_message=None
        )
        
        # Mock database manager
        mock_db_result = MagicMock()
        mock_db_result.success = True
        self.pipeline_manager.database_manager.save_pipeline_run = MagicMock(return_value=mock_db_result)
        self.pipeline_manager.database_manager.save_data_quality_metrics = MagicMock(return_value=mock_db_result)
        
        self.pipeline_manager._save_pipeline_run_completion(result)
        
        # Verify database manager was called
        self.pipeline_manager.database_manager.save_pipeline_run.assert_called_once()
        self.pipeline_manager.database_manager.save_data_quality_metrics.assert_called_once()
    
    def test_get_pipeline_status(self):
        """Test getting pipeline status"""
        # Mock database manager
        mock_runs_result = MagicMock()
        mock_runs_result.success = True
        mock_runs_result.data = pd.DataFrame([{
            'run_id': 'TEST-RUN-001',
            'status': 'completed',
            'start_time': '2024-01-15T10:00:00'
        }])
        
        mock_stats_result = MagicMock()
        mock_stats_result.success = True
        mock_stats_result.data = pd.DataFrame([{
            'orders_count': 100,
            'database_size_mb': 5.2
        }])
        
        self.pipeline_manager.database_manager.get_pipeline_runs = MagicMock(return_value=mock_runs_result)
        self.pipeline_manager.database_manager.get_database_stats = MagicMock(return_value=mock_stats_result)
        
        # Mock file manager
        self.pipeline_manager.file_manager.get_storage_summary = MagicMock(return_value={'total_files': 10})
        
        status = self.pipeline_manager.get_pipeline_status()
        
        assert isinstance(status, dict)
        assert 'pipeline_name' in status
        assert 'timestamp' in status
        assert 'recent_runs' in status
        assert 'database_stats' in status
        assert 'configuration' in status
    
    def test_generate_pipeline_report(self, sample_orders_data):
        """Test generating pipeline execution report"""
        # Create a mock pipeline result
        result = PipelineResult(
            success=True,
            run_id="TEST-RUN-001",
            pipeline_name="test_pipeline",
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:30:00",
            execution_time=1800.0,
            total_records_processed=len(sample_orders_data),
            total_records_failed=0,
            stages_completed=[PipelineStage.INGESTION.value, PipelineStage.STORAGE.value],
            stages_failed=[],
            stage_results={
                'ingestion': {
                    'success': True,
                    'total_records': len(sample_orders_data),
                    'summary': {'file_ingestion': {'enabled': True, 'success': True, 'records': 3}}
                },
                'storage': {
                    'success': True,
                    'records_stored': len(sample_orders_data),
                    'storage_summary': {'database': {'success': True}}
                }
            },
            error_message=None
        )
        
        report = self.pipeline_manager.generate_pipeline_report(result)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "# Data Ingestion Pipeline Execution Report" in report
        assert "## Executive Summary" in report
        assert "## Stage Results" in report
        assert result.run_id in report
        assert result.pipeline_name in report
    
    def test_generate_pipeline_report_with_failure(self):
        """Test generating pipeline report with failure"""
        # Create a mock failed pipeline result
        result = PipelineResult(
            success=False,
            run_id="TEST-RUN-002",
            pipeline_name="test_pipeline",
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:05:00",
            execution_time=300.0,
            total_records_processed=0,
            total_records_failed=0,
            stages_completed=[],
            stages_failed=[PipelineStage.INGESTION.value],
            stage_results={
                'ingestion': {
                    'success': False,
                    'error': 'No data sources available'
                }
            },
            error_message="Ingestion stage failed: No data sources available"
        )
        
        report = self.pipeline_manager.generate_pipeline_report(result)
        
        assert "❌ FAILED" in report
        assert "## Error Details" in report
        assert result.error_message in report
        assert "## Recommendations" in report
        assert "❌ Pipeline execution failed" in report
    
    @patch('src.pipeline.pipeline_manager.logger')
    def test_logging_calls(self, mock_logger):
        """Test that appropriate logging calls are made"""
        # Mock all stages to succeed quickly
        with patch.object(self.pipeline_manager, '_execute_ingestion_stage') as mock_ingestion, \
             patch.object(self.pipeline_manager, '_execute_transformation_stage') as mock_transform, \
             patch.object(self.pipeline_manager, '_execute_storage_stage') as mock_store:
            
            mock_ingestion.return_value = {
                'success': True,
                'data': pd.DataFrame([{'test': 'data'}]),
                'total_records': 1,
                'summary': {}
            }
            
            mock_transform.return_value = {
                'success': True,
                'data': pd.DataFrame([{'test': 'data'}]),
                'final_records': 1
            }
            
            mock_store.return_value = {
                'success': True,
                'records_stored': 1
            }
            
            result = self.pipeline_manager.run_pipeline()
            
            # Check that info logging was called
            mock_logger.info.assert_called()
            
            # Check for specific log messages
            log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("Starting pipeline execution" in msg for msg in log_calls)
            assert any("Pipeline execution completed successfully" in msg for msg in log_calls)
    
    def test_pipeline_run_id_generation(self):
        """Test that pipeline run IDs are generated correctly"""
        result1 = self.pipeline_manager.run_pipeline()
        
        # Create another pipeline manager
        pipeline2 = PipelineManager("test_pipeline_2")
        result2 = pipeline2.run_pipeline()
        
        # Run IDs should be different
        assert result1.run_id != result2.run_id
        assert result1.run_id.startswith("RUN-")
        assert result2.run_id.startswith("RUN-")
    
    def test_pipeline_execution_time_tracking(self):
        """Test that pipeline execution time is tracked correctly"""
        # Add a small delay to ensure measurable execution time
        import time
        
        with patch.object(self.pipeline_manager, '_execute_ingestion_stage') as mock_ingestion:
            mock_ingestion.return_value = {
                'success': False,
                'data': None,
                'total_records': 0,
                'error': 'Test error'
            }
            
            # Add delay in mock
            def delayed_ingestion():
                time.sleep(0.1)  # 100ms delay
                return mock_ingestion.return_value
            
            mock_ingestion.side_effect = delayed_ingestion
            
            result = self.pipeline_manager.run_pipeline()
            
            assert result.execution_time >= 0.1  # Should be at least 100ms
    
    def test_pipeline_stage_error_isolation(self, sample_orders_data):
        """Test that errors in one stage don't affect others inappropriately"""
        # Mock ingestion to succeed
        with patch.object(self.pipeline_manager, '_execute_ingestion_stage') as mock_ingestion:
            mock_ingestion.return_value = {
                'success': True,
                'data': sample_orders_data,
                'total_records': len(sample_orders_data),
                'summary': {}
            }
            
            # Mock transformation to fail
            with patch.object(self.pipeline_manager, '_execute_transformation_stage') as mock_transform:
                mock_transform.side_effect = Exception("Transformation error")
                
                result = self.pipeline_manager.run_pipeline()
                
                # Pipeline should fail, but ingestion should be marked as completed
                assert result.success is False
                assert PipelineStage.INGESTION.value in result.stages_completed
                assert PipelineStage.TRANSFORMATION.value in result.stages_failed
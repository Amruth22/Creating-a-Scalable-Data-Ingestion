"""
Main pipeline execution script
Runs the complete data ingestion pipeline with comprehensive logging and reporting
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root and src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

try:
    from src.pipeline.pipeline_manager import PipelineManager
    from src.utils.config import config
    from src.utils.helpers import ensure_directory_exists, format_duration
except ImportError:
    # Fallback for different import styles
    try:
        from pipeline.pipeline_manager import PipelineManager
        from utils.config import config
        from utils.helpers import ensure_directory_exists, format_duration
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please make sure you're running from the project root directory")
        print("Try: cd Creating-a-Scalable-Data-Ingestion && python scripts/run_pipeline.py")
        sys.exit(1)

def setup_logging(log_level: str = "INFO"):
    """Setup comprehensive logging"""
    # Ensure logs directory exists
    ensure_directory_exists("logs")
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # File handler
    log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

def print_banner():
    """Print pipeline banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           📥 DATA INGESTION PIPELINE EXECUTOR 📥             ║
    ║                                                              ║
    ║              Scalable • Reliable • Comprehensive            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_configuration():
    """Print current configuration"""
    print("🔧 PIPELINE CONFIGURATION")
    print("=" * 50)
    print(f"📊 Database Path: {config.database.path}")
    print(f"🌐 API Base URL: {config.api.base_url}")
    print(f"📁 Input Directory: {config.file.input_dir}")
    print(f"📤 Output Directory: {config.file.output_dir}")
    print(f"📦 Batch Size: {config.pipeline.batch_size}")
    print(f"👥 Max Workers: {config.pipeline.max_workers}")
    print(f"📝 Log Level: {config.pipeline.log_level}")
    print("=" * 50)
    print()

def run_pipeline_with_options(args):
    """Run pipeline with command line options"""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline manager
        pipeline_name = args.name or "data_ingestion_pipeline"
        pipeline = PipelineManager(pipeline_name)
        
        # Configure pipeline based on arguments
        if args.no_files:
            pipeline.enable_file_ingestion = False
            logger.info("🚫 File ingestion disabled")
        
        if args.no_api:
            pipeline.enable_api_ingestion = False
            logger.info("🚫 API ingestion disabled")
        
        if args.no_validation:
            pipeline.enable_validation = False
            logger.info("🚫 Data validation disabled")
        
        if args.no_cleaning:
            pipeline.enable_cleaning = False
            logger.info("🚫 Data cleaning disabled")
        
        if args.no_enrichment:
            pipeline.enable_enrichment = False
            logger.info("🚫 Data enrichment disabled")
        
        if args.no_standardization:
            pipeline.enable_standardization = False
            logger.info("🚫 Data standardization disabled")
        
        if args.no_database:
            pipeline.enable_database_storage = False
            logger.info("🚫 Database storage disabled")
        
        if args.no_export:
            pipeline.enable_file_export = False
            logger.info("🚫 File export disabled")
        
        # Print pipeline configuration
        print("🔄 PIPELINE STAGES")
        print("=" * 50)
        print(f"📁 File Ingestion: {'✅ Enabled' if pipeline.enable_file_ingestion else '❌ Disabled'}")
        print(f"🌐 API Ingestion: {'✅ Enabled' if pipeline.enable_api_ingestion else '❌ Disabled'}")
        print(f"🔍 Data Validation: {'✅ Enabled' if pipeline.enable_validation else '❌ Disabled'}")
        print(f"🧹 Data Cleaning: {'✅ Enabled' if pipeline.enable_cleaning else '❌ Disabled'}")
        print(f"➕ Data Enrichment: {'✅ Enabled' if pipeline.enable_enrichment else '❌ Disabled'}")
        print(f"📏 Data Standardization: {'✅ Enabled' if pipeline.enable_standardization else '❌ Disabled'}")
        print(f"🗄️ Database Storage: {'✅ Enabled' if pipeline.enable_database_storage else '❌ Disabled'}")
        print(f"📤 File Export: {'✅ Enabled' if pipeline.enable_file_export else '❌ Disabled'}")
        print("=" * 50)
        print()
        
        # Run pipeline
        logger.info(f"🚀 Starting pipeline execution: {pipeline_name}")
        start_time = datetime.now()
        
        result = pipeline.run_pipeline()
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Print results
        print_results(result, total_time)
        
        # Generate and save report
        if args.report:
            save_pipeline_report(pipeline, result, args.report)
        
        # Print pipeline status
        if args.status:
            print_pipeline_status(pipeline)
        
        return 0 if result.success else 1
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline execution interrupted by user")
        print("\n⚠️ Pipeline execution was interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def print_results(result, total_time):
    """Print pipeline execution results"""
    print("\n" + "=" * 60)
    print("📊 PIPELINE EXECUTION RESULTS")
    print("=" * 60)
    
    # Status
    status_icon = "🎉" if result.success else "💥"
    status_text = "SUCCESS" if result.success else "FAILED"
    print(f"{status_icon} Status: {status_text}")
    print(f"🆔 Run ID: {result.run_id}")
    print(f"⏱️ Execution Time: {format_duration(result.execution_time)}")
    print(f"📈 Records Processed: {result.total_records_processed:,}")
    print(f"📉 Records Failed: {result.total_records_failed:,}")
    
    if result.total_records_processed > 0:
        success_rate = ((result.total_records_processed - result.total_records_failed) / result.total_records_processed) * 100
        print(f"✅ Success Rate: {success_rate:.1f}%")
    
    # Stages
    print(f"\n🔄 Stages Completed: {len(result.stages_completed)}")
    for stage in result.stages_completed:
        print(f"  ✅ {stage.replace('_', ' ').title()}")
    
    if result.stages_failed:
        print(f"\n❌ Stages Failed: {len(result.stages_failed)}")
        for stage in result.stages_failed:
            print(f"  ❌ {stage.replace('_', ' ').title()}")
    
    # Stage Details
    print(f"\n📋 Stage Details:")
    for stage_name, stage_result in result.stage_results.items():
        status_icon = "✅" if stage_result.get('success', False) else "❌"
        print(f"  {status_icon} {stage_name.title()}: ", end="")
        
        if stage_name == 'ingestion':
            print(f"{stage_result.get('total_records', 0):,} records collected")
        elif stage_name == 'validation':
            print(f"{stage_result.get('quality_score', 0):.1f}% quality score")
        elif stage_name == 'transformation':
            print(f"{stage_result.get('final_records', 0):,} records processed")
        elif stage_name == 'storage':
            print(f"{stage_result.get('records_stored', 0):,} records stored")
        else:
            print("Completed")
    
    # Error details
    if result.error_message:
        print(f"\n❌ Error Details:")
        print(f"   {result.error_message}")
    
    print("=" * 60)

def save_pipeline_report(pipeline, result, report_path):
    """Save detailed pipeline report"""
    try:
        report_content = pipeline.generate_pipeline_report(result)
        
        # Ensure report directory exists
        report_dir = os.path.dirname(report_path) if os.path.dirname(report_path) else "data/output/reports"
        ensure_directory_exists(report_dir)
        
        # Generate filename if not provided
        if not report_path.endswith('.md'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(report_dir, f"pipeline_report_{result.run_id}_{timestamp}.md")
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📄 Pipeline report saved: {report_path}")
        
    except Exception as e:
        print(f"\n⚠️ Failed to save pipeline report: {e}")

def print_pipeline_status(pipeline):
    """Print current pipeline status"""
    try:
        status = pipeline.get_pipeline_status()
        
        print(f"\n📊 PIPELINE STATUS")
        print("=" * 50)
        print(f"Pipeline Name: {status.get('pipeline_name', 'Unknown')}")
        print(f"Current Run ID: {status.get('current_run_id', 'None')}")
        print(f"Timestamp: {status.get('timestamp', 'Unknown')}")
        
        # Recent runs
        recent_runs = status.get('recent_runs', [])
        if recent_runs:
            print(f"\n📈 Recent Runs ({len(recent_runs)}):")
            for run in recent_runs[:5]:  # Show last 5 runs
                run_status = "✅" if run.get('status') == 'completed' else "❌"
                print(f"  {run_status} {run.get('run_id', 'Unknown')} - {run.get('start_time', 'Unknown')}")
        
        # Database stats
        db_stats = status.get('database_stats', {})
        if db_stats:
            print(f"\n🗄️ Database Statistics:")
            print(f"  Orders: {db_stats.get('orders_count', 0):,}")
            print(f"  Pipeline Runs: {db_stats.get('pipeline_runs_count', 0):,}")
            print(f"  Database Size: {db_stats.get('database_size_mb', 0):.2f} MB")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"\n⚠️ Failed to get pipeline status: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Execute the data ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py                    # Run full pipeline
  python scripts/run_pipeline.py --no-api          # Skip API ingestion
  python scripts/run_pipeline.py --report          # Generate report
  python scripts/run_pipeline.py --status          # Show pipeline status
  python scripts/run_pipeline.py --name "daily_run" # Custom pipeline name
        """
    )
    
    # Pipeline options
    parser.add_argument('--name', type=str, help='Pipeline name')
    parser.add_argument('--no-files', action='store_true', help='Disable file ingestion')
    parser.add_argument('--no-api', action='store_true', help='Disable API ingestion')
    parser.add_argument('--no-validation', action='store_true', help='Disable data validation')
    parser.add_argument('--no-cleaning', action='store_true', help='Disable data cleaning')
    parser.add_argument('--no-enrichment', action='store_true', help='Disable data enrichment')
    parser.add_argument('--no-standardization', action='store_true', help='Disable data standardization')
    parser.add_argument('--no-database', action='store_true', help='Disable database storage')
    parser.add_argument('--no-export', action='store_true', help='Disable file export')
    
    # Output options
    parser.add_argument('--report', type=str, nargs='?', const='auto', help='Generate detailed report')
    parser.add_argument('--status', action='store_true', help='Show pipeline status')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Print banner
    print_banner()
    
    # Print configuration
    print_configuration()
    
    logger.info(f"Pipeline execution started - Log file: {log_file}")
    
    try:
        # Run pipeline
        exit_code = run_pipeline_with_options(args)
        
        logger.info(f"Pipeline execution completed with exit code: {exit_code}")
        
        print(f"\n📝 Full execution log saved to: {log_file}")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n💥 Unexpected error: {e}")
        print(f"📝 Check log file for details: {log_file}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
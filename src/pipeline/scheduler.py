"""
Pipeline scheduler module for automated job scheduling and execution
Provides cron-like functionality for running data ingestion pipelines on schedule
"""

import schedule
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import os
from concurrent.futures import ThreadPoolExecutor
import uuid

from .pipeline_manager import PipelineManager
from ..utils.config import config
from ..utils.helpers import ensure_directory_exists, format_duration
from ..utils.constants import ProcessingStatus, PipelineStage

# Configure logging
logger = logging.getLogger(__name__)

class ScheduleType(Enum):
    """Schedule type enumeration"""
    ONCE = "once"
    DAILY = "daily"
    HOURLY = "hourly"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CRON = "cron"
    INTERVAL = "interval"

class JobStatus(Enum):
    """Job status enumeration"""
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class ScheduledJob:
    """Scheduled job container"""
    job_id: str
    name: str
    pipeline_name: str
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    pipeline_config: Dict[str, Any]
    status: JobStatus
    created_at: str
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    enabled: bool = True
    max_retries: int = 3
    retry_delay: int = 300  # 5 minutes
    timeout: int = 3600  # 1 hour
    tags: List[str] = None

@dataclass
class JobExecution:
    """Job execution result container"""
    job_id: str
    execution_id: str
    pipeline_run_id: str
    start_time: str
    end_time: Optional[str]
    duration: float
    status: JobStatus
    records_processed: int
    error_message: Optional[str] = None
    retry_count: int = 0

class PipelineScheduler:
    """Comprehensive pipeline scheduler with cron-like functionality"""
    
    def __init__(self, scheduler_name: str = "pipeline_scheduler"):
        """
        Initialize pipeline scheduler
        
        Args:
            scheduler_name (str): Name of the scheduler instance
        """
        self.scheduler_name = scheduler_name
        self.jobs: Dict[str, ScheduledJob] = {}
        self.job_executions: List[JobExecution] = []
        self.is_running = False
        self.scheduler_thread = None
        self.executor = ThreadPoolExecutor(max_workers=config.pipeline.max_workers)
        
        # Scheduler configuration
        self.jobs_file = "data/scheduler_jobs.json"
        self.executions_file = "data/scheduler_executions.json"
        self.max_execution_history = 1000
        
        # Ensure data directory exists
        ensure_directory_exists("data")
        
        # Load existing jobs and executions
        self._load_jobs()
        self._load_executions()
        
        logger.info(f"Pipeline scheduler initialized: {scheduler_name}")
    
    def create_job(self, 
                   name: str,
                   pipeline_name: str,
                   schedule_type: ScheduleType,
                   schedule_config: Dict[str, Any],
                   pipeline_config: Optional[Dict[str, Any]] = None,
                   **kwargs) -> str:
        """
        Create a new scheduled job
        
        Args:
            name (str): Job name
            pipeline_name (str): Pipeline to execute
            schedule_type (ScheduleType): Type of schedule
            schedule_config (Dict): Schedule configuration
            pipeline_config (Dict, optional): Pipeline configuration
            **kwargs: Additional job parameters
            
        Returns:
            str: Job ID
        """
        job_id = f"JOB-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{str(uuid.uuid4())[:8]}"
        
        # Validate schedule configuration
        self._validate_schedule_config(schedule_type, schedule_config)
        
        # Create job
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            pipeline_name=pipeline_name,
            schedule_type=schedule_type,
            schedule_config=schedule_config,
            pipeline_config=pipeline_config or {},
            status=JobStatus.SCHEDULED,
            created_at=datetime.now().isoformat(),
            max_retries=kwargs.get('max_retries', 3),
            retry_delay=kwargs.get('retry_delay', 300),
            timeout=kwargs.get('timeout', 3600),
            tags=kwargs.get('tags', [])
        )
        
        # Calculate next run time
        job.next_run = self._calculate_next_run(job)
        
        # Store job
        self.jobs[job_id] = job
        self._save_jobs()
        
        # Schedule the job
        self._schedule_job(job)
        
        logger.info(f"Created scheduled job: {job_id} - {name}")
        logger.info(f"Next run: {job.next_run}")
        
        return job_id
    
    def create_daily_job(self, 
                        name: str, 
                        pipeline_name: str, 
                        time_str: str = "02:00",
                        **kwargs) -> str:
        """
        Create a daily scheduled job
        
        Args:
            name (str): Job name
            pipeline_name (str): Pipeline to execute
            time_str (str): Time to run (HH:MM format)
            **kwargs: Additional job parameters
            
        Returns:
            str: Job ID
        """
        schedule_config = {"time": time_str}
        return self.create_job(
            name=name,
            pipeline_name=pipeline_name,
            schedule_type=ScheduleType.DAILY,
            schedule_config=schedule_config,
            **kwargs
        )
    
    def create_hourly_job(self, 
                         name: str, 
                         pipeline_name: str, 
                         minute: int = 0,
                         **kwargs) -> str:
        """
        Create an hourly scheduled job
        
        Args:
            name (str): Job name
            pipeline_name (str): Pipeline to execute
            minute (int): Minute of the hour to run (0-59)
            **kwargs: Additional job parameters
            
        Returns:
            str: Job ID
        """
        schedule_config = {"minute": minute}
        return self.create_job(
            name=name,
            pipeline_name=pipeline_name,
            schedule_type=ScheduleType.HOURLY,
            schedule_config=schedule_config,
            **kwargs
        )
    
    def create_interval_job(self, 
                           name: str, 
                           pipeline_name: str, 
                           interval_minutes: int,
                           **kwargs) -> str:
        """
        Create an interval-based scheduled job
        
        Args:
            name (str): Job name
            pipeline_name (str): Pipeline to execute
            interval_minutes (int): Interval in minutes
            **kwargs: Additional job parameters
            
        Returns:
            str: Job ID
        """
        schedule_config = {"interval_minutes": interval_minutes}
        return self.create_job(
            name=name,
            pipeline_name=pipeline_name,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config=schedule_config,
            **kwargs
        )
    
    def create_weekly_job(self, 
                         name: str, 
                         pipeline_name: str, 
                         day: str = "monday", 
                         time_str: str = "02:00",
                         **kwargs) -> str:
        """
        Create a weekly scheduled job
        
        Args:
            name (str): Job name
            pipeline_name (str): Pipeline to execute
            day (str): Day of week (monday, tuesday, etc.)
            time_str (str): Time to run (HH:MM format)
            **kwargs: Additional job parameters
            
        Returns:
            str: Job ID
        """
        schedule_config = {"day": day.lower(), "time": time_str}
        return self.create_job(
            name=name,
            pipeline_name=pipeline_name,
            schedule_type=ScheduleType.WEEKLY,
            schedule_config=schedule_config,
            **kwargs
        )
    
    def create_cron_job(self, 
                       name: str, 
                       pipeline_name: str, 
                       cron_expression: str,
                       **kwargs) -> str:
        """
        Create a cron-style scheduled job
        
        Args:
            name (str): Job name
            pipeline_name (str): Pipeline to execute
            cron_expression (str): Cron expression (minute hour day month weekday)
            **kwargs: Additional job parameters
            
        Returns:
            str: Job ID
        """
        schedule_config = {"cron_expression": cron_expression}
        return self.create_job(
            name=name,
            pipeline_name=pipeline_name,
            schedule_type=ScheduleType.CRON,
            schedule_config=schedule_config,
            **kwargs
        )
    
    def _validate_schedule_config(self, schedule_type: ScheduleType, config: Dict[str, Any]):
        """Validate schedule configuration"""
        if schedule_type == ScheduleType.DAILY:
            if "time" not in config:
                raise ValueError("Daily schedule requires 'time' parameter")
            time_str = config["time"]
            try:
                datetime.strptime(time_str, "%H:%M")
            except ValueError:
                raise ValueError(f"Invalid time format: {time_str}. Use HH:MM format")
        
        elif schedule_type == ScheduleType.HOURLY:
            if "minute" not in config:
                config["minute"] = 0
            minute = config["minute"]
            if not 0 <= minute <= 59:
                raise ValueError(f"Minute must be between 0 and 59, got {minute}")
        
        elif schedule_type == ScheduleType.WEEKLY:
            if "day" not in config:
                raise ValueError("Weekly schedule requires 'day' parameter")
            if "time" not in config:
                raise ValueError("Weekly schedule requires 'time' parameter")
            
            valid_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            if config["day"].lower() not in valid_days:
                raise ValueError(f"Invalid day: {config['day']}. Must be one of {valid_days}")
        
        elif schedule_type == ScheduleType.INTERVAL:
            if "interval_minutes" not in config:
                raise ValueError("Interval schedule requires 'interval_minutes' parameter")
            interval = config["interval_minutes"]
            if interval <= 0:
                raise ValueError(f"Interval must be positive, got {interval}")
        
        elif schedule_type == ScheduleType.CRON:
            if "cron_expression" not in config:
                raise ValueError("Cron schedule requires 'cron_expression' parameter")
            # Basic cron validation (could be enhanced)
            cron_parts = config["cron_expression"].split()
            if len(cron_parts) != 5:
                raise ValueError("Cron expression must have 5 parts: minute hour day month weekday")
    
    def _calculate_next_run(self, job: ScheduledJob) -> str:
        """Calculate next run time for a job"""
        now = datetime.now()
        
        if job.schedule_type == ScheduleType.DAILY:
            time_str = job.schedule_config["time"]
            hour, minute = map(int, time_str.split(":"))
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        
        elif job.schedule_type == ScheduleType.HOURLY:
            minute = job.schedule_config["minute"]
            next_run = now.replace(minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(hours=1)
        
        elif job.schedule_type == ScheduleType.WEEKLY:
            day_name = job.schedule_config["day"]
            time_str = job.schedule_config["time"]
            hour, minute = map(int, time_str.split(":"))
            
            days_ahead = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].index(day_name) - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        elif job.schedule_type == ScheduleType.INTERVAL:
            interval_minutes = job.schedule_config["interval_minutes"]
            next_run = now + timedelta(minutes=interval_minutes)
        
        elif job.schedule_type == ScheduleType.ONCE:
            # For one-time jobs, run immediately
            next_run = now + timedelta(seconds=1)
        
        else:
            # Default to 1 hour from now
            next_run = now + timedelta(hours=1)
        
        return next_run.isoformat()
    
    def _schedule_job(self, job: ScheduledJob):
        """Schedule a job using the schedule library"""
        if not job.enabled:
            return
        
        if job.schedule_type == ScheduleType.DAILY:
            time_str = job.schedule_config["time"]
            schedule.every().day.at(time_str).do(self._execute_job_wrapper, job.job_id).tag(job.job_id)
        
        elif job.schedule_type == ScheduleType.HOURLY:
            minute = job.schedule_config["minute"]
            schedule.every().hour.at(f":{minute:02d}").do(self._execute_job_wrapper, job.job_id).tag(job.job_id)
        
        elif job.schedule_type == ScheduleType.WEEKLY:
            day_name = job.schedule_config["day"]
            time_str = job.schedule_config["time"]
            getattr(schedule.every(), day_name).at(time_str).do(self._execute_job_wrapper, job.job_id).tag(job.job_id)
        
        elif job.schedule_type == ScheduleType.INTERVAL:
            interval_minutes = job.schedule_config["interval_minutes"]
            schedule.every(interval_minutes).minutes.do(self._execute_job_wrapper, job.job_id).tag(job.job_id)
        
        elif job.schedule_type == ScheduleType.ONCE:
            # Schedule to run once immediately
            schedule.every().second.do(self._execute_job_wrapper, job.job_id).tag(job.job_id)
    
    def _execute_job_wrapper(self, job_id: str):
        """Wrapper for job execution with error handling"""
        try:
            self._execute_job(job_id)
        except Exception as e:
            logger.error(f"Error executing job {job_id}: {e}")
        
        # For one-time jobs, remove from schedule after execution
        job = self.jobs.get(job_id)
        if job and job.schedule_type == ScheduleType.ONCE:
            schedule.clear(job_id)
    
    def _execute_job(self, job_id: str):
        """Execute a scheduled job"""
        job = self.jobs.get(job_id)
        if not job or not job.enabled:
            return
        
        # Check if job is already running
        if job.status == JobStatus.RUNNING:
            logger.warning(f"Job {job_id} is already running, skipping execution")
            return
        
        execution_id = f"EXEC-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{str(uuid.uuid4())[:8]}"
        start_time = datetime.now()
        
        logger.info(f"🚀 Starting scheduled job execution: {job.name} ({job_id})")
        
        # Update job status
        job.status = JobStatus.RUNNING
        job.last_run = start_time.isoformat()
        job.run_count += 1
        
        # Create execution record
        execution = JobExecution(
            job_id=job_id,
            execution_id=execution_id,
            pipeline_run_id="",
            start_time=start_time.isoformat(),
            end_time=None,
            duration=0.0,
            status=JobStatus.RUNNING,
            records_processed=0
        )
        
        try:
            # Execute pipeline
            pipeline_manager = PipelineManager(job.pipeline_name)
            
            # Apply job-specific pipeline configuration
            if job.pipeline_config:
                for key, value in job.pipeline_config.items():
                    if hasattr(pipeline_manager, key):
                        setattr(pipeline_manager, key, value)
            
            # Run pipeline with timeout
            future = self.executor.submit(pipeline_manager.run_pipeline)
            
            try:
                pipeline_result = future.result(timeout=job.timeout)
                execution.pipeline_run_id = pipeline_result.run_id
                execution.records_processed = pipeline_result.total_records_processed
                
                if pipeline_result.success:
                    execution.status = JobStatus.COMPLETED
                    job.status = JobStatus.COMPLETED
                    job.success_count += 1
                    logger.info(f"✅ Job completed successfully: {job.name}")
                else:
                    execution.status = JobStatus.FAILED
                    job.status = JobStatus.FAILED
                    job.failure_count += 1
                    execution.error_message = pipeline_result.error_message
                    logger.error(f"❌ Job failed: {job.name} - {pipeline_result.error_message}")
                
            except TimeoutError:
                execution.status = JobStatus.FAILED
                job.status = JobStatus.FAILED
                job.failure_count += 1
                execution.error_message = f"Job timed out after {job.timeout} seconds"
                logger.error(f"⏰ Job timed out: {job.name}")
                future.cancel()
        
        except Exception as e:
            execution.status = JobStatus.FAILED
            job.status = JobStatus.FAILED
            job.failure_count += 1
            execution.error_message = str(e)
            logger.error(f"💥 Job execution error: {job.name} - {e}")
        
        finally:
            # Update execution record
            end_time = datetime.now()
            execution.end_time = end_time.isoformat()
            execution.duration = (end_time - start_time).total_seconds()
            
            # Store execution
            self.job_executions.append(execution)
            self._save_executions()
            
            # Update job for next run
            if job.schedule_type != ScheduleType.ONCE:
                job.status = JobStatus.SCHEDULED
                job.next_run = self._calculate_next_run(job)
            
            # Save job updates
            self._save_jobs()
            
            logger.info(f"📊 Job execution completed: {job.name} - Duration: {format_duration(execution.duration)}")
    
    def start_scheduler(self):
        """Start the scheduler in a background thread"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        
        # Schedule all existing jobs
        for job in self.jobs.values():
            if job.enabled and job.status != JobStatus.RUNNING:
                self._schedule_job(job)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"🚀 Pipeline scheduler started: {len(self.jobs)} jobs scheduled")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("⏹️ Pipeline scheduler stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        logger.info("📅 Scheduler loop started")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)
        
        logger.info("📅 Scheduler loop stopped")
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a scheduled job"""
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return False
        
        job.enabled = False
        job.status = JobStatus.PAUSED
        schedule.clear(job_id)
        self._save_jobs()
        
        logger.info(f"⏸️ Job paused: {job.name}")
        return True
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job"""
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return False
        
        job.enabled = True
        job.status = JobStatus.SCHEDULED
        job.next_run = self._calculate_next_run(job)
        
        if self.is_running:
            self._schedule_job(job)
        
        self._save_jobs()
        
        logger.info(f"▶️ Job resumed: {job.name}")
        return True
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a scheduled job"""
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job not found: {job_id}")
            return False
        
        # Remove from schedule
        schedule.clear(job_id)
        
        # Remove from jobs
        del self.jobs[job_id]
        self._save_jobs()
        
        logger.info(f"🗑️ Job deleted: {job.name}")
        return True
    
    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def list_jobs(self, status_filter: Optional[JobStatus] = None, 
                  tag_filter: Optional[str] = None) -> List[ScheduledJob]:
        """List all jobs with optional filtering"""
        jobs = list(self.jobs.values())
        
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        
        if tag_filter:
            jobs = [job for job in jobs if tag_filter in (job.tags or [])]
        
        return jobs
    
    def get_job_executions(self, job_id: Optional[str] = None, 
                          limit: int = 50) -> List[JobExecution]:
        """Get job execution history"""
        executions = self.job_executions
        
        if job_id:
            executions = [exec for exec in executions if exec.job_id == job_id]
        
        # Sort by start time (most recent first)
        executions.sort(key=lambda x: x.start_time, reverse=True)
        
        return executions[:limit]
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics"""
        total_jobs = len(self.jobs)
        active_jobs = len([job for job in self.jobs.values() if job.enabled])
        running_jobs = len([job for job in self.jobs.values() if job.status == JobStatus.RUNNING])
        
        # Recent executions (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_executions = [
            exec for exec in self.job_executions 
            if datetime.fromisoformat(exec.start_time) > recent_cutoff
        ]
        
        successful_recent = len([exec for exec in recent_executions if exec.status == JobStatus.COMPLETED])
        failed_recent = len([exec for exec in recent_executions if exec.status == JobStatus.FAILED])
        
        return {
            'scheduler_name': self.scheduler_name,
            'is_running': self.is_running,
            'total_jobs': total_jobs,
            'active_jobs': active_jobs,
            'running_jobs': running_jobs,
            'paused_jobs': len([job for job in self.jobs.values() if job.status == JobStatus.PAUSED]),
            'total_executions': len(self.job_executions),
            'recent_executions_24h': len(recent_executions),
            'successful_executions_24h': successful_recent,
            'failed_executions_24h': failed_recent,
            'success_rate_24h': (successful_recent / len(recent_executions) * 100) if recent_executions else 0,
            'next_scheduled_jobs': self._get_next_scheduled_jobs(5),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_next_scheduled_jobs(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get next scheduled jobs"""
        scheduled_jobs = [
            {
                'job_id': job.job_id,
                'name': job.name,
                'next_run': job.next_run,
                'schedule_type': job.schedule_type.value
            }
            for job in self.jobs.values()
            if job.enabled and job.next_run and job.status == JobStatus.SCHEDULED
        ]
        
        # Sort by next run time
        scheduled_jobs.sort(key=lambda x: x['next_run'])
        
        return scheduled_jobs[:limit]
    
    def _load_jobs(self):
        """Load jobs from file"""
        try:
            if os.path.exists(self.jobs_file):
                with open(self.jobs_file, 'r') as f:
                    jobs_data = json.load(f)
                
                for job_data in jobs_data:
                    job = ScheduledJob(
                        job_id=job_data['job_id'],
                        name=job_data['name'],
                        pipeline_name=job_data['pipeline_name'],
                        schedule_type=ScheduleType(job_data['schedule_type']),
                        schedule_config=job_data['schedule_config'],
                        pipeline_config=job_data.get('pipeline_config', {}),
                        status=JobStatus(job_data['status']),
                        created_at=job_data['created_at'],
                        last_run=job_data.get('last_run'),
                        next_run=job_data.get('next_run'),
                        run_count=job_data.get('run_count', 0),
                        success_count=job_data.get('success_count', 0),
                        failure_count=job_data.get('failure_count', 0),
                        enabled=job_data.get('enabled', True),
                        max_retries=job_data.get('max_retries', 3),
                        retry_delay=job_data.get('retry_delay', 300),
                        timeout=job_data.get('timeout', 3600),
                        tags=job_data.get('tags', [])
                    )
                    self.jobs[job.job_id] = job
                
                logger.info(f"Loaded {len(self.jobs)} scheduled jobs")
        
        except Exception as e:
            logger.error(f"Error loading jobs: {e}")
    
    def _save_jobs(self):
        """Save jobs to file"""
        try:
            jobs_data = []
            for job in self.jobs.values():
                jobs_data.append({
                    'job_id': job.job_id,
                    'name': job.name,
                    'pipeline_name': job.pipeline_name,
                    'schedule_type': job.schedule_type.value,
                    'schedule_config': job.schedule_config,
                    'pipeline_config': job.pipeline_config,
                    'status': job.status.value,
                    'created_at': job.created_at,
                    'last_run': job.last_run,
                    'next_run': job.next_run,
                    'run_count': job.run_count,
                    'success_count': job.success_count,
                    'failure_count': job.failure_count,
                    'enabled': job.enabled,
                    'max_retries': job.max_retries,
                    'retry_delay': job.retry_delay,
                    'timeout': job.timeout,
                    'tags': job.tags
                })
            
            with open(self.jobs_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving jobs: {e}")
    
    def _load_executions(self):
        """Load execution history from file"""
        try:
            if os.path.exists(self.executions_file):
                with open(self.executions_file, 'r') as f:
                    executions_data = json.load(f)
                
                for exec_data in executions_data:
                    execution = JobExecution(
                        job_id=exec_data['job_id'],
                        execution_id=exec_data['execution_id'],
                        pipeline_run_id=exec_data['pipeline_run_id'],
                        start_time=exec_data['start_time'],
                        end_time=exec_data.get('end_time'),
                        duration=exec_data['duration'],
                        status=JobStatus(exec_data['status']),
                        records_processed=exec_data['records_processed'],
                        error_message=exec_data.get('error_message'),
                        retry_count=exec_data.get('retry_count', 0)
                    )
                    self.job_executions.append(execution)
                
                logger.info(f"Loaded {len(self.job_executions)} job executions")
        
        except Exception as e:
            logger.error(f"Error loading executions: {e}")
    
    def _save_executions(self):
        """Save execution history to file"""
        try:
            # Keep only recent executions to prevent file from growing too large
            if len(self.job_executions) > self.max_execution_history:
                self.job_executions = self.job_executions[-self.max_execution_history:]
            
            executions_data = []
            for execution in self.job_executions:
                executions_data.append({
                    'job_id': execution.job_id,
                    'execution_id': execution.execution_id,
                    'pipeline_run_id': execution.pipeline_run_id,
                    'start_time': execution.start_time,
                    'end_time': execution.end_time,
                    'duration': execution.duration,
                    'status': execution.status.value,
                    'records_processed': execution.records_processed,
                    'error_message': execution.error_message,
                    'retry_count': execution.retry_count
                })
            
            with open(self.executions_file, 'w') as f:
                json.dump(executions_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving executions: {e}")
    
    def generate_scheduler_report(self) -> str:
        """Generate comprehensive scheduler report"""
        status = self.get_scheduler_status()
        
        report = []
        report.append("# Pipeline Scheduler Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Scheduler**: {status['scheduler_name']}")
        report.append(f"- **Status**: {'🟢 Running' if status['is_running'] else '🔴 Stopped'}")
        report.append(f"- **Total Jobs**: {status['total_jobs']}")
        report.append(f"- **Active Jobs**: {status['active_jobs']}")
        report.append(f"- **Running Jobs**: {status['running_jobs']}")
        report.append(f"- **Paused Jobs**: {status['paused_jobs']}")
        report.append("")
        
        # Performance (last 24 hours)
        report.append("## Performance (Last 24 Hours)")
        report.append(f"- **Total Executions**: {status['recent_executions_24h']}")
        report.append(f"- **Successful**: {status['successful_executions_24h']}")
        report.append(f"- **Failed**: {status['failed_executions_24h']}")
        report.append(f"- **Success Rate**: {status['success_rate_24h']:.1f}%")
        report.append("")
        
        # Next scheduled jobs
        if status['next_scheduled_jobs']:
            report.append("## Next Scheduled Jobs")
            for job in status['next_scheduled_jobs']:
                next_run = datetime.fromisoformat(job['next_run']).strftime('%Y-%m-%d %H:%M:%S')
                report.append(f"- **{job['name']}** ({job['schedule_type']}) - {next_run}")
            report.append("")
        
        # Job details
        report.append("## Job Details")
        for job in self.jobs.values():
            status_icon = {
                JobStatus.SCHEDULED: "📅",
                JobStatus.RUNNING: "🔄",
                JobStatus.COMPLETED: "✅",
                JobStatus.FAILED: "❌",
                JobStatus.CANCELLED: "⏹️",
                JobStatus.PAUSED: "⏸️"
            }.get(job.status, "❓")
            
            report.append(f"### {status_icon} {job.name}")
            report.append(f"- **ID**: {job.job_id}")
            report.append(f"- **Pipeline**: {job.pipeline_name}")
            report.append(f"- **Schedule**: {job.schedule_type.value}")
            report.append(f"- **Status**: {job.status.value}")
            report.append(f"- **Runs**: {job.run_count} (Success: {job.success_count}, Failed: {job.failure_count})")
            if job.next_run:
                next_run = datetime.fromisoformat(job.next_run).strftime('%Y-%m-%d %H:%M:%S')
                report.append(f"- **Next Run**: {next_run}")
            report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test scheduler
    print("Testing Pipeline Scheduler...")
    
    # Initialize scheduler
    scheduler = PipelineScheduler("test_scheduler")
    
    # Create test jobs
    daily_job_id = scheduler.create_daily_job(
        name="Daily Data Processing",
        pipeline_name="daily_pipeline",
        time_str="02:00"
    )
    
    hourly_job_id = scheduler.create_hourly_job(
        name="Hourly Data Sync",
        pipeline_name="sync_pipeline",
        minute=30
    )
    
    interval_job_id = scheduler.create_interval_job(
        name="Frequent Updates",
        pipeline_name="update_pipeline",
        interval_minutes=15
    )
    
    print(f"Created jobs: {daily_job_id}, {hourly_job_id}, {interval_job_id}")
    
    # Get scheduler status
    status = scheduler.get_scheduler_status()
    print(f"Scheduler status: {status['total_jobs']} jobs, {status['active_jobs']} active")
    
    # Generate report
    report = scheduler.generate_scheduler_report()
    print(f"Generated scheduler report ({len(report)} characters)")
    
    # List jobs
    jobs = scheduler.list_jobs()
    print(f"Listed {len(jobs)} jobs")
    
    print("Pipeline scheduler test completed!")
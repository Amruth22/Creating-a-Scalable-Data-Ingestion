"""
Advanced alerting system for data pipeline monitoring
Provides multi-channel notifications, escalation, and alert management
"""

import smtplib
import json
import requests
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from enum import Enum
import threading
import time
from collections import defaultdict

from ..utils.helpers import ensure_directory_exists
from .logger import get_logger
from .metrics import MetricsCollector

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    escalation_count: int = 0

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_minutes: int = 5
    escalation_minutes: int = 30
    max_escalations: int = 3
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    name: str
    type: str  # email, slack, webhook, sms, pagerduty
    config: Dict[str, Any]
    enabled: bool = True
    severity_threshold: AlertSeverity = AlertSeverity.LOW

class AlertManager:
    """Comprehensive alert management system"""
    
    def __init__(self, db_path: str = "data/alerts.db", 
                 metrics_collector: MetricsCollector = None):
        self.db_path = db_path
        self.metrics = metrics_collector
        self.logger = get_logger("AlertManager")
        
        # Alert storage
        self.active_alerts = {}
        self.alert_rules = {}
        self.notification_channels = {}
        self.alert_history = []
        
        # Cooldown tracking
        self.rule_cooldowns = defaultdict(datetime)
        self.escalation_timers = {}
        
        # Background processing
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        # Initialize database
        self._init_database()
        
        # Load default rules and channels
        self._setup_default_rules()
        self._setup_default_channels()
        
        # Start background processing
        self.start_processing()
        
        self.logger.info("Alert manager initialized", db_path=db_path)
    
    def _init_database(self):
        """Initialize alerts database"""
        ensure_directory_exists(os.path.dirname(self.db_path))
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT,
                    resolved_at TEXT,
                    escalated BOOLEAN DEFAULT FALSE,
                    escalation_count INTEGER DEFAULT 0,
                    tags TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Alert notifications table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    channel_name TEXT NOT NULL,
                    channel_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (alert_id) REFERENCES alerts (id)
                )
            ''')
            
            # Alert rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    name TEXT PRIMARY KEY,
                    severity TEXT NOT NULL,
                    message_template TEXT NOT NULL,
                    cooldown_minutes INTEGER NOT NULL,
                    escalation_minutes INTEGER NOT NULL,
                    max_escalations INTEGER NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    tags TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="pipeline_failure",
                condition=lambda metrics: metrics.get("pipeline_success", True) == False,
                severity=AlertSeverity.CRITICAL,
                message_template="Pipeline {pipeline_id} failed: {error_message}",
                cooldown_minutes=5,
                escalation_minutes=15,
                max_escalations=3
            ),
            AlertRule(
                name="high_execution_time",
                condition=lambda metrics: metrics.get("execution_time", 0) > 300,
                severity=AlertSeverity.HIGH,
                message_template="Pipeline {pipeline_id} execution time {execution_time:.1f}s exceeds threshold",
                cooldown_minutes=10,
                escalation_minutes=30,
                max_escalations=2
            ),
            AlertRule(
                name="low_data_quality",
                condition=lambda metrics: metrics.get("data_quality_score", 100) < 80,
                severity=AlertSeverity.HIGH,
                message_template="Data quality score {data_quality_score:.1f}% below threshold for pipeline {pipeline_id}",
                cooldown_minutes=5,
                escalation_minutes=20,
                max_escalations=2
            ),
            AlertRule(
                name="low_throughput",
                condition=lambda metrics: metrics.get("throughput", float('inf')) < 10,
                severity=AlertSeverity.MEDIUM,
                message_template="Pipeline {pipeline_id} throughput {throughput:.1f} records/s below threshold",
                cooldown_minutes=15,
                escalation_minutes=60,
                max_escalations=1
            ),
            AlertRule(
                name="high_error_rate",
                condition=lambda metrics: metrics.get("error_rate", 0) > 0.05,
                severity=AlertSeverity.HIGH,
                message_template="Error rate {error_rate:.1%} exceeds threshold for pipeline {pipeline_id}",
                cooldown_minutes=5,
                escalation_minutes=20,
                max_escalations=2
            ),
            AlertRule(
                name="system_resource_exhaustion",
                condition=lambda metrics: (
                    metrics.get("cpu_usage", 0) > 90 or 
                    metrics.get("memory_usage", 0) > 90 or
                    metrics.get("disk_usage", 0) > 95
                ),
                severity=AlertSeverity.CRITICAL,
                message_template="System resource exhaustion: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%, Disk {disk_usage:.1f}%",
                cooldown_minutes=2,
                escalation_minutes=10,
                max_escalations=3
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def _setup_default_channels(self):
        """Setup default notification channels"""
        # Email channel
        self.add_notification_channel(NotificationChannel(
            name="email_alerts",
            type="email",
            config={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",  # To be configured
                "password": "",  # To be configured
                "from_email": "",
                "to_emails": []
            },
            enabled=False,  # Disabled by default until configured
            severity_threshold=AlertSeverity.MEDIUM
        ))
        
        # Slack channel
        self.add_notification_channel(NotificationChannel(
            name="slack_alerts",
            type="slack",
            config={
                "webhook_url": "",  # To be configured
                "channel": "#alerts",
                "username": "Pipeline Alert Bot"
            },
            enabled=False,  # Disabled by default until configured
            severity_threshold=AlertSeverity.LOW
        ))
        
        # Webhook channel
        self.add_notification_channel(NotificationChannel(
            name="webhook_alerts",
            type="webhook",
            config={
                "url": "",  # To be configured
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "timeout": 30
            },
            enabled=False,  # Disabled by default until configured
            severity_threshold=AlertSeverity.MEDIUM
        ))
        
        # Console channel (always enabled for testing)
        self.add_notification_channel(NotificationChannel(
            name="console_alerts",
            type="console",
            config={},
            enabled=True,
            severity_threshold=AlertSeverity.LOW
        ))
    
    def add_alert_rule(self, rule: AlertRule):
        """Add or update an alert rule"""
        self.alert_rules[rule.name] = rule
        
        # Persist to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alert_rules 
                    (name, severity, message_template, cooldown_minutes, escalation_minutes, 
                     max_escalations, enabled, tags, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rule.name, rule.severity.value, rule.message_template,
                    rule.cooldown_minutes, rule.escalation_minutes, rule.max_escalations,
                    rule.enabled, json.dumps(rule.tags), datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error persisting alert rule {rule.name}: {e}")
        
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add or update a notification channel"""
        self.notification_channels[channel.name] = channel
        self.logger.info(f"Added notification channel: {channel.name} ({channel.type})")
    
    def evaluate_metrics(self, metrics: Dict[str, Any], source: str = "pipeline"):
        """Evaluate metrics against alert rules"""
        triggered_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown
                if self._is_in_cooldown(rule_name):
                    continue
                
                # Evaluate condition
                if rule.condition(metrics):
                    alert = self._create_alert(rule, metrics, source)
                    triggered_alerts.append(alert)
                    
                    # Set cooldown
                    self.rule_cooldowns[rule_name] = datetime.now()
                    
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule_name}: {e}")
        
        return triggered_alerts
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if rule is in cooldown period"""
        if rule_name not in self.rule_cooldowns:
            return False
        
        rule = self.alert_rules[rule_name]
        cooldown_end = self.rule_cooldowns[rule_name] + timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def _create_alert(self, rule: AlertRule, metrics: Dict[str, Any], source: str) -> Alert:
        """Create alert from rule and metrics"""
        alert_id = f"{rule.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(metrics)) % 10000}"
        
        # Format message
        try:
            message = rule.message_template.format(**metrics)
        except KeyError as e:
            message = f"{rule.message_template} (Missing key: {e})"
        
        alert = Alert(
            id=alert_id,
            name=rule.name,
            severity=rule.severity,
            message=message,
            source=source,
            timestamp=datetime.now(),
            tags=rule.tags.copy(),
            metadata=metrics.copy()
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Persist to database
        self._persist_alert(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        # Schedule escalation if needed
        if rule.escalation_minutes > 0:
            self._schedule_escalation(alert, rule)
        
        self.logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
        
        return alert
    
    def _persist_alert(self, alert: Alert):
        """Persist alert to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts 
                    (id, name, severity, message, source, status, timestamp, 
                     escalated, escalation_count, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id, alert.name, alert.severity.value, alert.message,
                    alert.source, alert.status.value, alert.timestamp.isoformat(),
                    alert.escalated, alert.escalation_count,
                    json.dumps(alert.tags), json.dumps(alert.metadata, default=str)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error persisting alert {alert.id}: {e}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for alert"""
        severity_levels = {
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4
        }
        
        alert_level = severity_levels[alert.severity]
        
        for channel_name, channel in self.notification_channels.items():
            if not channel.enabled:
                continue
            
            threshold_level = severity_levels[channel.severity_threshold]
            
            if alert_level >= threshold_level:
                self._send_notification(alert, channel)
    
    def _send_notification(self, alert: Alert, channel: NotificationChannel):
        """Send notification to specific channel"""
        try:
            if channel.type == "email":
                success = self._send_email_notification(alert, channel)
            elif channel.type == "slack":
                success = self._send_slack_notification(alert, channel)
            elif channel.type == "webhook":
                success = self._send_webhook_notification(alert, channel)
            elif channel.type == "console":
                success = self._send_console_notification(alert, channel)
            else:
                self.logger.warning(f"Unknown notification channel type: {channel.type}")
                success = False
            
            # Log notification attempt
            self._log_notification(alert.id, channel.name, channel.type, 
                                 "SUCCESS" if success else "FAILED")
            
        except Exception as e:
            self.logger.error(f"Error sending notification via {channel.name}: {e}")
            self._log_notification(alert.id, channel.name, channel.type, "ERROR", str(e))
    
    def _send_email_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send email notification"""
        config = channel.config
        
        if not all([config.get("smtp_server"), config.get("username"), 
                   config.get("password"), config.get("from_email"), 
                   config.get("to_emails")]):
            self.logger.warning("Email channel not fully configured")
            return False
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = config["from_email"]
            msg['To'] = ", ".join(config["to_emails"])
            msg['Subject'] = f"[{alert.severity.value}] Pipeline Alert: {alert.name}"
            
            # Email body
            body = f"""
Alert Details:
- Alert ID: {alert.id}
- Severity: {alert.severity.value}
- Source: {alert.source}
- Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- Message: {alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2, default=str)}

Tags:
{json.dumps(alert.tags, indent=2)}

This is an automated alert from the Data Pipeline Monitoring System.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config["smtp_server"], config.get("smtp_port", 587))
            server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _send_slack_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send Slack notification"""
        config = channel.config
        
        if not config.get("webhook_url"):
            self.logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            # Severity colors
            colors = {
                AlertSeverity.LOW: "#36a64f",      # Green
                AlertSeverity.MEDIUM: "#ff9500",   # Orange
                AlertSeverity.HIGH: "#ff0000",     # Red
                AlertSeverity.CRITICAL: "#8B0000"  # Dark Red
            }
            
            # Create Slack message
            payload = {
                "channel": config.get("channel", "#alerts"),
                "username": config.get("username", "Pipeline Alert Bot"),
                "attachments": [{
                    "color": colors.get(alert.severity, "#36a64f"),
                    "title": f"[{alert.severity.value}] {alert.name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Alert ID", "value": alert.id, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Timestamp", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ],
                    "footer": "Pipeline Monitoring System",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            # Send to Slack
            response = requests.post(
                config["webhook_url"],
                json=payload,
                timeout=30
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send webhook notification"""
        config = channel.config
        
        if not config.get("url"):
            self.logger.warning("Webhook URL not configured")
            return False
        
        try:
            # Create webhook payload
            payload = {
                "alert_id": alert.id,
                "name": alert.name,
                "severity": alert.severity.value,
                "message": alert.message,
                "source": alert.source,
                "timestamp": alert.timestamp.isoformat(),
                "status": alert.status.value,
                "tags": alert.tags,
                "metadata": alert.metadata
            }
            
            # Send webhook
            response = requests.request(
                method=config.get("method", "POST"),
                url=config["url"],
                json=payload,
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 30)
            )
            
            return response.status_code < 400
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def _send_console_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send console notification"""
        severity_icons = {
            AlertSeverity.LOW: "🟡",
            AlertSeverity.MEDIUM: "🟠",
            AlertSeverity.HIGH: "🔴",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        icon = severity_icons.get(alert.severity, "⚠️")
        
        print(f"\n{icon} ALERT [{alert.severity.value}] {alert.name}")
        print(f"   Message: {alert.message}")
        print(f"   Source: {alert.source}")
        print(f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Alert ID: {alert.id}")
        
        return True
    
    def _log_notification(self, alert_id: str, channel_name: str, channel_type: str, 
                         status: str, error_message: str = None):
        """Log notification attempt"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alert_notifications 
                    (alert_id, channel_name, channel_type, status, sent_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert_id, channel_name, channel_type, status,
                    datetime.now().isoformat(), error_message
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging notification: {e}")
    
    def _schedule_escalation(self, alert: Alert, rule: AlertRule):
        """Schedule alert escalation"""
        if alert.escalation_count >= rule.max_escalations:
            return
        
        escalation_time = datetime.now() + timedelta(minutes=rule.escalation_minutes)
        self.escalation_timers[alert.id] = escalation_time
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()
        
        # Update database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE alerts 
                    SET status = ?, acknowledged_by = ?, acknowledged_at = ?
                    WHERE id = ?
                ''', (
                    alert.status.value, alert.acknowledged_by,
                    alert.acknowledged_at.isoformat(), alert_id
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
        
        # Cancel escalation
        if alert_id in self.escalation_timers:
            del self.escalation_timers[alert_id]
        
        self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        # Update database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE alerts 
                    SET status = ?, resolved_at = ?
                    WHERE id = ?
                ''', (
                    alert.status.value, alert.resolved_at.isoformat(), alert_id
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        # Cancel escalation
        if alert_id in self.escalation_timers:
            del self.escalation_timers[alert_id]
        
        self.logger.info(f"Alert resolved: {alert_id}")
        return True
    
    def start_processing(self):
        """Start background alert processing"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(
                target=self._process_alerts,
                daemon=True
            )
            self.processing_thread.start()
            self.logger.info("Started alert processing thread")
    
    def stop_processing(self):
        """Stop background alert processing"""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        self.logger.info("Stopped alert processing thread")
    
    def _process_alerts(self):
        """Background alert processing"""
        while not self.stop_processing.wait(60):  # Check every minute
            try:
                self._process_escalations()
                self._cleanup_old_alerts()
            except Exception as e:
                self.logger.error(f"Error in alert processing: {e}")
    
    def _process_escalations(self):
        """Process alert escalations"""
        current_time = datetime.now()
        
        for alert_id, escalation_time in list(self.escalation_timers.items()):
            if current_time >= escalation_time:
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    rule = self.alert_rules.get(alert.name)
                    
                    if rule and alert.escalation_count < rule.max_escalations:
                        self._escalate_alert(alert, rule)
                
                # Remove from escalation timers
                del self.escalation_timers[alert_id]
    
    def _escalate_alert(self, alert: Alert, rule: AlertRule):
        """Escalate an alert"""
        alert.escalated = True
        alert.escalation_count += 1
        
        # Update severity for escalation
        if alert.severity == AlertSeverity.LOW:
            alert.severity = AlertSeverity.MEDIUM
        elif alert.severity == AlertSeverity.MEDIUM:
            alert.severity = AlertSeverity.HIGH
        elif alert.severity == AlertSeverity.HIGH:
            alert.severity = AlertSeverity.CRITICAL
        
        # Send escalated notifications
        self._send_notifications(alert)
        
        # Schedule next escalation if needed
        if alert.escalation_count < rule.max_escalations:
            self._schedule_escalation(alert, rule)
        
        # Update database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE alerts 
                    SET escalated = ?, escalation_count = ?, severity = ?
                    WHERE id = ?
                ''', (
                    alert.escalated, alert.escalation_count,
                    alert.severity.value, alert.id
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error updating escalated alert {alert.id}: {e}")
        
        self.logger.warning(f"Alert escalated: {alert.id} (escalation #{alert.escalation_count})")
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old resolved alerts
                cursor.execute('''
                    DELETE FROM alerts 
                    WHERE status = 'RESOLVED' AND resolved_at < ?
                ''', (cutoff_time.isoformat(),))
                
                deleted_count = cursor.rowcount
                
                # Delete associated notifications
                cursor.execute('''
                    DELETE FROM alert_notifications 
                    WHERE alert_id NOT IN (SELECT id FROM alerts)
                ''')
                
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old alerts")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old alerts: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get alert summary statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total alerts
                cursor.execute('''
                    SELECT COUNT(*) FROM alerts 
                    WHERE timestamp >= ?
                ''', (cutoff_time.isoformat(),))
                total_alerts = cursor.fetchone()[0]
                
                # Alerts by severity
                cursor.execute('''
                    SELECT severity, COUNT(*) FROM alerts 
                    WHERE timestamp >= ?
                    GROUP BY severity
                ''', (cutoff_time.isoformat(),))
                severity_counts = dict(cursor.fetchall())
                
                # Alerts by status
                cursor.execute('''
                    SELECT status, COUNT(*) FROM alerts 
                    WHERE timestamp >= ?
                    GROUP BY status
                ''', (cutoff_time.isoformat(),))
                status_counts = dict(cursor.fetchall())
                
                return {
                    'time_period_hours': hours_back,
                    'total_alerts': total_alerts,
                    'active_alerts': len(self.active_alerts),
                    'severity_breakdown': severity_counts,
                    'status_breakdown': status_counts,
                    'escalated_alerts': sum(1 for alert in self.active_alerts.values() if alert.escalated)
                }
                
        except Exception as e:
            self.logger.error(f"Error getting alert summary: {e}")
            return {'error': str(e)}


# Global alert manager instance
_global_alert_manager = None

def get_alert_manager(db_path: str = "data/alerts.db", 
                     metrics_collector: MetricsCollector = None) -> AlertManager:
    """Get or create global alert manager"""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager(db_path, metrics_collector)
    return _global_alert_manager


if __name__ == "__main__":
    # Test the alerting system
    print("Testing Alerting System...")
    
    # Test alert manager
    alert_manager = get_alert_manager("test_alerts.db")
    
    # Test normal metrics (should not trigger alerts)
    normal_metrics = {
        "pipeline_success": True,
        "execution_time": 45.2,
        "data_quality_score": 95.0,
        "throughput": 25.5,
        "error_rate": 0.01,
        "pipeline_id": "TEST-001"
    }
    
    alerts1 = alert_manager.evaluate_metrics(normal_metrics, "test_pipeline")
    print(f"Normal metrics: {len(alerts1)} alerts triggered")
    
    # Test problematic metrics (should trigger alerts)
    problem_metrics = {
        "pipeline_success": False,
        "execution_time": 350.0,
        "data_quality_score": 75.0,
        "throughput": 5.0,
        "error_rate": 0.08,
        "error_message": "Database connection timeout",
        "pipeline_id": "TEST-002"
    }
    
    alerts2 = alert_manager.evaluate_metrics(problem_metrics, "test_pipeline")
    print(f"Problem metrics: {len(alerts2)} alerts triggered")
    
    # Test alert acknowledgment
    if alerts2:
        alert_id = alerts2[0].id
        success = alert_manager.acknowledge_alert(alert_id, "test_user")
        print(f"Alert acknowledgment: {'Success' if success else 'Failed'}")
        
        # Test alert resolution
        success = alert_manager.resolve_alert(alert_id)
        print(f"Alert resolution: {'Success' if success else 'Failed'}")
    
    # Get alert summary
    summary = alert_manager.get_alert_summary()
    print(f"\nAlert Summary:")
    print(json.dumps(summary, indent=2))
    
    # Cleanup
    alert_manager.stop_processing()
    
    print("✅ Alerting system test completed!")
    
    # Clean up test database
    import os
    try:
        os.remove("test_alerts.db")
    except:
        pass
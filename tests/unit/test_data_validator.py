"""
Unit tests for data validation module
Tests data quality validation, schema validation, and business rules
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.validation.data_validator import DataValidator, ValidationResult
from src.utils.constants import DataQualityLevel, QualityScore

@pytest.mark.unit
class TestDataValidator:
    """Test cases for DataValidator class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.validator = DataValidator()
    
    def test_initialization(self):
        """Test DataValidator initialization"""
        assert self.validator is not None
        assert hasattr(self.validator, 'required_fields')
        assert hasattr(self.validator, 'optional_fields')
        assert hasattr(self.validator, 'validation_rules')
        assert len(self.validator.validation_rules) > 0
    
    def test_validate_orders_valid_data(self, sample_orders_data):
        """Test validation with completely valid data"""
        result = self.validator.validate_orders(sample_orders_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.quality_score >= 90  # Should be high quality
        assert result.total_records == len(sample_orders_data)
        assert result.valid_records == len(sample_orders_data)
        assert result.invalid_records == 0
        assert len(result.errors) == 0
        assert result.quality_level in [DataQualityLevel.EXCELLENT.value, DataQualityLevel.GOOD.value]
        assert result.validation_time > 0
    
    def test_validate_orders_invalid_data(self, invalid_orders_data):
        """Test validation with invalid data"""
        result = self.validator.validate_orders(invalid_orders_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.quality_score < 50  # Should be low quality
        assert result.total_records == len(invalid_orders_data)
        assert result.invalid_records > 0
        assert len(result.errors) > 0
        assert result.quality_level == DataQualityLevel.CRITICAL.value
    
    def test_validate_orders_empty_dataframe(self):
        """Test validation with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = self.validator.validate_orders(empty_df)
        
        assert result.is_valid is False
        assert result.quality_score == 0
        assert result.total_records == 0
        assert len(result.errors) > 0
        # Should have schema error for empty dataset
        assert any(error['type'] == 'schema_error' for error in result.errors)
    
    def test_validate_orders_missing_required_fields(self, sample_orders_data):
        """Test validation with missing required fields"""
        # Remove a required field
        incomplete_data = sample_orders_data.drop('order_id', axis=1)
        result = self.validator.validate_orders(incomplete_data)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        # Should have schema error for missing required field
        schema_errors = [error for error in result.errors if error['type'] == 'schema_error']
        assert len(schema_errors) > 0
        assert any('order_id' in error['message'] for error in schema_errors)
    
    def test_validate_order_id_valid(self):
        """Test order ID validation with valid IDs"""
        test_data = pd.DataFrame({
            'order_id': ['ORD-2024-001', 'ORD-2024-002', 'API-2024-003']
        })
        
        # Test individual validation
        errors = self.validator._validate_order_id('ORD-2024-001', test_data.iloc[0], 0)
        assert len(errors) == 0
    
    def test_validate_order_id_invalid_format(self):
        """Test order ID validation with invalid formats"""
        test_cases = [
            'INVALID-ID',      # Wrong format
            'ORD-24-001',      # Wrong year format
            'ORD-2024-1',      # Wrong number format
            'ord-2024-001',    # Lowercase
            '123-2024-001',    # Numbers instead of letters
        ]
        
        for invalid_id in test_cases:
            test_data = pd.Series({'order_id': invalid_id})
            errors = self.validator._validate_order_id(invalid_id, test_data, 0)
            assert len(errors) > 0
            assert any(error['type'] == 'format_error' for error in errors)
    
    def test_validate_order_id_null_empty(self):
        """Test order ID validation with null/empty values"""
        test_cases = [None, '', '   ', np.nan]
        
        for invalid_value in test_cases:
            test_data = pd.Series({'order_id': invalid_value})
            errors = self.validator._validate_order_id(invalid_value, test_data, 0)
            assert len(errors) > 0
            assert any(error['type'] == 'required_field_error' for error in errors)
            assert any(error['severity'] == 'critical' for error in errors)
    
    def test_validate_customer_name_valid(self):
        """Test customer name validation with valid names"""
        valid_names = [
            'John Doe',
            'Jane Smith-Johnson',
            "Mary O'Connor",
            'José García',
            'Li Wei'
        ]
        
        for name in valid_names:
            test_data = pd.Series({'customer_name': name})
            errors = self.validator._validate_customer_name(name, test_data, 0)
            # Should have no errors or only warnings
            critical_errors = [e for e in errors if e['severity'] == 'critical']
            assert len(critical_errors) == 0
    
    def test_validate_customer_name_invalid(self):
        """Test customer name validation with invalid names"""
        invalid_names = [
            '',           # Empty
            '   ',        # Whitespace only
            'A',          # Too short
            'test',       # Test data
            'dummy',      # Test data
            'X' * 101,    # Too long
        ]
        
        for name in invalid_names:
            test_data = pd.Series({'customer_name': name})
            errors = self.validator._validate_customer_name(name, test_data, 0)
            assert len(errors) > 0
    
    def test_validate_quantity_valid(self):
        """Test quantity validation with valid values"""
        valid_quantities = [1, 2, 10, 100, 999]
        
        for quantity in valid_quantities:
            test_data = pd.Series({'quantity': quantity})
            errors = self.validator._validate_quantity(quantity, test_data, 0)
            # Should have no errors or only warnings
            critical_errors = [e for e in errors if e['severity'] in ['critical', 'high']]
            assert len(critical_errors) == 0
    
    def test_validate_quantity_invalid(self):
        """Test quantity validation with invalid values"""
        invalid_quantities = [
            0,           # Zero
            -1,          # Negative
            -10,         # Negative
            1.5,         # Decimal
            10000,       # Too high
            'abc',       # Non-numeric
            None,        # Null
        ]
        
        for quantity in invalid_quantities:
            test_data = pd.Series({'quantity': quantity})
            errors = self.validator._validate_quantity(quantity, test_data, 0)
            assert len(errors) > 0
    
    def test_validate_price_valid(self):
        """Test price validation with valid values"""
        valid_prices = [0.01, 1.00, 99.99, 1999.99, 10000.00]
        
        for price in valid_prices:
            test_data = pd.Series({'price': price})
            errors = self.validator._validate_price(price, test_data, 0)
            # Should have no errors or only warnings
            critical_errors = [e for e in errors if e['severity'] in ['critical', 'high']]
            assert len(critical_errors) == 0
    
    def test_validate_price_invalid(self):
        """Test price validation with invalid values"""
        invalid_prices = [
            0,           # Zero
            -1,          # Negative
            -100.50,     # Negative
            100000,      # Too high
            'abc',       # Non-numeric
            None,        # Null
        ]
        
        for price in invalid_prices:
            test_data = pd.Series({'price': price})
            errors = self.validator._validate_price(price, test_data, 0)
            assert len(errors) > 0
    
    def test_validate_order_date_valid(self):
        """Test order date validation with valid dates"""
        valid_dates = [
            '2024-01-15',
            '2023-12-31',
            datetime.now().strftime('%Y-%m-%d'),
            (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
        ]
        
        for date_str in valid_dates:
            test_data = pd.Series({'order_date': date_str})
            errors = self.validator._validate_order_date(date_str, test_data, 0)
            # Should have no errors or only warnings
            critical_errors = [e for e in errors if e['severity'] in ['critical', 'high']]
            assert len(critical_errors) == 0
    
    def test_validate_order_date_invalid(self):
        """Test order date validation with invalid dates"""
        future_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        very_old_date = (datetime.now() - timedelta(days=6*365)).strftime('%Y-%m-%d')
        
        invalid_dates = [
            future_date,     # Future date
            very_old_date,   # Very old date
            '2024-13-01',    # Invalid month
            '2024-01-32',    # Invalid day
            'invalid-date',  # Invalid format
            None,            # Null
            '',              # Empty
        ]
        
        for date_str in invalid_dates:
            test_data = pd.Series({'order_date': date_str})
            errors = self.validator._validate_order_date(date_str, test_data, 0)
            assert len(errors) > 0
    
    def test_validate_customer_email_valid(self):
        """Test email validation with valid emails"""
        valid_emails = [
            'user@example.com',
            'test.email@domain.co.uk',
            'user+tag@example.org',
            'user123@test-domain.com',
        ]
        
        for email in valid_emails:
            test_data = pd.Series({'customer_email': email})
            errors = self.validator._validate_customer_email(email, test_data, 0)
            assert len(errors) == 0
    
    def test_validate_customer_email_invalid(self):
        """Test email validation with invalid emails"""
        invalid_emails = [
            'invalid-email',
            '@domain.com',
            'user@',
            'user@domain',
            'user.domain.com',
            'user@domain.',
        ]
        
        for email in invalid_emails:
            test_data = pd.Series({'customer_email': email})
            errors = self.validator._validate_customer_email(email, test_data, 0)
            assert len(errors) > 0
            assert any(error['type'] == 'format_error' for error in errors)
    
    def test_validate_customer_email_null_allowed(self):
        """Test that null email values are allowed"""
        test_data = pd.Series({'customer_email': None})
        errors = self.validator._validate_customer_email(None, test_data, 0)
        assert len(errors) == 0  # Null should be allowed for email
    
    def test_validate_discount_valid(self):
        """Test discount validation with valid values"""
        valid_discounts = [0, 0.0, 10.50, 100.00]
        
        for discount in valid_discounts:
            test_data = pd.Series({'discount': discount, 'price': 200.00})
            errors = self.validator._validate_discount(discount, test_data, 0)
            # Should have no errors or only warnings
            critical_errors = [e for e in errors if e['severity'] in ['critical', 'high']]
            assert len(critical_errors) == 0
    
    def test_validate_discount_invalid(self):
        """Test discount validation with invalid values"""
        # Test negative discount
        test_data = pd.Series({'discount': -10.0, 'price': 100.0})
        errors = self.validator._validate_discount(-10.0, test_data, 0)
        assert len(errors) > 0
        assert any(error['type'] == 'business_rule_error' for error in errors)
        
        # Test discount exceeding price
        test_data = pd.Series({'discount': 150.0, 'price': 100.0})
        errors = self.validator._validate_discount(150.0, test_data, 0)
        assert len(errors) > 0
        assert any(error['type'] == 'business_rule_error' for error in errors)
    
    def test_validate_total_amount_correct_calculation(self):
        """Test total amount validation with correct calculation"""
        test_data = pd.Series({
            'total_amount': 199.99,
            'price': 99.99,
            'quantity': 2,
            'discount': 0.0
        })
        
        errors = self.validator._validate_total_amount(199.98, test_data, 0)
        # Should allow small rounding differences
        assert len(errors) == 0
    
    def test_validate_total_amount_incorrect_calculation(self):
        """Test total amount validation with incorrect calculation"""
        test_data = pd.Series({
            'total_amount': 150.00,  # Incorrect
            'price': 99.99,
            'quantity': 2,
            'discount': 0.0
        })
        
        errors = self.validator._validate_total_amount(150.00, test_data, 0)
        assert len(errors) > 0
        assert any(error['type'] == 'calculation_error' for error in errors)
    
    def test_validate_cross_fields_duplicate_order_ids(self):
        """Test cross-field validation for duplicate order IDs"""
        duplicate_data = pd.DataFrame({
            'order_id': ['ORD-2024-001', 'ORD-2024-001', 'ORD-2024-002'],
            'customer_name': ['John Doe', 'Jane Smith', 'Bob Wilson'],
            'product': ['Product A', 'Product B', 'Product C'],
            'quantity': [1, 1, 1],
            'price': [100.0, 200.0, 300.0],
            'order_date': ['2024-01-15', '2024-01-16', '2024-01-17']
        })
        
        errors = self.validator._validate_cross_fields(duplicate_data)
        assert len(errors) > 0
        assert any(error['type'] == 'duplicate_error' for error in errors)
        assert any('order_id' in error['field'] for error in errors)
    
    def test_check_data_quality_completeness(self):
        """Test data quality checks for completeness"""
        # Create data with missing values
        incomplete_data = pd.DataFrame({
            'order_id': ['ORD-2024-001', 'ORD-2024-002', 'ORD-2024-003'],
            'customer_name': ['John Doe', None, 'Bob Wilson'],  # Missing value
            'product': ['Product A', 'Product B', 'Product C'],
            'quantity': [1, 1, 1],
            'price': [100.0, 200.0, 300.0],
            'order_date': ['2024-01-15', '2024-01-16', '2024-01-17']
        })
        
        warnings = self.validator._check_data_quality(incomplete_data)
        assert len(warnings) > 0
        completeness_warnings = [w for w in warnings if w['type'] == 'completeness_warning']
        assert len(completeness_warnings) > 0
    
    def test_check_data_quality_suspicious_patterns(self):
        """Test data quality checks for suspicious patterns"""
        # Create data with suspicious patterns
        suspicious_data = pd.DataFrame({
            'order_id': ['ORD-2024-001', 'ORD-2024-002', 'ORD-2024-003'],
            'customer_name': ['Test User', 'Test User', 'Test User'],  # Same name repeated
            'product': ['Product A', 'Product B', 'Product C'],
            'quantity': [1, 1, 1],
            'price': [100.0, 200.0, 300.0],
            'order_date': ['2024-01-15', '2024-01-16', '2024-01-17']
        })
        
        warnings = self.validator._check_data_quality(suspicious_data)
        pattern_warnings = [w for w in warnings if w['type'] == 'pattern_warning']
        # Should detect suspicious customer name patterns
        assert len(pattern_warnings) > 0
    
    def test_determine_quality_level(self):
        """Test quality level determination based on score"""
        test_cases = [
            (98.0, DataQualityLevel.EXCELLENT.value),
            (90.0, DataQualityLevel.GOOD.value),
            (75.0, DataQualityLevel.FAIR.value),
            (55.0, DataQualityLevel.POOR.value),
            (30.0, DataQualityLevel.CRITICAL.value),
        ]
        
        for score, expected_level in test_cases:
            level = self.validator._determine_quality_level(score)
            assert level == expected_level
    
    def test_generate_validation_report(self, sample_orders_data):
        """Test validation report generation"""
        result = self.validator.validate_orders(sample_orders_data)
        report = self.validator.generate_validation_report(result)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "# Data Validation Report" in report
        assert "## Summary" in report
        assert f"Total Records**: {result.total_records:,}" in report
        assert f"Quality Score**: {result.quality_score:.1f}%" in report
    
    def test_generate_validation_report_with_errors(self, invalid_orders_data):
        """Test validation report generation with errors"""
        result = self.validator.validate_orders(invalid_orders_data)
        report = self.validator.generate_validation_report(result)
        
        assert "## Errors" in report
        assert "### Top Errors" in report
        # Should include error details
        assert len([line for line in report.split('\n') if line.strip().startswith('1.')]) > 0
    
    def test_generate_validation_report_with_warnings(self):
        """Test validation report generation with warnings"""
        # Create data that will generate warnings
        warning_data = pd.DataFrame({
            'order_id': ['ORD-2024-001'],
            'customer_name': ['John Doe'],
            'product': ['iPhone 15'],
            'quantity': [1],
            'price': [999.99],
            'order_date': ['2024-01-15'],
            'customer_email': [None]  # Missing optional field
        })
        
        result = self.validator.validate_orders(warning_data)
        report = self.validator.generate_validation_report(result)
        
        if result.warnings:
            assert "## Warnings" in report
    
    @patch('src.validation.data_validator.logger')
    def test_logging_calls(self, mock_logger, sample_orders_data):
        """Test that appropriate logging calls are made"""
        result = self.validator.validate_orders(sample_orders_data)
        
        # Check that info logging was called
        mock_logger.info.assert_called()
        
        # Check for specific log messages
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Starting validation" in msg for msg in log_calls)
        assert any("Validation completed" in msg for msg in log_calls)
    
    def test_validation_with_mixed_data_types(self):
        """Test validation with mixed data types"""
        mixed_data = pd.DataFrame({
            'order_id': ['ORD-2024-001', 'ORD-2024-002'],
            'customer_name': ['John Doe', 'Jane Smith'],
            'product': ['iPhone 15', 'MacBook Pro'],
            'quantity': ['1', 2],  # Mixed string and int
            'price': [999.99, '1999.99'],  # Mixed float and string
            'order_date': ['2024-01-15', '2024-01-16']
        })
        
        result = self.validator.validate_orders(mixed_data)
        
        # Should handle mixed types gracefully
        assert isinstance(result, ValidationResult)
        assert result.total_records == 2
    
    def test_validation_performance_large_dataset(self):
        """Test validation performance with larger dataset"""
        # Create a larger dataset
        large_data = pd.DataFrame({
            'order_id': [f'ORD-2024-{i:03d}' for i in range(1, 1001)],
            'customer_name': [f'Customer {i}' for i in range(1, 1001)],
            'product': ['iPhone 15'] * 1000,
            'quantity': [1] * 1000,
            'price': [999.99] * 1000,
            'order_date': ['2024-01-15'] * 1000
        })
        
        result = self.validator.validate_orders(large_data)
        
        assert result.total_records == 1000
        assert result.validation_time > 0
        # Should complete in reasonable time (less than 10 seconds)
        assert result.validation_time < 10.0
    
    def test_validation_error_categorization(self, invalid_orders_data):
        """Test that validation errors are properly categorized by severity"""
        result = self.validator.validate_orders(invalid_orders_data)
        
        # Check that errors have severity levels
        for error in result.errors:
            assert 'severity' in error
            assert error['severity'] in ['critical', 'high', 'medium', 'low']
        
        # Should have critical errors for required field violations
        critical_errors = [e for e in result.errors if e['severity'] == 'critical']
        assert len(critical_errors) > 0
    
    def test_validation_with_custom_thresholds(self):
        """Test validation with custom quality thresholds"""
        # This would require modifying the validator to accept custom thresholds
        # For now, test that the current thresholds work as expected
        
        test_cases = [
            (100.0, DataQualityLevel.EXCELLENT.value),
            (95.0, DataQualityLevel.EXCELLENT.value),
            (94.9, DataQualityLevel.GOOD.value),
            (85.0, DataQualityLevel.GOOD.value),
            (84.9, DataQualityLevel.FAIR.value),
        ]
        
        for score, expected_level in test_cases:
            level = self.validator._determine_quality_level(score)
            assert level == expected_level
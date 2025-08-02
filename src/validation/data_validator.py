"""
Data validation module for ensuring data quality and integrity
Validates business rules, data types, and data completeness
"""

import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils.constants import ValidationRule, DataQualityLevel, QualityScore
from ..utils.helpers import validate_email, safe_divide

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Data validation result container"""
    is_valid: bool
    quality_score: float
    total_records: int
    valid_records: int
    invalid_records: int
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    quality_level: str
    validation_time: float

class DataValidator:
    """Comprehensive data validator for order data"""
    
    def __init__(self):
        """Initialize data validator"""
        self.required_fields = ValidationRule.REQUIRED_FIELDS
        self.optional_fields = ValidationRule.OPTIONAL_FIELDS
        self.numeric_fields = ValidationRule.NUMERIC_FIELDS
        self.date_fields = ValidationRule.DATE_FIELDS
        self.email_fields = ValidationRule.EMAIL_FIELDS
        
        # Validation rules
        self.validation_rules = {
            'order_id': self._validate_order_id,
            'customer_name': self._validate_customer_name,
            'product': self._validate_product,
            'quantity': self._validate_quantity,
            'price': self._validate_price,
            'order_date': self._validate_order_date,
            'customer_email': self._validate_customer_email,
            'discount': self._validate_discount,
            'total_amount': self._validate_total_amount
        }
        
        logger.info("Data validator initialized")
    
    def validate_orders(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate order data comprehensively
        
        Args:
            data (pd.DataFrame): Order data to validate
            
        Returns:
            ValidationResult: Comprehensive validation results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting validation of {len(data)} records")
            
            errors = []
            warnings = []
            valid_record_count = 0
            
            # 1. Schema validation
            schema_errors = self._validate_schema(data)
            errors.extend(schema_errors)
            
            # 2. Data type validation
            type_errors = self._validate_data_types(data)
            errors.extend(type_errors)
            
            # 3. Business rule validation
            for index, row in data.iterrows():
                record_errors = self._validate_record(row, index)
                if not record_errors:
                    valid_record_count += 1
                else:
                    errors.extend(record_errors)
            
            # 4. Cross-field validation
            cross_field_errors = self._validate_cross_fields(data)
            errors.extend(cross_field_errors)
            
            # 5. Data quality checks
            quality_warnings = self._check_data_quality(data)
            warnings.extend(quality_warnings)
            
            # Calculate metrics
            total_records = len(data)
            invalid_records = total_records - valid_record_count
            quality_score = safe_divide(valid_record_count, total_records, 0) * 100
            quality_level = self._determine_quality_level(quality_score)
            
            validation_time = (datetime.now() - start_time).total_seconds()
            
            result = ValidationResult(
                is_valid=len(errors) == 0,
                quality_score=quality_score,
                total_records=total_records,
                valid_records=valid_record_count,
                invalid_records=invalid_records,
                errors=errors,
                warnings=warnings,
                quality_level=quality_level,
                validation_time=validation_time
            )
            
            logger.info(f"Validation completed: {quality_score:.1f}% quality score, {len(errors)} errors, {len(warnings)} warnings")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                total_records=len(data) if data is not None else 0,
                valid_records=0,
                invalid_records=len(data) if data is not None else 0,
                errors=[{'type': 'validation_error', 'message': str(e)}],
                warnings=[],
                quality_level=DataQualityLevel.CRITICAL.value,
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_schema(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate data schema"""
        errors = []
        
        # Check required fields
        missing_fields = [field for field in self.required_fields if field not in data.columns]
        if missing_fields:
            errors.append({
                'type': 'schema_error',
                'field': 'schema',
                'message': f"Missing required fields: {', '.join(missing_fields)}",
                'severity': 'critical'
            })
        
        # Check for empty DataFrame
        if data.empty:
            errors.append({
                'type': 'schema_error',
                'field': 'data',
                'message': "Dataset is empty",
                'severity': 'critical'
            })
        
        return errors
    
    def _validate_data_types(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate data types"""
        errors = []
        
        for field in self.numeric_fields:
            if field in data.columns:
                # Check for non-numeric values
                non_numeric = data[~pd.to_numeric(data[field], errors='coerce').notna()]
                if not non_numeric.empty:
                    errors.append({
                        'type': 'data_type_error',
                        'field': field,
                        'message': f"Non-numeric values found in {field}: {len(non_numeric)} records",
                        'severity': 'high',
                        'affected_records': len(non_numeric)
                    })
        
        for field in self.date_fields:
            if field in data.columns:
                # Check for invalid dates
                try:
                    pd.to_datetime(data[field], errors='coerce')
                except Exception as e:
                    errors.append({
                        'type': 'data_type_error',
                        'field': field,
                        'message': f"Invalid date format in {field}: {str(e)}",
                        'severity': 'high'
                    })
        
        return errors
    
    def _validate_record(self, record: pd.Series, index: int) -> List[Dict[str, Any]]:
        """Validate individual record"""
        errors = []
        
        for field, validator_func in self.validation_rules.items():
            if field in record.index:
                try:
                    field_errors = validator_func(record[field], record, index)
                    errors.extend(field_errors)
                except Exception as e:
                    errors.append({
                        'type': 'validation_error',
                        'field': field,
                        'record_index': index,
                        'message': f"Validation error for {field}: {str(e)}",
                        'severity': 'medium'
                    })
        
        return errors
    
    def _validate_order_id(self, value: Any, record: pd.Series, index: int) -> List[Dict[str, Any]]:
        """Validate order ID"""
        errors = []
        
        if pd.isna(value) or str(value).strip() == '':
            errors.append({
                'type': 'required_field_error',
                'field': 'order_id',
                'record_index': index,
                'message': "Order ID is required",
                'severity': 'critical'
            })
        else:
            order_id = str(value).strip()
            # Check format (should be like ORD-2024-001)
            if not re.match(r'^[A-Z]{3}-\d{4}-\d{3}$', order_id):
                errors.append({
                    'type': 'format_error',
                    'field': 'order_id',
                    'record_index': index,
                    'value': order_id,
                    'message': f"Invalid order ID format: {order_id}. Expected format: XXX-YYYY-NNN",
                    'severity': 'medium'
                })
        
        return errors
    
    def _validate_customer_name(self, value: Any, record: pd.Series, index: int) -> List[Dict[str, Any]]:
        """Validate customer name"""
        errors = []
        
        if pd.isna(value) or str(value).strip() == '':
            errors.append({
                'type': 'required_field_error',
                'field': 'customer_name',
                'record_index': index,
                'message': "Customer name is required",
                'severity': 'critical'
            })
        else:
            name = str(value).strip()
            if len(name) < 2:
                errors.append({
                    'type': 'validation_error',
                    'field': 'customer_name',
                    'record_index': index,
                    'value': name,
                    'message': f"Customer name too short: {name}",
                    'severity': 'medium'
                })
            elif len(name) > 100:
                errors.append({
                    'type': 'validation_error',
                    'field': 'customer_name',
                    'record_index': index,
                    'value': name[:50] + "...",
                    'message': f"Customer name too long: {len(name)} characters",
                    'severity': 'low'
                })
        
        return errors
    
    def _validate_product(self, value: Any, record: pd.Series, index: int) -> List[Dict[str, Any]]:
        """Validate product name"""
        errors = []
        
        if pd.isna(value) or str(value).strip() == '':
            errors.append({
                'type': 'required_field_error',
                'field': 'product',
                'record_index': index,
                'message': "Product name is required",
                'severity': 'critical'
            })
        else:
            product = str(value).strip()
            if len(product) < 2:
                errors.append({
                    'type': 'validation_error',
                    'field': 'product',
                    'record_index': index,
                    'value': product,
                    'message': f"Product name too short: {product}",
                    'severity': 'medium'
                })
        
        return errors
    
    def _validate_quantity(self, value: Any, record: pd.Series, index: int) -> List[Dict[str, Any]]:
        """Validate quantity"""
        errors = []
        
        if pd.isna(value):
            errors.append({
                'type': 'required_field_error',
                'field': 'quantity',
                'record_index': index,
                'message': "Quantity is required",
                'severity': 'critical'
            })
        else:
            try:
                quantity = float(value)
                if quantity <= 0:
                    errors.append({
                        'type': 'business_rule_error',
                        'field': 'quantity',
                        'record_index': index,
                        'value': quantity,
                        'message': f"Quantity must be positive: {quantity}",
                        'severity': 'high'
                    })
                elif quantity > 1000:
                    errors.append({
                        'type': 'business_rule_error',
                        'field': 'quantity',
                        'record_index': index,
                        'value': quantity,
                        'message': f"Unusually high quantity: {quantity}",
                        'severity': 'low'
                    })
                elif quantity != int(quantity):
                    errors.append({
                        'type': 'validation_error',
                        'field': 'quantity',
                        'record_index': index,
                        'value': quantity,
                        'message': f"Quantity should be a whole number: {quantity}",
                        'severity': 'medium'
                    })
            except (ValueError, TypeError):
                errors.append({
                    'type': 'data_type_error',
                    'field': 'quantity',
                    'record_index': index,
                    'value': str(value),
                    'message': f"Invalid quantity format: {value}",
                    'severity': 'high'
                })
        
        return errors
    
    def _validate_price(self, value: Any, record: pd.Series, index: int) -> List[Dict[str, Any]]:
        """Validate price"""
        errors = []
        
        if pd.isna(value):
            errors.append({
                'type': 'required_field_error',
                'field': 'price',
                'record_index': index,
                'message': "Price is required",
                'severity': 'critical'
            })
        else:
            try:
                price = float(value)
                if price <= 0:
                    errors.append({
                        'type': 'business_rule_error',
                        'field': 'price',
                        'record_index': index,
                        'value': price,
                        'message': f"Price must be positive: {price}",
                        'severity': 'high'
                    })
                elif price > 50000:
                    errors.append({
                        'type': 'business_rule_error',
                        'field': 'price',
                        'record_index': index,
                        'value': price,
                        'message': f"Unusually high price: ${price:,.2f}",
                        'severity': 'low'
                    })
                elif price < 0.01:
                    errors.append({
                        'type': 'business_rule_error',
                        'field': 'price',
                        'record_index': index,
                        'value': price,
                        'message': f"Price too low: ${price:.4f}",
                        'severity': 'medium'
                    })
            except (ValueError, TypeError):
                errors.append({
                    'type': 'data_type_error',
                    'field': 'price',
                    'record_index': index,
                    'value': str(value),
                    'message': f"Invalid price format: {value}",
                    'severity': 'high'
                })
        
        return errors
    
    def _validate_order_date(self, value: Any, record: pd.Series, index: int) -> List[Dict[str, Any]]:
        """Validate order date"""
        errors = []
        
        if pd.isna(value) or str(value).strip() == '':
            errors.append({
                'type': 'required_field_error',
                'field': 'order_date',
                'record_index': index,
                'message': "Order date is required",
                'severity': 'critical'
            })
        else:
            try:
                order_date = pd.to_datetime(value)
                current_date = datetime.now()
                
                # Check if date is in the future
                if order_date > current_date:
                    errors.append({
                        'type': 'business_rule_error',
                        'field': 'order_date',
                        'record_index': index,
                        'value': str(value),
                        'message': f"Order date cannot be in the future: {order_date.strftime('%Y-%m-%d')}",
                        'severity': 'high'
                    })
                
                # Check if date is too old (more than 5 years)
                five_years_ago = current_date - timedelta(days=5*365)
                if order_date < five_years_ago:
                    errors.append({
                        'type': 'business_rule_error',
                        'field': 'order_date',
                        'record_index': index,
                        'value': str(value),
                        'message': f"Order date is very old: {order_date.strftime('%Y-%m-%d')}",
                        'severity': 'low'
                    })
                    
            except (ValueError, TypeError):
                errors.append({
                    'type': 'data_type_error',
                    'field': 'order_date',
                    'record_index': index,
                    'value': str(value),
                    'message': f"Invalid date format: {value}",
                    'severity': 'high'
                })
        
        return errors
    
    def _validate_customer_email(self, value: Any, record: pd.Series, index: int) -> List[Dict[str, Any]]:
        """Validate customer email"""
        errors = []
        
        if pd.notna(value) and str(value).strip() != '':
            email = str(value).strip()
            if not validate_email(email):
                errors.append({
                    'type': 'format_error',
                    'field': 'customer_email',
                    'record_index': index,
                    'value': email,
                    'message': f"Invalid email format: {email}",
                    'severity': 'medium'
                })
        
        return errors
    
    def _validate_discount(self, value: Any, record: pd.Series, index: int) -> List[Dict[str, Any]]:
        """Validate discount"""
        errors = []
        
        if pd.notna(value):
            try:
                discount = float(value)
                if discount < 0:
                    errors.append({
                        'type': 'business_rule_error',
                        'field': 'discount',
                        'record_index': index,
                        'value': discount,
                        'message': f"Discount cannot be negative: {discount}",
                        'severity': 'high'
                    })
                elif 'price' in record.index and pd.notna(record['price']):
                    price = float(record['price'])
                    if discount > price:
                        errors.append({
                            'type': 'business_rule_error',
                            'field': 'discount',
                            'record_index': index,
                            'value': discount,
                            'message': f"Discount (${discount:.2f}) cannot exceed price (${price:.2f})",
                            'severity': 'high'
                        })
            except (ValueError, TypeError):
                errors.append({
                    'type': 'data_type_error',
                    'field': 'discount',
                    'record_index': index,
                    'value': str(value),
                    'message': f"Invalid discount format: {value}",
                    'severity': 'medium'
                })
        
        return errors
    
    def _validate_total_amount(self, value: Any, record: pd.Series, index: int) -> List[Dict[str, Any]]:
        """Validate total amount"""
        errors = []
        
        if pd.notna(value):
            try:
                total_amount = float(value)
                
                # Calculate expected total if we have price and quantity
                if all(field in record.index and pd.notna(record[field]) for field in ['price', 'quantity']):
                    price = float(record['price'])
                    quantity = float(record['quantity'])
                    discount = float(record.get('discount', 0))
                    
                    expected_total = (price * quantity) - discount
                    
                    # Allow small rounding differences
                    if abs(total_amount - expected_total) > 0.01:
                        errors.append({
                            'type': 'calculation_error',
                            'field': 'total_amount',
                            'record_index': index,
                            'value': total_amount,
                            'expected': expected_total,
                            'message': f"Total amount mismatch: got ${total_amount:.2f}, expected ${expected_total:.2f}",
                            'severity': 'medium'
                        })
                        
            except (ValueError, TypeError):
                errors.append({
                    'type': 'data_type_error',
                    'field': 'total_amount',
                    'record_index': index,
                    'value': str(value),
                    'message': f"Invalid total amount format: {value}",
                    'severity': 'medium'
                })
        
        return errors
    
    def _validate_cross_fields(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate relationships between fields"""
        errors = []
        
        # Check for duplicate order IDs
        if 'order_id' in data.columns:
            duplicates = data[data.duplicated(subset=['order_id'], keep=False)]
            if not duplicates.empty:
                duplicate_ids = duplicates['order_id'].unique()
                errors.append({
                    'type': 'duplicate_error',
                    'field': 'order_id',
                    'message': f"Duplicate order IDs found: {', '.join(duplicate_ids[:5])}{'...' if len(duplicate_ids) > 5 else ''}",
                    'severity': 'high',
                    'affected_records': len(duplicates)
                })
        
        return errors
    
    def _check_data_quality(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check overall data quality"""
        warnings = []
        
        # Check completeness
        for field in self.required_fields:
            if field in data.columns:
                missing_count = data[field].isna().sum()
                if missing_count > 0:
                    missing_percentage = (missing_count / len(data)) * 100
                    warnings.append({
                        'type': 'completeness_warning',
                        'field': field,
                        'message': f"{field} has {missing_count} missing values ({missing_percentage:.1f}%)",
                        'severity': 'medium' if missing_percentage > 10 else 'low',
                        'missing_count': missing_count,
                        'missing_percentage': missing_percentage
                    })
        
        # Check for unusual patterns
        if 'customer_name' in data.columns:
            # Check for too many identical customer names
            name_counts = data['customer_name'].value_counts()
            suspicious_names = name_counts[name_counts > len(data) * 0.1]  # More than 10% of records
            if not suspicious_names.empty:
                warnings.append({
                    'type': 'pattern_warning',
                    'field': 'customer_name',
                    'message': f"Suspicious customer name patterns: {list(suspicious_names.index[:3])}",
                    'severity': 'low'
                })
        
        return warnings
    
    def _determine_quality_level(self, quality_score: float) -> str:
        """Determine data quality level based on score"""
        if quality_score >= QualityScore.EXCELLENT_MIN:
            return DataQualityLevel.EXCELLENT.value
        elif quality_score >= QualityScore.GOOD_MIN:
            return DataQualityLevel.GOOD.value
        elif quality_score >= QualityScore.FAIR_MIN:
            return DataQualityLevel.FAIR.value
        elif quality_score >= QualityScore.POOR_MIN:
            return DataQualityLevel.POOR.value
        else:
            return DataQualityLevel.CRITICAL.value
    
    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate detailed validation report"""
        report = []
        report.append("# Data Validation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Total Records**: {result.total_records:,}")
        report.append(f"- **Valid Records**: {result.valid_records:,}")
        report.append(f"- **Invalid Records**: {result.invalid_records:,}")
        report.append(f"- **Quality Score**: {result.quality_score:.1f}%")
        report.append(f"- **Quality Level**: {result.quality_level.upper()}")
        report.append(f"- **Validation Time**: {result.validation_time:.2f} seconds")
        report.append("")
        
        # Errors
        if result.errors:
            report.append("## Errors")
            error_counts = {}
            for error in result.errors:
                error_type = error.get('type', 'unknown')
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in error_counts.items():
                report.append(f"- **{error_type.replace('_', ' ').title()}**: {count}")
            report.append("")
            
            # Top errors
            report.append("### Top Errors")
            for i, error in enumerate(result.errors[:10]):
                report.append(f"{i+1}. **{error.get('field', 'N/A')}**: {error.get('message', 'N/A')}")
            report.append("")
        
        # Warnings
        if result.warnings:
            report.append("## Warnings")
            for i, warning in enumerate(result.warnings[:10]):
                report.append(f"{i+1}. **{warning.get('field', 'N/A')}**: {warning.get('message', 'N/A')}")
            report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test data validator
    import pandas as pd
    
    # Create test data
    test_data = pd.DataFrame([
        {
            'order_id': 'ORD-2024-001',
            'customer_name': 'John Doe',
            'product': 'iPhone 15',
            'quantity': 1,
            'price': 999.99,
            'order_date': '2024-01-15',
            'customer_email': 'john@example.com',
            'discount': 0.0,
            'total_amount': 999.99
        },
        {
            'order_id': 'INVALID-ID',
            'customer_name': '',
            'product': 'MacBook Pro',
            'quantity': -1,
            'price': -100,
            'order_date': '2025-12-31',
            'customer_email': 'invalid-email',
            'discount': 50.0,
            'total_amount': 1000.0
        }
    ])
    
    # Test validator
    validator = DataValidator()
    result = validator.validate_orders(test_data)
    
    print("Data Validation Test Results:")
    print(f"Quality Score: {result.quality_score:.1f}%")
    print(f"Valid Records: {result.valid_records}/{result.total_records}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    if result.errors:
        print("\nSample Errors:")
        for error in result.errors[:3]:
            print(f"  - {error['field']}: {error['message']}")
    
    # Generate report
    report = validator.generate_validation_report(result)
    print(f"\nValidation report generated ({len(report)} characters)")
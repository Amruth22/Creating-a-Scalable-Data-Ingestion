"""
Schema validation module for data structure validation
Validates data schemas, field types, and data structure integrity
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json

from ..utils.constants import ValidationRule, DataType
from ..utils.helpers import validate_email

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FieldSchema:
    """Schema definition for a field"""
    name: str
    data_type: str
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    description: Optional[str] = None

@dataclass
class SchemaValidationResult:
    """Schema validation result"""
    is_valid: bool
    schema_name: str
    total_fields: int
    valid_fields: int
    invalid_fields: int
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    validation_time: float

class SchemaValidator:
    """Schema validator for data structure validation"""
    
    def __init__(self):
        """Initialize schema validator"""
        self.schemas = self._load_default_schemas()
        logger.info("Schema validator initialized")
    
    def _load_default_schemas(self) -> Dict[str, List[FieldSchema]]:
        """Load default schemas for common data types"""
        schemas = {}
        
        # Order schema
        schemas['orders'] = [
            FieldSchema('order_id', DataType.STRING, required=True, min_length=8, max_length=20, 
                       pattern=r'^[A-Z]{3}-\d{4}-\d{3}$', description='Order identifier'),
            FieldSchema('customer_name', DataType.STRING, required=True, min_length=2, max_length=100,
                       description='Customer full name'),
            FieldSchema('product', DataType.STRING, required=True, min_length=2, max_length=200,
                       description='Product name'),
            FieldSchema('quantity', DataType.INTEGER, required=True, min_value=1, max_value=1000,
                       description='Order quantity'),
            FieldSchema('price', DataType.FLOAT, required=True, min_value=0.01, max_value=50000,
                       description='Unit price'),
            FieldSchema('order_date', DataType.DATE, required=True,
                       description='Order date'),
            FieldSchema('source', DataType.STRING, required=False, max_length=50,
                       allowed_values=['website', 'mobile_app', 'store', 'phone', 'partner'],
                       description='Order source'),
            FieldSchema('customer_email', DataType.EMAIL, required=False, max_length=255,
                       description='Customer email address'),
            FieldSchema('discount', DataType.FLOAT, required=False, min_value=0, max_value=10000,
                       description='Discount amount'),
            FieldSchema('total_amount', DataType.FLOAT, required=False, min_value=0,
                       description='Total order amount'),
            FieldSchema('store_location', DataType.STRING, required=False, max_length=100,
                       description='Store location for store orders'),
            FieldSchema('product_category', DataType.STRING, required=False, max_length=50,
                       description='Product category'),
            FieldSchema('notes', DataType.STRING, required=False, max_length=500,
                       description='Order notes')
        ]
        
        # Customer schema
        schemas['customers'] = [
            FieldSchema('customer_id', DataType.STRING, required=True, min_length=5, max_length=20,
                       pattern=r'^CUST-\d{3}$', description='Customer identifier'),
            FieldSchema('name', DataType.STRING, required=True, min_length=2, max_length=100,
                       description='Customer name'),
            FieldSchema('email', DataType.EMAIL, required=False, max_length=255,
                       description='Customer email'),
            FieldSchema('phone', DataType.STRING, required=False, min_length=10, max_length=20,
                       description='Customer phone number'),
            FieldSchema('address', DataType.STRING, required=False, max_length=500,
                       description='Customer address')
        ]
        
        # Product schema
        schemas['products'] = [
            FieldSchema('product_id', DataType.STRING, required=True, min_length=5, max_length=20,
                       pattern=r'^PROD-\d{3}$', description='Product identifier'),
            FieldSchema('name', DataType.STRING, required=True, min_length=2, max_length=200,
                       description='Product name'),
            FieldSchema('category', DataType.STRING, required=False, max_length=50,
                       description='Product category'),
            FieldSchema('price', DataType.FLOAT, required=True, min_value=0.01, max_value=50000,
                       description='Product price'),
            FieldSchema('in_stock', DataType.BOOLEAN, required=False,
                       description='Product availability')
        ]
        
        return schemas
    
    def validate_schema(self, data: pd.DataFrame, schema_name: str) -> SchemaValidationResult:
        """
        Validate data against a schema
        
        Args:
            data (pd.DataFrame): Data to validate
            schema_name (str): Name of schema to use
            
        Returns:
            SchemaValidationResult: Validation results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Validating schema '{schema_name}' for {len(data)} records")
            
            if schema_name not in self.schemas:
                return SchemaValidationResult(
                    is_valid=False,
                    schema_name=schema_name,
                    total_fields=0,
                    valid_fields=0,
                    invalid_fields=0,
                    errors=[{'type': 'schema_error', 'message': f"Schema '{schema_name}' not found"}],
                    warnings=[],
                    validation_time=(datetime.now() - start_time).total_seconds()
                )
            
            schema = self.schemas[schema_name]
            errors = []
            warnings = []
            valid_fields = 0
            
            # Validate each field in schema
            for field_schema in schema:
                field_errors, field_warnings = self._validate_field_schema(data, field_schema)
                
                if not field_errors:
                    valid_fields += 1
                
                errors.extend(field_errors)
                warnings.extend(field_warnings)
            
            # Check for unexpected fields
            schema_fields = {fs.name for fs in schema}
            data_fields = set(data.columns)
            unexpected_fields = data_fields - schema_fields
            
            if unexpected_fields:
                warnings.append({
                    'type': 'unexpected_fields',
                    'message': f"Unexpected fields found: {', '.join(unexpected_fields)}",
                    'fields': list(unexpected_fields)
                })
            
            validation_time = (datetime.now() - start_time).total_seconds()
            
            result = SchemaValidationResult(
                is_valid=len(errors) == 0,
                schema_name=schema_name,
                total_fields=len(schema),
                valid_fields=valid_fields,
                invalid_fields=len(schema) - valid_fields,
                errors=errors,
                warnings=warnings,
                validation_time=validation_time
            )
            
            logger.info(f"Schema validation completed: {valid_fields}/{len(schema)} fields valid")
            return result
            
        except Exception as e:
            logger.error(f"Error during schema validation: {e}")
            return SchemaValidationResult(
                is_valid=False,
                schema_name=schema_name,
                total_fields=0,
                valid_fields=0,
                invalid_fields=0,
                errors=[{'type': 'validation_error', 'message': str(e)}],
                warnings=[],
                validation_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _validate_field_schema(self, data: pd.DataFrame, field_schema: FieldSchema) -> tuple:
        """Validate a single field against its schema"""
        errors = []
        warnings = []
        field_name = field_schema.name
        
        # Check if required field exists
        if field_schema.required and field_name not in data.columns:
            errors.append({
                'type': 'missing_required_field',
                'field': field_name,
                'message': f"Required field '{field_name}' is missing",
                'severity': 'critical'
            })
            return errors, warnings
        
        # Skip validation if field doesn't exist and is not required
        if field_name not in data.columns:
            return errors, warnings
        
        field_data = data[field_name]
        
        # Check data type
        type_errors = self._validate_field_type(field_data, field_schema)
        errors.extend(type_errors)
        
        # Check constraints
        constraint_errors, constraint_warnings = self._validate_field_constraints(field_data, field_schema)
        errors.extend(constraint_errors)
        warnings.extend(constraint_warnings)
        
        return errors, warnings
    
    def _validate_field_type(self, field_data: pd.Series, field_schema: FieldSchema) -> List[Dict[str, Any]]:
        """Validate field data type"""
        errors = []
        field_name = field_schema.name
        expected_type = field_schema.data_type
        
        # Skip null values for type checking
        non_null_data = field_data.dropna()
        if non_null_data.empty:
            return errors
        
        try:
            if expected_type == DataType.STRING:
                # Check if values can be converted to string
                non_string_count = 0
                for value in non_null_data:
                    if not isinstance(value, (str, int, float)):
                        non_string_count += 1
                
                if non_string_count > 0:
                    errors.append({
                        'type': 'data_type_error',
                        'field': field_name,
                        'expected_type': expected_type,
                        'message': f"{non_string_count} values cannot be converted to string",
                        'severity': 'medium'
                    })
            
            elif expected_type == DataType.INTEGER:
                # Check if values are integers or can be converted
                try:
                    pd.to_numeric(non_null_data, errors='raise', downcast='integer')
                except (ValueError, TypeError):
                    errors.append({
                        'type': 'data_type_error',
                        'field': field_name,
                        'expected_type': expected_type,
                        'message': f"Field contains non-integer values",
                        'severity': 'high'
                    })
            
            elif expected_type == DataType.FLOAT:
                # Check if values are numeric
                try:
                    pd.to_numeric(non_null_data, errors='raise')
                except (ValueError, TypeError):
                    errors.append({
                        'type': 'data_type_error',
                        'field': field_name,
                        'expected_type': expected_type,
                        'message': f"Field contains non-numeric values",
                        'severity': 'high'
                    })
            
            elif expected_type == DataType.BOOLEAN:
                # Check if values are boolean or can be converted
                boolean_values = {True, False, 'true', 'false', 'True', 'False', 1, 0, '1', '0'}
                invalid_count = 0
                for value in non_null_data:
                    if value not in boolean_values:
                        invalid_count += 1
                
                if invalid_count > 0:
                    errors.append({
                        'type': 'data_type_error',
                        'field': field_name,
                        'expected_type': expected_type,
                        'message': f"{invalid_count} values are not valid boolean values",
                        'severity': 'medium'
                    })
            
            elif expected_type == DataType.DATE:
                # Check if values are valid dates
                try:
                    pd.to_datetime(non_null_data, errors='raise')
                except (ValueError, TypeError):
                    errors.append({
                        'type': 'data_type_error',
                        'field': field_name,
                        'expected_type': expected_type,
                        'message': f"Field contains invalid date values",
                        'severity': 'high'
                    })
            
            elif expected_type == DataType.EMAIL:
                # Check if values are valid email addresses
                invalid_emails = 0
                for value in non_null_data:
                    if not validate_email(str(value)):
                        invalid_emails += 1
                
                if invalid_emails > 0:
                    errors.append({
                        'type': 'data_type_error',
                        'field': field_name,
                        'expected_type': expected_type,
                        'message': f"{invalid_emails} invalid email addresses found",
                        'severity': 'medium'
                    })
        
        except Exception as e:
            errors.append({
                'type': 'validation_error',
                'field': field_name,
                'message': f"Error validating field type: {str(e)}",
                'severity': 'medium'
            })
        
        return errors
    
    def _validate_field_constraints(self, field_data: pd.Series, field_schema: FieldSchema) -> tuple:
        """Validate field constraints"""
        errors = []
        warnings = []
        field_name = field_schema.name
        
        # Skip null values for constraint checking
        non_null_data = field_data.dropna()
        if non_null_data.empty:
            return errors, warnings
        
        try:
            # Length constraints for string fields
            if field_schema.data_type == DataType.STRING:
                if field_schema.min_length is not None:
                    short_values = non_null_data[non_null_data.astype(str).str.len() < field_schema.min_length]
                    if not short_values.empty:
                        errors.append({
                            'type': 'constraint_error',
                            'field': field_name,
                            'constraint': 'min_length',
                            'expected': field_schema.min_length,
                            'message': f"{len(short_values)} values are shorter than minimum length {field_schema.min_length}",
                            'severity': 'medium'
                        })
                
                if field_schema.max_length is not None:
                    long_values = non_null_data[non_null_data.astype(str).str.len() > field_schema.max_length]
                    if not long_values.empty:
                        errors.append({
                            'type': 'constraint_error',
                            'field': field_name,
                            'constraint': 'max_length',
                            'expected': field_schema.max_length,
                            'message': f"{len(long_values)} values exceed maximum length {field_schema.max_length}",
                            'severity': 'medium'
                        })
            
            # Value constraints for numeric fields
            if field_schema.data_type in [DataType.INTEGER, DataType.FLOAT]:
                numeric_data = pd.to_numeric(non_null_data, errors='coerce')
                
                if field_schema.min_value is not None:
                    low_values = numeric_data[numeric_data < field_schema.min_value]
                    if not low_values.empty:
                        errors.append({
                            'type': 'constraint_error',
                            'field': field_name,
                            'constraint': 'min_value',
                            'expected': field_schema.min_value,
                            'message': f"{len(low_values)} values are below minimum {field_schema.min_value}",
                            'severity': 'high'
                        })
                
                if field_schema.max_value is not None:
                    high_values = numeric_data[numeric_data > field_schema.max_value]
                    if not high_values.empty:
                        severity = 'medium' if len(high_values) < len(numeric_data) * 0.1 else 'high'
                        errors.append({
                            'type': 'constraint_error',
                            'field': field_name,
                            'constraint': 'max_value',
                            'expected': field_schema.max_value,
                            'message': f"{len(high_values)} values exceed maximum {field_schema.max_value}",
                            'severity': severity
                        })
            
            # Pattern constraints
            if field_schema.pattern is not None:
                import re
                pattern = re.compile(field_schema.pattern)
                invalid_pattern = 0
                
                for value in non_null_data:
                    if not pattern.match(str(value)):
                        invalid_pattern += 1
                
                if invalid_pattern > 0:
                    errors.append({
                        'type': 'pattern_error',
                        'field': field_name,
                        'pattern': field_schema.pattern,
                        'message': f"{invalid_pattern} values don't match required pattern",
                        'severity': 'medium'
                    })
            
            # Allowed values constraints
            if field_schema.allowed_values is not None:
                allowed_set = set(field_schema.allowed_values)
                invalid_values = non_null_data[~non_null_data.isin(allowed_set)]
                
                if not invalid_values.empty:
                    unique_invalid = invalid_values.unique()
                    errors.append({
                        'type': 'allowed_values_error',
                        'field': field_name,
                        'allowed_values': field_schema.allowed_values,
                        'invalid_values': list(unique_invalid[:5]),  # Show first 5
                        'message': f"{len(invalid_values)} values not in allowed list",
                        'severity': 'medium'
                    })
        
        except Exception as e:
            errors.append({
                'type': 'validation_error',
                'field': field_name,
                'message': f"Error validating field constraints: {str(e)}",
                'severity': 'medium'
            })
        
        return errors, warnings
    
    def add_custom_schema(self, schema_name: str, fields: List[FieldSchema]):
        """Add a custom schema"""
        self.schemas[schema_name] = fields
        logger.info(f"Added custom schema '{schema_name}' with {len(fields)} fields")
    
    def get_schema(self, schema_name: str) -> Optional[List[FieldSchema]]:
        """Get schema by name"""
        return self.schemas.get(schema_name)
    
    def list_schemas(self) -> List[str]:
        """List available schemas"""
        return list(self.schemas.keys())
    
    def export_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Export schema as dictionary"""
        if schema_name not in self.schemas:
            return None
        
        schema = self.schemas[schema_name]
        return {
            'name': schema_name,
            'fields': [
                {
                    'name': field.name,
                    'data_type': field.data_type,
                    'required': field.required,
                    'min_length': field.min_length,
                    'max_length': field.max_length,
                    'min_value': field.min_value,
                    'max_value': field.max_value,
                    'pattern': field.pattern,
                    'allowed_values': field.allowed_values,
                    'description': field.description
                }
                for field in schema
            ]
        }
    
    def import_schema(self, schema_dict: Dict[str, Any]):
        """Import schema from dictionary"""
        schema_name = schema_dict['name']
        fields = []
        
        for field_dict in schema_dict['fields']:
            field = FieldSchema(
                name=field_dict['name'],
                data_type=field_dict['data_type'],
                required=field_dict.get('required', False),
                min_length=field_dict.get('min_length'),
                max_length=field_dict.get('max_length'),
                min_value=field_dict.get('min_value'),
                max_value=field_dict.get('max_value'),
                pattern=field_dict.get('pattern'),
                allowed_values=field_dict.get('allowed_values'),
                description=field_dict.get('description')
            )
            fields.append(field)
        
        self.add_custom_schema(schema_name, fields)
    
    def generate_schema_report(self, result: SchemaValidationResult) -> str:
        """Generate schema validation report"""
        report = []
        report.append("# Schema Validation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Schema**: {result.schema_name}")
        report.append(f"- **Total Fields**: {result.total_fields}")
        report.append(f"- **Valid Fields**: {result.valid_fields}")
        report.append(f"- **Invalid Fields**: {result.invalid_fields}")
        report.append(f"- **Validation Status**: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
        report.append(f"- **Validation Time**: {result.validation_time:.2f} seconds")
        report.append("")
        
        # Errors
        if result.errors:
            report.append("## Errors")
            for i, error in enumerate(result.errors, 1):
                report.append(f"{i}. **{error.get('field', 'N/A')}**: {error.get('message', 'N/A')}")
                if 'severity' in error:
                    report.append(f"   - Severity: {error['severity']}")
            report.append("")
        
        # Warnings
        if result.warnings:
            report.append("## Warnings")
            for i, warning in enumerate(result.warnings, 1):
                report.append(f"{i}. {warning.get('message', 'N/A')}")
            report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test schema validator
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
            'source': 'website',
            'customer_email': 'john@example.com'
        },
        {
            'order_id': 'INVALID',  # Invalid format
            'customer_name': 'A',   # Too short
            'product': 'MacBook Pro',
            'quantity': 0,          # Below minimum
            'price': 60000,         # Above maximum
            'order_date': '2024-01-15',
            'source': 'invalid_source',  # Not in allowed values
            'customer_email': 'invalid-email'
        }
    ])
    
    # Test validator
    validator = SchemaValidator()
    result = validator.validate_schema(test_data, 'orders')
    
    print("Schema Validation Test Results:")
    print(f"Schema: {result.schema_name}")
    print(f"Valid: {result.is_valid}")
    print(f"Valid Fields: {result.valid_fields}/{result.total_fields}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    if result.errors:
        print("\nSample Errors:")
        for error in result.errors[:3]:
            print(f"  - {error.get('field', 'N/A')}: {error.get('message', 'N/A')}")
    
    # Test schema export/import
    schema_dict = validator.export_schema('orders')
    print(f"\nExported schema has {len(schema_dict['fields'])} fields")
    
    # Generate report
    report = validator.generate_schema_report(result)
    print(f"\nSchema report generated ({len(report)} characters)")
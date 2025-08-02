"""
Data standardization module for consistent data formatting
Ensures consistent formats, units, and representations across all data
"""

import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from ..utils.helpers import validate_email, safe_divide
from ..utils.constants import TimeFormat

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class StandardizationResult:
    """Data standardization result container"""
    success: bool
    records_processed: int
    fields_standardized: int
    standardization_operations: List[str]
    standardization_time: float
    data: Optional[pd.DataFrame] = None
    errors: List[str] = None

class DataStandardizer:
    """Data standardizer for consistent formatting and representation"""
    
    def __init__(self):
        """Initialize data standardizer"""
        self.currency_symbols = ['$', '€', '£', '¥', '₹']
        self.phone_patterns = [
            r'^\+?1?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$',  # US format
            r'^(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})$',  # Simple US format
            r'^\+(\d{1,3})[-.\s]?(\d{1,4})[-.\s]?(\d{1,4})[-.\s]?(\d{1,4})$'  # International
        ]
        
        # Standard formats
        self.standard_formats = {
            'phone': '+1-XXX-XXX-XXXX',
            'currency': '$X,XXX.XX',
            'date': 'YYYY-MM-DD',
            'datetime': 'YYYY-MM-DD HH:MM:SS',
            'email': 'lowercase@domain.com',
            'percentage': 'XX.X%'
        }
        
        logger.info("Data standardizer initialized")
    
    def standardize_orders(self, data: pd.DataFrame) -> StandardizationResult:
        """
        Standardize order data formats
        
        Args:
            data (pd.DataFrame): Data to standardize
            
        Returns:
            StandardizationResult: Standardization results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting data standardization for {len(data)} records")
            
            standardized_data = data.copy()
            operations = []
            fields_standardized = 0
            
            # 1. Standardize text fields
            text_fields = self._standardize_text_fields(standardized_data)
            if text_fields['changed']:
                operations.append(f"Standardized text fields: {', '.join(text_fields['fields'])}")
                fields_standardized += len(text_fields['fields'])
            
            # 2. Standardize numeric fields
            numeric_fields = self._standardize_numeric_fields(standardized_data)
            if numeric_fields['changed']:
                operations.append(f"Standardized numeric fields: {', '.join(numeric_fields['fields'])}")
                fields_standardized += len(numeric_fields['fields'])
            
            # 3. Standardize date fields
            date_fields = self._standardize_date_fields(standardized_data)
            if date_fields['changed']:
                operations.append(f"Standardized date fields: {', '.join(date_fields['fields'])}")
                fields_standardized += len(date_fields['fields'])
            
            # 4. Standardize email fields
            email_fields = self._standardize_email_fields(standardized_data)
            if email_fields['changed']:
                operations.append(f"Standardized email fields: {', '.join(email_fields['fields'])}")
                fields_standardized += len(email_fields['fields'])
            
            # 5. Standardize phone fields
            phone_fields = self._standardize_phone_fields(standardized_data)
            if phone_fields['changed']:
                operations.append(f"Standardized phone fields: {', '.join(phone_fields['fields'])}")
                fields_standardized += len(phone_fields['fields'])
            
            # 6. Standardize categorical fields
            categorical_fields = self._standardize_categorical_fields(standardized_data)
            if categorical_fields['changed']:
                operations.append(f"Standardized categorical fields: {', '.join(categorical_fields['fields'])}")
                fields_standardized += len(categorical_fields['fields'])
            
            # 7. Standardize ID fields
            id_fields = self._standardize_id_fields(standardized_data)
            if id_fields['changed']:
                operations.append(f"Standardized ID fields: {', '.join(id_fields['fields'])}")
                fields_standardized += len(id_fields['fields'])
            
            # 8. Add standardization metadata
            standardized_data = self._add_standardization_metadata(standardized_data)
            operations.append("Added standardization metadata")
            
            standardization_time = (datetime.now() - start_time).total_seconds()
            
            result = StandardizationResult(
                success=True,
                records_processed=len(standardized_data),
                fields_standardized=fields_standardized,
                standardization_operations=operations,
                standardization_time=standardization_time,
                data=standardized_data,
                errors=[]
            )
            
            logger.info(f"Data standardization completed: {fields_standardized} fields standardized ({standardization_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during data standardization: {e}")
            return StandardizationResult(
                success=False,
                records_processed=len(data) if data is not None else 0,
                fields_standardized=0,
                standardization_operations=[],
                standardization_time=(datetime.now() - start_time).total_seconds(),
                data=None,
                errors=[str(e)]
            )
    
    def _standardize_text_fields(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Standardize text fields"""
        changed_fields = []
        
        # Customer name standardization
        if 'customer_name' in data.columns:
            original = data['customer_name'].copy()
            # Title case with proper handling of prefixes
            data['customer_name'] = data['customer_name'].str.title()
            # Fix common title case issues
            data['customer_name'] = data['customer_name'].str.replace(r'\bMc([A-Z])', r'Mc\1', regex=True)
            data['customer_name'] = data['customer_name'].str.replace(r'\bO\'([A-Z])', r"O'\1", regex=True)
            data['customer_name'] = data['customer_name'].str.replace(r'\bDe ([A-Z])', r'de \1', regex=True)
            data['customer_name'] = data['customer_name'].str.replace(r'\bVan ([A-Z])', r'van \1', regex=True)
            
            if not data['customer_name'].equals(original):
                changed_fields.append('customer_name')
        
        # Product name standardization
        if 'product' in data.columns:
            original = data['product'].copy()
            # Consistent capitalization for product names
            data['product'] = data['product'].str.strip()
            # Remove extra spaces
            data['product'] = data['product'].str.replace(r'\s+', ' ', regex=True)
            
            if not data['product'].equals(original):
                changed_fields.append('product')
        
        # Address standardization
        address_fields = ['address', 'store_location', 'shipping_address', 'billing_address']
        for field in address_fields:
            if field in data.columns:
                original = data[field].copy()
                # Standardize address format
                data[field] = data[field].str.title()
                # Standardize common abbreviations
                address_replacements = {
                    r'\bSt\.?\b': 'Street',
                    r'\bAve\.?\b': 'Avenue',
                    r'\bBlvd\.?\b': 'Boulevard',
                    r'\bDr\.?\b': 'Drive',
                    r'\bRd\.?\b': 'Road',
                    r'\bLn\.?\b': 'Lane',
                    r'\bCt\.?\b': 'Court',
                    r'\bPl\.?\b': 'Place',
                    r'\bApt\.?\b': 'Apartment',
                    r'\bSte\.?\b': 'Suite'
                }
                
                for pattern, replacement in address_replacements.items():
                    data[field] = data[field].str.replace(pattern, replacement, regex=True)
                
                if not data[field].equals(original):
                    changed_fields.append(field)
        
        # Notes and description fields
        text_fields = ['notes', 'description', 'comments']
        for field in text_fields:
            if field in data.columns:
                original = data[field].copy()
                # Sentence case for notes
                data[field] = data[field].str.strip()
                data[field] = data[field].str.replace(r'\s+', ' ', regex=True)
                # Capitalize first letter of sentences
                data[field] = data[field].str.replace(r'(^|[.!?]\s+)([a-z])', 
                                                    lambda m: m.group(1) + m.group(2).upper(), 
                                                    regex=True)
                
                if not data[field].equals(original):
                    changed_fields.append(field)
        
        return {'changed': len(changed_fields) > 0, 'fields': changed_fields}
    
    def _standardize_numeric_fields(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Standardize numeric fields"""
        changed_fields = []
        
        # Price fields - standardize to 2 decimal places
        price_fields = ['price', 'unit_price', 'total_amount', 'discount', 'tax_amount', 'shipping_cost']
        for field in price_fields:
            if field in data.columns:
                original = data[field].copy()
                # Remove currency symbols if present
                if data[field].dtype == 'object':
                    for symbol in self.currency_symbols:
                        data[field] = data[field].astype(str).str.replace(symbol, '', regex=False)
                    # Remove commas
                    data[field] = data[field].str.replace(',', '')
                    # Convert to numeric
                    data[field] = pd.to_numeric(data[field], errors='coerce')
                
                # Round to 2 decimal places
                data[field] = data[field].round(2)
                
                if not data[field].equals(original):
                    changed_fields.append(field)
        
        # Quantity fields - standardize to integers
        quantity_fields = ['quantity', 'items_count', 'units_ordered']
        for field in quantity_fields:
            if field in data.columns:
                original = data[field].copy()
                # Convert to integer
                data[field] = pd.to_numeric(data[field], errors='coerce')
                data[field] = data[field].round().astype('Int64')  # Nullable integer
                
                if not data[field].equals(original):
                    changed_fields.append(field)
        
        # Percentage fields
        percentage_fields = ['discount_percentage', 'tax_rate', 'commission_rate']
        for field in percentage_fields:
            if field in data.columns:
                original = data[field].copy()
                # Remove % symbol if present
                if data[field].dtype == 'object':
                    data[field] = data[field].astype(str).str.replace('%', '')
                    data[field] = pd.to_numeric(data[field], errors='coerce')
                
                # Ensure percentages are in 0-100 range (not 0-1)
                mask = (data[field] <= 1) & (data[field] >= 0)
                data.loc[mask, field] = data.loc[mask, field] * 100
                
                # Round to 1 decimal place
                data[field] = data[field].round(1)
                
                if not data[field].equals(original):
                    changed_fields.append(field)
        
        return {'changed': len(changed_fields) > 0, 'fields': changed_fields}
    
    def _standardize_date_fields(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Standardize date and datetime fields"""
        changed_fields = []
        
        # Date fields
        date_fields = ['order_date', 'delivery_date', 'ship_date', 'created_date']
        for field in date_fields:
            if field in data.columns:
                original = data[field].copy()
                # Convert to datetime and then to standard date format
                data[field] = pd.to_datetime(data[field], errors='coerce')
                # Convert to date string in ISO format
                data[field] = data[field].dt.strftime(TimeFormat.ISO_DATE)
                
                if not data[field].equals(original.astype(str)):
                    changed_fields.append(field)
        
        # Datetime fields
        datetime_fields = ['created_at', 'updated_at', 'processed_at', 'ingested_at', 'enriched_at', 'cleaned_at']
        for field in datetime_fields:
            if field in data.columns:
                original = data[field].copy()
                # Convert to datetime and then to standard datetime format
                if data[field].dtype == 'object':
                    data[field] = pd.to_datetime(data[field], errors='coerce')
                # Convert to datetime string in ISO format
                data[field] = data[field].dt.strftime(TimeFormat.ISO_DATETIME)
                
                if not data[field].equals(original.astype(str)):
                    changed_fields.append(field)
        
        return {'changed': len(changed_fields) > 0, 'fields': changed_fields}
    
    def _standardize_email_fields(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Standardize email fields"""
        changed_fields = []
        
        email_fields = ['customer_email', 'contact_email', 'billing_email', 'shipping_email']
        for field in email_fields:
            if field in data.columns:
                original = data[field].copy()
                # Convert to lowercase and trim
                data[field] = data[field].astype(str).str.lower().str.strip()
                # Replace 'nan' with actual NaN
                data[field] = data[field].replace('nan', np.nan)
                
                if not data[field].equals(original):
                    changed_fields.append(field)
        
        return {'changed': len(changed_fields) > 0, 'fields': changed_fields}
    
    def _standardize_phone_fields(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Standardize phone number fields"""
        changed_fields = []
        
        phone_fields = ['phone', 'customer_phone', 'contact_phone', 'mobile_phone']
        for field in phone_fields:
            if field in data.columns:
                original = data[field].copy()
                data[field] = data[field].astype(str).apply(self._format_phone_number)
                
                if not data[field].equals(original):
                    changed_fields.append(field)
        
        return {'changed': len(changed_fields) > 0, 'fields': changed_fields}
    
    def _format_phone_number(self, phone: str) -> str:
        """Format phone number to standard format"""
        if pd.isna(phone) or phone == 'nan' or phone == '':
            return np.nan
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', str(phone))
        
        # Handle different phone number lengths
        if len(digits) == 10:
            # US phone number without country code
            return f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits.startswith('1'):
            # US phone number with country code
            return f"+1-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
        elif len(digits) >= 7:
            # International or other format - keep as is with + prefix
            return f"+{digits}"
        else:
            # Invalid phone number
            return np.nan
    
    def _standardize_categorical_fields(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Standardize categorical fields"""
        changed_fields = []
        
        # Source field
        if 'source' in data.columns:
            original = data['source'].copy()
            # Standardize source values
            source_mapping = {
                'web': 'website',
                'online': 'website',
                'internet': 'website',
                'app': 'mobile_app',
                'mobile': 'mobile_app',
                'smartphone': 'mobile_app',
                'retail': 'store',
                'shop': 'store',
                'physical': 'store',
                'call': 'phone',
                'telephone': 'phone',
                'tel': 'phone',
                'partner': 'partner',
                'affiliate': 'partner',
                'reseller': 'partner'
            }
            
            data['source'] = data['source'].str.lower().replace(source_mapping)
            
            if not data['source'].equals(original):
                changed_fields.append('source')
        
        # Status fields
        status_fields = ['order_status', 'payment_status', 'shipping_status']
        for field in status_fields:
            if field in data.columns:
                original = data[field].copy()
                # Standardize status values to title case
                data[field] = data[field].str.lower().str.replace('_', ' ').str.title()
                
                if not data[field].equals(original):
                    changed_fields.append(field)
        
        # Boolean fields
        boolean_fields = ['is_gift', 'is_express', 'is_international', 'is_business']
        for field in boolean_fields:
            if field in data.columns:
                original = data[field].copy()
                # Standardize boolean values
                boolean_mapping = {
                    'yes': True, 'y': True, '1': True, 'true': True, 'on': True,
                    'no': False, 'n': False, '0': False, 'false': False, 'off': False
                }
                
                data[field] = data[field].astype(str).str.lower().replace(boolean_mapping)
                data[field] = data[field].astype(bool)
                
                if not data[field].equals(original):
                    changed_fields.append(field)
        
        return {'changed': len(changed_fields) > 0, 'fields': changed_fields}
    
    def _standardize_id_fields(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Standardize ID fields"""
        changed_fields = []
        
        # Order ID
        if 'order_id' in data.columns:
            original = data['order_id'].copy()
            # Ensure order IDs are uppercase
            data['order_id'] = data['order_id'].astype(str).str.upper().str.strip()
            
            if not data['order_id'].equals(original):
                changed_fields.append('order_id')
        
        # Customer ID
        if 'customer_id' in data.columns:
            original = data['customer_id'].copy()
            # Ensure customer IDs are uppercase
            data['customer_id'] = data['customer_id'].astype(str).str.upper().str.strip()
            
            if not data['customer_id'].equals(original):
                changed_fields.append('customer_id')
        
        # Product ID
        if 'product_id' in data.columns:
            original = data['product_id'].copy()
            # Ensure product IDs are uppercase
            data['product_id'] = data['product_id'].astype(str).str.upper().str.strip()
            
            if not data['product_id'].equals(original):
                changed_fields.append('product_id')
        
        return {'changed': len(changed_fields) > 0, 'fields': changed_fields}
    
    def _add_standardization_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add metadata about standardization"""
        data['standardized_at'] = datetime.now().strftime(TimeFormat.ISO_DATETIME)
        data['standardization_version'] = '1.0'
        
        return data
    
    def validate_standardization(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate that data meets standardization requirements"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'field_validations': {}
        }
        
        # Validate date formats
        date_fields = ['order_date', 'delivery_date', 'ship_date']
        for field in date_fields:
            if field in data.columns:
                try:
                    pd.to_datetime(data[field], format=TimeFormat.ISO_DATE, errors='raise')
                    validation_results['field_validations'][field] = 'Valid'
                except:
                    validation_results['is_valid'] = False
                    validation_results['issues'].append(f"Invalid date format in {field}")
                    validation_results['field_validations'][field] = 'Invalid'
        
        # Validate email formats
        email_fields = ['customer_email', 'contact_email']
        for field in email_fields:
            if field in data.columns:
                valid_emails = data[field].dropna().apply(validate_email).all()
                if valid_emails:
                    validation_results['field_validations'][field] = 'Valid'
                else:
                    validation_results['is_valid'] = False
                    validation_results['issues'].append(f"Invalid email format in {field}")
                    validation_results['field_validations'][field] = 'Invalid'
        
        # Validate numeric formats
        numeric_fields = ['price', 'total_amount', 'discount']
        for field in numeric_fields:
            if field in data.columns:
                if pd.api.types.is_numeric_dtype(data[field]):
                    validation_results['field_validations'][field] = 'Valid'
                else:
                    validation_results['is_valid'] = False
                    validation_results['issues'].append(f"Non-numeric values in {field}")
                    validation_results['field_validations'][field] = 'Invalid'
        
        return validation_results
    
    def generate_standardization_report(self, result: StandardizationResult) -> str:
        """Generate detailed standardization report"""
        report = []
        report.append("# Data Standardization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Records Processed**: {result.records_processed:,}")
        report.append(f"- **Fields Standardized**: {result.fields_standardized}")
        report.append(f"- **Standardization Time**: {result.standardization_time:.2f} seconds")
        report.append(f"- **Status**: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        report.append("")
        
        # Operations performed
        if result.standardization_operations:
            report.append("## Standardization Operations")
            for i, operation in enumerate(result.standardization_operations, 1):
                report.append(f"{i}. {operation}")
            report.append("")
        
        # Standard formats applied
        report.append("## Standard Formats Applied")
        for format_type, format_example in self.standard_formats.items():
            report.append(f"- **{format_type.title()}**: {format_example}")
        report.append("")
        
        # Validation results
        if result.data is not None:
            validation = self.validate_standardization(result.data)
            report.append("## Validation Results")
            report.append(f"- **Overall Status**: {'✅ VALID' if validation['is_valid'] else '❌ INVALID'}")
            
            if validation['field_validations']:
                report.append("- **Field Validations**:")
                for field, status in validation['field_validations'].items():
                    status_icon = '✅' if status == 'Valid' else '❌'
                    report.append(f"  - {field}: {status_icon} {status}")
            
            if validation['issues']:
                report.append("- **Issues Found**:")
                for issue in validation['issues']:
                    report.append(f"  - {issue}")
            report.append("")
        
        # Errors
        if result.errors:
            report.append("## Errors")
            for i, error in enumerate(result.errors, 1):
                report.append(f"{i}. {error}")
            report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test data standardizer
    import pandas as pd
    
    # Create test data with various formatting issues
    test_data = pd.DataFrame([
        {
            'order_id': 'ord-2024-001',
            'customer_name': 'john doe',
            'customer_email': 'JOHN@EXAMPLE.COM',
            'phone': '555-123-4567',
            'price': '$999.99',
            'discount_percentage': '10%',
            'order_date': '2024-01-15',
            'source': 'web',
            'address': '123 main st.',
            'created_at': '2024-01-15 10:30:00'
        },
        {
            'order_id': 'ORD-2024-002',
            'customer_name': 'jane o\'connor',
            'customer_email': 'jane@COMPANY.COM',
            'phone': '(555) 987-6543',
            'price': '1,999.99',
            'discount_percentage': '0.05',  # As decimal
            'order_date': '01/16/2024',
            'source': 'mobile',
            'address': '456 oak ave apt 2b',
            'created_at': '2024-01-16T14:45:30'
        }
    ])
    
    # Test standardizer
    standardizer = DataStandardizer()
    result = standardizer.standardize_orders(test_data)
    
    print("Data Standardization Test Results:")
    print(f"Success: {result.success}")
    print(f"Records Processed: {result.records_processed}")
    print(f"Fields Standardized: {result.fields_standardized}")
    print(f"Standardization Time: {result.standardization_time:.2f}s")
    
    if result.data is not None:
        print(f"\nStandardized Data Sample:")
        display_fields = ['customer_name', 'customer_email', 'phone', 'price', 'order_date', 'source']
        available_fields = [field for field in display_fields if field in result.data.columns]
        if available_fields:
            print(result.data[available_fields].head())
    
    # Validate standardization
    if result.data is not None:
        validation = standardizer.validate_standardization(result.data)
        print(f"\nValidation Status: {'✅ VALID' if validation['is_valid'] else '❌ INVALID'}")
        if validation['issues']:
            print("Issues found:")
            for issue in validation['issues']:
                print(f"  - {issue}")
    
    # Generate report
    report = standardizer.generate_standardization_report(result)
    print(f"\nStandardization report generated ({len(report)} characters)")
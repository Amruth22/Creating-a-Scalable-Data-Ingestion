"""
Data standardization module for format normalization and value standardization
Ensures consistent formats, units, and representations across all data fields
"""

import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..utils.helpers import safe_divide, validate_email
from ..utils.constants import DataSourceType

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class StandardizationResult:
    """Data standardization result container"""
    success: bool
    records_processed: int
    fields_standardized: int
    standardization_time: float
    data: Optional[pd.DataFrame] = None
    errors: List[str] = None

class DataStandardizer:
    """Data standardizer for format normalization and value standardization"""
    
    def __init__(self):
        """Initialize data standardizer"""
        self.phone_patterns = self._load_phone_patterns()
        self.email_domains = self._load_email_domains()
        self.address_abbreviations = self._load_address_abbreviations()
        self.currency_symbols = self._load_currency_symbols()
        
        logger.info("Data standardizer initialized")
    
    def _load_phone_patterns(self) -> Dict[str, str]:
        """Load phone number standardization patterns"""
        return {
            r'^\+?1?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$': r'+1-\1-\2-\3',
            r'^(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})$': r'+1-\1-\2-\3',
            r'^\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$': r'+1-\1-\2-\3'
        }
    
    def _load_email_domains(self) -> Dict[str, str]:
        """Load email domain standardization mappings"""
        return {
            'gmail.com': 'gmail.com',
            'googlemail.com': 'gmail.com',
            'yahoo.com': 'yahoo.com',
            'ymail.com': 'yahoo.com',
            'hotmail.com': 'outlook.com',
            'live.com': 'outlook.com',
            'msn.com': 'outlook.com'
        }
    
    def _load_address_abbreviations(self) -> Dict[str, str]:
        """Load address standardization mappings"""
        return {
            # Street types
            'st': 'Street',
            'street': 'Street',
            'ave': 'Avenue',
            'avenue': 'Avenue',
            'blvd': 'Boulevard',
            'boulevard': 'Boulevard',
            'rd': 'Road',
            'road': 'Road',
            'dr': 'Drive',
            'drive': 'Drive',
            'ln': 'Lane',
            'lane': 'Lane',
            'ct': 'Court',
            'court': 'Court',
            'pl': 'Place',
            'place': 'Place',
            
            # Directions
            'n': 'North',
            'north': 'North',
            's': 'South',
            'south': 'South',
            'e': 'East',
            'east': 'East',
            'w': 'West',
            'west': 'West',
            'ne': 'Northeast',
            'nw': 'Northwest',
            'se': 'Southeast',
            'sw': 'Southwest',
            
            # States (common abbreviations)
            'ca': 'California',
            'ny': 'New York',
            'tx': 'Texas',
            'fl': 'Florida',
            'il': 'Illinois',
            'pa': 'Pennsylvania',
            'oh': 'Ohio',
            'ga': 'Georgia',
            'nc': 'North Carolina',
            'mi': 'Michigan'
        }
    
    def _load_currency_symbols(self) -> Dict[str, str]:
        """Load currency symbol mappings"""
        return {
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '₹': 'INR',
            '₽': 'RUB',
            '₩': 'KRW',
            '¢': 'USD_CENTS'
        }
    
    def standardize_orders(self, data: pd.DataFrame) -> StandardizationResult:
        """
        Standardize order data formats and values
        
        Args:
            data (pd.DataFrame): Order data to standardize
            
        Returns:
            StandardizationResult: Standardization results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting data standardization for {len(data)} records")
            
            standardized_data = data.copy()
            fields_standardized = 0
            
            # 1. Standardize text fields
            text_fields = ['customer_name', 'product', 'store_location', 'notes']
            for field in text_fields:
                if field in standardized_data.columns:
                    standardized_data = self._standardize_text_field(standardized_data, field)
                    fields_standardized += 1
            
            # 2. Standardize email addresses
            if 'customer_email' in standardized_data.columns:
                standardized_data = self._standardize_email_addresses(standardized_data)
                fields_standardized += 1
            
            # 3. Standardize phone numbers
            phone_fields = ['customer_phone', 'contact_phone', 'phone']
            for field in phone_fields:
                if field in standardized_data.columns:
                    standardized_data = self._standardize_phone_numbers(standardized_data, field)
                    fields_standardized += 1
            
            # 4. Standardize addresses
            address_fields = ['address', 'billing_address', 'shipping_address', 'store_location']
            for field in address_fields:
                if field in standardized_data.columns:
                    standardized_data = self._standardize_addresses(standardized_data, field)
                    fields_standardized += 1
            
            # 5. Standardize dates
            date_fields = ['order_date', 'delivery_date', 'created_at', 'updated_at']
            for field in date_fields:
                if field in standardized_data.columns:
                    standardized_data = self._standardize_dates(standardized_data, field)
                    fields_standardized += 1
            
            # 6. Standardize numeric fields
            numeric_fields = ['price', 'quantity', 'discount', 'total_amount']
            for field in numeric_fields:
                if field in standardized_data.columns:
                    standardized_data = self._standardize_numeric_field(standardized_data, field)
                    fields_standardized += 1
            
            # 7. Standardize categorical fields
            categorical_fields = ['source', 'product_category', 'customer_segment']
            for field in categorical_fields:
                if field in standardized_data.columns:
                    standardized_data = self._standardize_categorical_field(standardized_data, field)
                    fields_standardized += 1
            
            # 8. Standardize order IDs
            if 'order_id' in standardized_data.columns:
                standardized_data = self._standardize_order_ids(standardized_data)
                fields_standardized += 1
            
            # 9. Add standardization metadata
            standardized_data = self._add_standardization_metadata(standardized_data)
            
            standardization_time = (datetime.now() - start_time).total_seconds()
            
            result = StandardizationResult(
                success=True,
                records_processed=len(standardized_data),
                fields_standardized=fields_standardized,
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
                records_processed=0,
                fields_standardized=0,
                standardization_time=(datetime.now() - start_time).total_seconds(),
                data=None,
                errors=[str(e)]
            )
    
    def _standardize_text_field(self, data: pd.DataFrame, field: str) -> pd.DataFrame:
        """Standardize text field formatting"""
        if field not in data.columns:
            return data
        
        # Convert to string and handle NaN
        data[field] = data[field].astype(str)
        data[field] = data[field].replace('nan', np.nan)
        
        # Basic text cleaning
        data[field] = data[field].str.strip()  # Remove leading/trailing whitespace
        data[field] = data[field].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
        
        # Field-specific standardization
        if field == 'customer_name':
            # Title case for names
            data[field] = data[field].str.title()
            # Fix common name issues
            data[field] = data[field].str.replace(r'\bMc([a-z])', r'Mc\1', regex=True)  # McDonald -> McDonald
            data[field] = data[field].str.replace(r'\bO\'([a-z])', r"O'\1", regex=True)  # O'connor -> O'Connor
        
        elif field == 'product':
            # Proper case for products
            data[field] = data[field].str.title()
            # Fix brand names
            brand_fixes = {
                'Iphone': 'iPhone',
                'Ipad': 'iPad',
                'Macbook': 'MacBook',
                'Airpods': 'AirPods',
                'Playstation': 'PlayStation',
                'Xbox': 'Xbox'
            }
            for wrong, correct in brand_fixes.items():
                data[field] = data[field].str.replace(wrong, correct, case=False)
        
        elif field in ['store_location', 'notes']:
            # Title case for locations and notes
            data[field] = data[field].str.title()
        
        return data
    
    def _standardize_email_addresses(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize email address formats"""
        email_field = 'customer_email'
        if email_field not in data.columns:
            return data
        
        # Convert to lowercase
        data[email_field] = data[email_field].str.lower().str.strip()
        
        # Remove extra spaces
        data[email_field] = data[email_field].str.replace(r'\s+', '', regex=True)
        
        # Standardize domains
        for domain, standard_domain in self.email_domains.items():
            pattern = f'@{re.escape(domain)}$'
            replacement = f'@{standard_domain}'
            data[email_field] = data[email_field].str.replace(pattern, replacement, regex=True)
        
        # Validate and clean invalid emails
        valid_email_mask = data[email_field].apply(validate_email)
        data.loc[~valid_email_mask, email_field] = np.nan
        
        return data
    
    def _standardize_phone_numbers(self, data: pd.DataFrame, field: str) -> pd.DataFrame:
        """Standardize phone number formats"""
        if field not in data.columns:
            return data
        
        # Convert to string and clean
        data[field] = data[field].astype(str).str.strip()
        data[field] = data[field].replace('nan', np.nan)
        
        # Remove non-digit characters except + and -
        data[field] = data[field].str.replace(r'[^\d+\-]', '', regex=True)
        
        # Apply standardization patterns
        for pattern, replacement in self.phone_patterns.items():
            mask = data[field].str.match(pattern, na=False)
            if mask.any():
                data.loc[mask, field] = data.loc[mask, field].str.replace(pattern, replacement, regex=True)
        
        return data
    
    def _standardize_addresses(self, data: pd.DataFrame, field: str) -> pd.DataFrame:
        """Standardize address formats"""
        if field not in data.columns:
            return data
        
        # Convert to string and clean
        data[field] = data[field].astype(str).str.strip()
        data[field] = data[field].replace('nan', np.nan)
        
        # Title case
        data[field] = data[field].str.title()
        
        # Standardize abbreviations (case-insensitive)
        for abbrev, full_form in self.address_abbreviations.items():
            # Word boundary patterns to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            data[field] = data[field].str.replace(pattern, full_form, case=False, regex=True)
        
        # Clean up extra spaces
        data[field] = data[field].str.replace(r'\s+', ' ', regex=True)
        
        return data
    
    def _standardize_dates(self, data: pd.DataFrame, field: str) -> pd.DataFrame:
        """Standardize date formats"""
        if field not in data.columns:
            return data
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data[field]):
            data[field] = pd.to_datetime(data[field], errors='coerce')
        
        # Standardize to ISO format string
        data[field] = data[field].dt.strftime('%Y-%m-%d')
        
        return data
    
    def _standardize_numeric_field(self, data: pd.DataFrame, field: str) -> pd.DataFrame:
        """Standardize numeric field formats"""
        if field not in data.columns:
            return data
        
        # Remove currency symbols and convert to numeric
        if data[field].dtype == 'object':
            # Remove currency symbols
            for symbol in self.currency_symbols.keys():
                data[field] = data[field].astype(str).str.replace(re.escape(symbol), '', regex=True)
            
            # Remove commas and other formatting
            data[field] = data[field].str.replace(',', '')
            data[field] = data[field].str.replace(' ', '')
            
            # Convert to numeric
            data[field] = pd.to_numeric(data[field], errors='coerce')
        
        # Round to appropriate decimal places
        if field in ['price', 'discount', 'total_amount']:
            data[field] = data[field].round(2)
        elif field == 'quantity':
            data[field] = data[field].round(0).astype('Int64')  # Integer with NaN support
        
        return data
    
    def _standardize_categorical_field(self, data: pd.DataFrame, field: str) -> pd.DataFrame:
        """Standardize categorical field values"""
        if field not in data.columns:
            return data
        
        # Convert to string and clean
        data[field] = data[field].astype(str).str.strip().str.lower()
        data[field] = data[field].replace('nan', np.nan)
        
        # Field-specific standardization
        if field == 'source':
            source_mapping = {
                'web': 'website',
                'online': 'website',
                'internet': 'website',
                'app': 'mobile_app',
                'mobile': 'mobile_app',
                'phone': 'phone',
                'call': 'phone',
                'telephone': 'phone',
                'retail': 'store',
                'shop': 'store',
                'physical': 'store',
                'partner': 'partner',
                'affiliate': 'partner'
            }
            data[field] = data[field].replace(source_mapping)
        
        elif field == 'product_category':
            category_mapping = {
                'electronic': 'electronics',
                'tech': 'electronics',
                'technology': 'electronics',
                'game': 'gaming',
                'games': 'gaming',
                'book': 'books',
                'reading': 'books',
                'home': 'smart_home',
                'smart home': 'smart_home',
                'fitness': 'health_fitness',
                'health': 'health_fitness',
                'sport': 'health_fitness'
            }
            data[field] = data[field].replace(category_mapping)
            # Title case for final result
            data[field] = data[field].str.title()
        
        elif field == 'customer_segment':
            segment_mapping = {
                'vip': 'VIP',
                'premium': 'Premium',
                'standard': 'Standard',
                'regular': 'Standard',
                'budget': 'Budget',
                'basic': 'Budget'
            }
            data[field] = data[field].replace(segment_mapping)
        
        return data
    
    def _standardize_order_ids(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize order ID formats"""
        if 'order_id' not in data.columns:
            return data
        
        # Convert to string and clean
        data['order_id'] = data['order_id'].astype(str).str.strip().str.upper()
        
        # Standardize format to XXX-YYYY-NNN
        # Handle various formats
        patterns = [
            (r'^([A-Z]{3})(\d{4})(\d{3})$', r'\1-\2-\3'),  # ORDYYYYNNN -> ORD-YYYY-NNN
            (r'^([A-Z]{3})[-_](\d{4})[-_](\d{3})$', r'\1-\2-\3'),  # ORD_YYYY_NNN -> ORD-YYYY-NNN
            (r'^([A-Z]{3})\s+(\d{4})\s+(\d{3})$', r'\1-\2-\3'),  # ORD YYYY NNN -> ORD-YYYY-NNN
        ]
        
        for pattern, replacement in patterns:
            mask = data['order_id'].str.match(pattern, na=False)
            if mask.any():
                data.loc[mask, 'order_id'] = data.loc[mask, 'order_id'].str.replace(pattern, replacement, regex=True)
        
        return data
    
    def _add_standardization_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add metadata about the standardization process"""
        data['standardized_at'] = datetime.now().isoformat()
        data['standardization_version'] = '1.0'
        
        return data
    
    def validate_standardization(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate standardization results"""
        validation_results = {
            'total_records': len(data),
            'validation_checks': {},
            'issues': [],
            'overall_score': 0
        }
        
        checks_passed = 0
        total_checks = 0
        
        # Check order ID format
        if 'order_id' in data.columns:
            total_checks += 1
            valid_order_ids = data['order_id'].str.match(r'^[A-Z]{3}-\d{4}-\d{3}$', na=False).sum()
            validation_results['validation_checks']['order_id_format'] = {
                'valid': valid_order_ids,
                'total': len(data),
                'percentage': safe_divide(valid_order_ids, len(data), 0) * 100
            }
            if valid_order_ids / len(data) > 0.9:  # 90% threshold
                checks_passed += 1
            else:
                validation_results['issues'].append(f"Order ID format issues: {len(data) - valid_order_ids} records")
        
        # Check email format
        if 'customer_email' in data.columns:
            total_checks += 1
            valid_emails = data['customer_email'].apply(lambda x: validate_email(x) if pd.notna(x) else True).sum()
            validation_results['validation_checks']['email_format'] = {
                'valid': valid_emails,
                'total': data['customer_email'].notna().sum(),
                'percentage': safe_divide(valid_emails, data['customer_email'].notna().sum(), 0) * 100
            }
            if valid_emails / data['customer_email'].notna().sum() > 0.95:  # 95% threshold
                checks_passed += 1
            else:
                validation_results['issues'].append(f"Email format issues: {data['customer_email'].notna().sum() - valid_emails} records")
        
        # Check numeric fields
        numeric_fields = ['price', 'quantity', 'total_amount']
        for field in numeric_fields:
            if field in data.columns:
                total_checks += 1
                valid_numeric = pd.to_numeric(data[field], errors='coerce').notna().sum()
                validation_results['validation_checks'][f'{field}_numeric'] = {
                    'valid': valid_numeric,
                    'total': len(data),
                    'percentage': safe_divide(valid_numeric, len(data), 0) * 100
                }
                if valid_numeric / len(data) > 0.95:  # 95% threshold
                    checks_passed += 1
                else:
                    validation_results['issues'].append(f"{field} numeric format issues: {len(data) - valid_numeric} records")
        
        # Calculate overall score
        validation_results['overall_score'] = safe_divide(checks_passed, total_checks, 0) * 100
        
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
        
        # Validation results
        if result.data is not None:
            validation = self.validate_standardization(result.data)
            report.append("## Validation Results")
            report.append(f"- **Overall Score**: {validation['overall_score']:.1f}%")
            report.append(f"- **Total Records**: {validation['total_records']:,}")
            
            if validation['validation_checks']:
                report.append("\n### Field Validation")
                for check_name, check_result in validation['validation_checks'].items():
                    report.append(f"- **{check_name.replace('_', ' ').title()}**: {check_result['valid']}/{check_result['total']} ({check_result['percentage']:.1f}%)")
            
            if validation['issues']:
                report.append("\n### Issues Found")
                for i, issue in enumerate(validation['issues'], 1):
                    report.append(f"{i}. {issue}")
            
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
            'order_id': 'ord2024001',
            'customer_name': 'john doe',
            'customer_email': 'JOHN@GMAIL.COM',
            'product': 'iphone 15',
            'price': '$999.99',
            'quantity': '1.0',
            'order_date': '2024-01-15',
            'source': 'web',
            'store_location': '123 main st, ny'
        },
        {
            'order_id': 'ORD_2024_002',
            'customer_name': 'jane o\'connor',
            'customer_email': 'jane@hotmail.com',
            'product': 'macbook pro',
            'price': '1,999.99',
            'quantity': '1',
            'order_date': '01/16/2024',
            'source': 'retail',
            'store_location': '456 oak ave, ca'
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
        sample_cols = ['order_id', 'customer_name', 'customer_email', 'product', 'price', 'source']
        available_cols = [col for col in sample_cols if col in result.data.columns]
        print(result.data[available_cols].head())
    
    # Validate standardization
    if result.data is not None:
        validation = standardizer.validate_standardization(result.data)
        print(f"\nValidation Score: {validation['overall_score']:.1f}%")
        if validation['issues']:
            print("Issues found:")
            for issue in validation['issues']:
                print(f"  - {issue}")
    
    # Generate report
    report = standardizer.generate_standardization_report(result)
    print(f"\nStandardization report generated ({len(report)} characters)")
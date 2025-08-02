"""
Data cleaning module for data quality improvement
Handles data cleaning, deduplication, and basic transformations
"""

import pandas as pd
import numpy as np
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils.helpers import safe_divide, validate_email
from ..utils.constants import DataSourceType

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class CleaningResult:
    """Data cleaning result container"""
    success: bool
    original_records: int
    cleaned_records: int
    removed_records: int
    cleaning_operations: List[str]
    cleaning_time: float
    data: Optional[pd.DataFrame] = None
    errors: List[str] = None

class DataCleaner:
    """Comprehensive data cleaner for order data"""
    
    def __init__(self):
        """Initialize data cleaner"""
        self.cleaning_operations = []
        self.product_mappings = self._load_product_mappings()
        self.location_mappings = self._load_location_mappings()
        
        logger.info("Data cleaner initialized")
    
    def _load_product_mappings(self) -> Dict[str, str]:
        """Load product name standardization mappings"""
        return {
            # Apple products
            'iphone 15': 'iPhone 15',
            'iphone15': 'iPhone 15',
            'iphone-15': 'iPhone 15',
            'apple iphone 15': 'iPhone 15',
            'macbook pro': 'MacBook Pro',
            'macbookpro': 'MacBook Pro',
            'macbook-pro': 'MacBook Pro',
            'apple macbook pro': 'MacBook Pro',
            'airpods pro': 'AirPods Pro',
            'airpodspro': 'AirPods Pro',
            'air pods pro': 'AirPods Pro',
            'apple airpods pro': 'AirPods Pro',
            'ipad air': 'iPad Air',
            'ipadair': 'iPad Air',
            'ipad-air': 'iPad Air',
            'apple ipad air': 'iPad Air',
            'apple watch': 'Apple Watch',
            'applewatch': 'Apple Watch',
            'apple-watch': 'Apple Watch',
            
            # Samsung products
            'samsung galaxy s24': 'Samsung Galaxy S24',
            'galaxy s24': 'Samsung Galaxy S24',
            'galaxys24': 'Samsung Galaxy S24',
            's24': 'Samsung Galaxy S24',
            
            # Other products
            'nintendo switch': 'Nintendo Switch',
            'nintendoswitch': 'Nintendo Switch',
            'switch': 'Nintendo Switch',
            'playstation 5': 'PlayStation 5',
            'playstation5': 'PlayStation 5',
            'ps5': 'PlayStation 5',
            'xbox series x': 'Xbox Series X',
            'xboxseriesx': 'Xbox Series X',
            'xbox x': 'Xbox Series X',
            'kindle paperwhite': 'Kindle Paperwhite',
            'kindlepaperwhite': 'Kindle Paperwhite',
            'kindle': 'Kindle Paperwhite'
        }
    
    def _load_location_mappings(self) -> Dict[str, str]:
        """Load location standardization mappings"""
        return {
            'ny': 'New York',
            'nyc': 'New York',
            'new york city': 'New York',
            'la': 'Los Angeles',
            'los angeles ca': 'Los Angeles',
            'chicago il': 'Chicago',
            'houston tx': 'Houston',
            'phoenix az': 'Phoenix',
            'philadelphia pa': 'Philadelphia',
            'san antonio tx': 'San Antonio',
            'san diego ca': 'San Diego',
            'dallas tx': 'Dallas',
            'san jose ca': 'San Jose'
        }
    
    def clean_orders(self, data: pd.DataFrame) -> CleaningResult:
        """
        Clean order data comprehensively
        
        Args:
            data (pd.DataFrame): Raw order data
            
        Returns:
            CleaningResult: Cleaning results with cleaned data
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting data cleaning for {len(data)} records")
            
            original_records = len(data)
            cleaned_data = data.copy()
            operations = []
            
            # 1. Remove exact duplicates
            before_dedup = len(cleaned_data)
            cleaned_data = self._remove_duplicates(cleaned_data)
            after_dedup = len(cleaned_data)
            if before_dedup != after_dedup:
                operations.append(f"Removed {before_dedup - after_dedup} duplicate records")
            
            # 2. Clean and standardize text fields
            cleaned_data = self._clean_text_fields(cleaned_data)
            operations.append("Cleaned and standardized text fields")
            
            # 3. Fix data types
            cleaned_data = self._fix_data_types(cleaned_data)
            operations.append("Fixed data types")
            
            # 4. Handle missing values
            before_missing = len(cleaned_data)
            cleaned_data = self._handle_missing_values(cleaned_data)
            after_missing = len(cleaned_data)
            if before_missing != after_missing:
                operations.append(f"Handled missing values, removed {before_missing - after_missing} records")
            
            # 5. Clean numeric fields
            cleaned_data = self._clean_numeric_fields(cleaned_data)
            operations.append("Cleaned numeric fields")
            
            # 6. Clean date fields
            cleaned_data = self._clean_date_fields(cleaned_data)
            operations.append("Cleaned date fields")
            
            # 7. Standardize categorical fields
            cleaned_data = self._standardize_categorical_fields(cleaned_data)
            operations.append("Standardized categorical fields")
            
            # 8. Remove invalid records
            before_validation = len(cleaned_data)
            cleaned_data = self._remove_invalid_records(cleaned_data)
            after_validation = len(cleaned_data)
            if before_validation != after_validation:
                operations.append(f"Removed {before_validation - after_validation} invalid records")
            
            # 9. Add cleaning metadata
            cleaned_data = self._add_cleaning_metadata(cleaned_data)
            operations.append("Added cleaning metadata")
            
            cleaning_time = (datetime.now() - start_time).total_seconds()
            
            result = CleaningResult(
                success=True,
                original_records=original_records,
                cleaned_records=len(cleaned_data),
                removed_records=original_records - len(cleaned_data),
                cleaning_operations=operations,
                cleaning_time=cleaning_time,
                data=cleaned_data,
                errors=[]
            )
            
            logger.info(f"Data cleaning completed: {len(cleaned_data)}/{original_records} records retained ({cleaning_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            return CleaningResult(
                success=False,
                original_records=len(data) if data is not None else 0,
                cleaned_records=0,
                removed_records=0,
                cleaning_operations=[],
                cleaning_time=(datetime.now() - start_time).total_seconds(),
                data=None,
                errors=[str(e)]
            )
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records"""
        # Remove exact duplicates
        data = data.drop_duplicates()
        
        # Remove duplicates based on order_id if it exists
        if 'order_id' in data.columns:
            data = data.drop_duplicates(subset=['order_id'], keep='first')
        
        # Remove duplicates based on key fields
        key_fields = ['customer_name', 'product', 'order_date', 'price']
        existing_fields = [field for field in key_fields if field in data.columns]
        if len(existing_fields) >= 3:
            data = data.drop_duplicates(subset=existing_fields, keep='first')
        
        return data
    
    def _clean_text_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize text fields"""
        text_fields = ['customer_name', 'product', 'store_location', 'notes', 'source']
        
        for field in text_fields:
            if field in data.columns:
                # Basic text cleaning
                data[field] = data[field].astype(str)
                data[field] = data[field].str.strip()  # Remove leading/trailing whitespace
                data[field] = data[field].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
                data[field] = data[field].replace('nan', np.nan)  # Convert 'nan' strings back to NaN
                
                # Field-specific cleaning
                if field == 'customer_name':
                    data[field] = data[field].str.title()  # Title case
                    # Remove special characters except spaces, hyphens, apostrophes
                    data[field] = data[field].str.replace(r"[^a-zA-Z\s\-']", '', regex=True)
                
                elif field == 'product':
                    data[field] = self._standardize_product_names(data[field])
                
                elif field == 'store_location':
                    data[field] = self._standardize_locations(data[field])
                
                elif field == 'source':
                    data[field] = data[field].str.lower()
                    # Standardize source values
                    source_mapping = {
                        'web': 'website',
                        'online': 'website',
                        'app': 'mobile_app',
                        'mobile': 'mobile_app',
                        'retail': 'store',
                        'shop': 'store',
                        'call': 'phone',
                        'telephone': 'phone'
                    }
                    data[field] = data[field].replace(source_mapping)
        
        return data
    
    def _standardize_product_names(self, product_series: pd.Series) -> pd.Series:
        """Standardize product names"""
        # Convert to lowercase for mapping
        product_lower = product_series.str.lower()
        
        # Apply mappings
        for key, value in self.product_mappings.items():
            product_series = product_series.where(product_lower != key, value)
        
        return product_series
    
    def _standardize_locations(self, location_series: pd.Series) -> pd.Series:
        """Standardize location names"""
        # Convert to lowercase for mapping
        location_lower = location_series.str.lower()
        
        # Apply mappings
        for key, value in self.location_mappings.items():
            location_series = location_series.where(location_lower != key, value)
        
        # Title case for locations
        location_series = location_series.str.title()
        
        return location_series
    
    def _fix_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for all fields"""
        # Numeric fields
        numeric_fields = ['quantity', 'price', 'discount', 'total_amount']
        for field in numeric_fields:
            if field in data.columns:
                # Convert to numeric, coercing errors to NaN
                data[field] = pd.to_numeric(data[field], errors='coerce')
        
        # Date fields
        date_fields = ['order_date', 'delivery_date', 'created_at', 'updated_at']
        for field in date_fields:
            if field in data.columns:
                try:
                    data[field] = pd.to_datetime(data[field], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {field} to datetime: {e}")
        
        # Email fields
        email_fields = ['customer_email', 'contact_email']
        for field in email_fields:
            if field in data.columns:
                # Clean email addresses
                data[field] = data[field].astype(str).str.lower().str.strip()
                # Replace 'nan' with actual NaN
                data[field] = data[field].replace('nan', np.nan)
                # Validate email format and set invalid ones to NaN
                mask = data[field].notna()
                valid_emails = data.loc[mask, field].apply(validate_email)
                data.loc[mask & ~valid_emails, field] = np.nan
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        # Critical fields - remove records if missing
        critical_fields = ['order_id', 'customer_name', 'product', 'price']
        for field in critical_fields:
            if field in data.columns:
                before_count = len(data)
                data = data.dropna(subset=[field])
                after_count = len(data)
                if before_count != after_count:
                    logger.info(f"Removed {before_count - after_count} records with missing {field}")
        
        # Fill missing values for non-critical fields
        if 'quantity' in data.columns:
            data['quantity'] = data['quantity'].fillna(1)
        
        if 'discount' in data.columns:
            data['discount'] = data['discount'].fillna(0.0)
        
        if 'source' in data.columns:
            data['source'] = data['source'].fillna('unknown')
        
        if 'product_category' in data.columns:
            data['product_category'] = data['product_category'].fillna('Electronics')
        
        return data
    
    def _clean_numeric_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric fields"""
        # Quantity cleaning
        if 'quantity' in data.columns:
            # Remove negative quantities
            data = data[data['quantity'] > 0]
            # Round to integers
            data['quantity'] = data['quantity'].round().astype(int)
        
        # Price cleaning
        if 'price' in data.columns:
            # Remove negative or zero prices
            data = data[data['price'] > 0]
            # Round to 2 decimal places
            data['price'] = data['price'].round(2)
        
        # Discount cleaning
        if 'discount' in data.columns:
            # Remove negative discounts
            data.loc[data['discount'] < 0, 'discount'] = 0
            # Round to 2 decimal places
            data['discount'] = data['discount'].round(2)
        
        # Total amount cleaning
        if 'total_amount' in data.columns:
            # Remove negative total amounts
            data = data[data['total_amount'] >= 0]
            # Round to 2 decimal places
            data['total_amount'] = data['total_amount'].round(2)
        
        return data
    
    def _clean_date_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean date fields"""
        current_date = datetime.now()
        
        if 'order_date' in data.columns:
            # Remove future dates
            future_mask = data['order_date'] > current_date
            if future_mask.any():
                logger.warning(f"Removing {future_mask.sum()} records with future order dates")
                data = data[~future_mask]
            
            # Remove very old dates (more than 10 years)
            ten_years_ago = current_date - timedelta(days=10*365)
            old_mask = data['order_date'] < ten_years_ago
            if old_mask.any():
                logger.warning(f"Removing {old_mask.sum()} records with very old order dates")
                data = data[~old_mask]
        
        return data
    
    def _standardize_categorical_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical fields"""
        # Source field standardization
        if 'source' in data.columns:
            valid_sources = ['website', 'mobile_app', 'store', 'phone', 'partner', 'unknown']
            invalid_mask = ~data['source'].isin(valid_sources)
            if invalid_mask.any():
                data.loc[invalid_mask, 'source'] = 'unknown'
        
        # Product category standardization
        if 'product_category' in data.columns:
            category_mapping = {
                'electronic': 'Electronics',
                'electronics': 'Electronics',
                'tech': 'Electronics',
                'technology': 'Electronics',
                'gaming': 'Gaming',
                'games': 'Gaming',
                'game': 'Gaming',
                'book': 'Books',
                'books': 'Books',
                'reading': 'Books',
                'smart home': 'Smart Home',
                'smarthome': 'Smart Home',
                'home': 'Smart Home',
                'fitness': 'Fitness',
                'health': 'Fitness',
                'sport': 'Fitness'
            }
            
            data['product_category'] = data['product_category'].str.lower()
            data['product_category'] = data['product_category'].replace(category_mapping)
            data['product_category'] = data['product_category'].str.title()
        
        return data
    
    def _remove_invalid_records(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove records that are clearly invalid"""
        original_count = len(data)
        
        # Remove records with impossible values
        if 'quantity' in data.columns and 'price' in data.columns:
            # Remove records where quantity * price is unreasonably high (> $100,000)
            invalid_total = (data['quantity'] * data['price']) > 100000
            data = data[~invalid_total]
        
        # Remove records with suspicious customer names
        if 'customer_name' in data.columns:
            # Remove very short names (likely invalid)
            short_names = data['customer_name'].str.len() < 2
            data = data[~short_names]
            
            # Remove names that are clearly test data
            test_patterns = ['test', 'dummy', 'sample', 'example', 'null', 'none']
            test_mask = data['customer_name'].str.lower().str.contains('|'.join(test_patterns), na=False)
            data = data[~test_mask]
        
        removed_count = original_count - len(data)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} invalid records")
        
        return data
    
    def _add_cleaning_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add metadata about the cleaning process"""
        data['cleaned_at'] = datetime.now().isoformat()
        data['data_quality_score'] = self._calculate_quality_score(data)
        
        return data
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate data quality score for each record"""
        scores = pd.Series(100.0, index=data.index)  # Start with perfect score
        
        # Deduct points for missing optional fields
        optional_fields = ['customer_email', 'store_location', 'product_category', 'notes']
        for field in optional_fields:
            if field in data.columns:
                missing_mask = data[field].isna()
                scores[missing_mask] -= 5  # Deduct 5 points for each missing optional field
        
        # Deduct points for suspicious patterns
        if 'customer_name' in data.columns:
            # Very short names
            short_names = data['customer_name'].str.len() < 5
            scores[short_names] -= 10
        
        if 'price' in data.columns and 'quantity' in data.columns:
            # Unusually high quantities
            high_quantity = data['quantity'] > 10
            scores[high_quantity] -= 5
            
            # Unusually high prices
            high_price = data['price'] > 5000
            scores[high_price] -= 5
        
        # Ensure scores don't go below 0
        scores = scores.clip(lower=0)
        
        return scores
    
    def generate_cleaning_report(self, result: CleaningResult) -> str:
        """Generate detailed cleaning report"""
        report = []
        report.append("# Data Cleaning Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Original Records**: {result.original_records:,}")
        report.append(f"- **Cleaned Records**: {result.cleaned_records:,}")
        report.append(f"- **Removed Records**: {result.removed_records:,}")
        report.append(f"- **Retention Rate**: {safe_divide(result.cleaned_records, result.original_records, 0) * 100:.1f}%")
        report.append(f"- **Cleaning Time**: {result.cleaning_time:.2f} seconds")
        report.append(f"- **Status**: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        report.append("")
        
        # Operations performed
        if result.cleaning_operations:
            report.append("## Cleaning Operations")
            for i, operation in enumerate(result.cleaning_operations, 1):
                report.append(f"{i}. {operation}")
            report.append("")
        
        # Data quality summary
        if result.data is not None and 'data_quality_score' in result.data.columns:
            avg_quality = result.data['data_quality_score'].mean()
            min_quality = result.data['data_quality_score'].min()
            max_quality = result.data['data_quality_score'].max()
            
            report.append("## Data Quality")
            report.append(f"- **Average Quality Score**: {avg_quality:.1f}/100")
            report.append(f"- **Minimum Quality Score**: {min_quality:.1f}/100")
            report.append(f"- **Maximum Quality Score**: {max_quality:.1f}/100")
            report.append("")
        
        # Errors
        if result.errors:
            report.append("## Errors")
            for i, error in enumerate(result.errors, 1):
                report.append(f"{i}. {error}")
            report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test data cleaner
    import pandas as pd
    
    # Create test data with various issues
    test_data = pd.DataFrame([
        {
            'order_id': 'ORD-2024-001',
            'customer_name': '  john doe  ',
            'product': 'iphone 15',
            'quantity': 1.0,
            'price': 999.99,
            'order_date': '2024-01-15',
            'source': 'web',
            'customer_email': 'JOHN@EXAMPLE.COM',
            'discount': 0,
            'store_location': 'ny'
        },
        {
            'order_id': 'ORD-2024-001',  # Duplicate
            'customer_name': '  john doe  ',
            'product': 'iphone 15',
            'quantity': 1.0,
            'price': 999.99,
            'order_date': '2024-01-15',
            'source': 'web',
            'customer_email': 'JOHN@EXAMPLE.COM',
            'discount': 0,
            'store_location': 'ny'
        },
        {
            'order_id': 'ORD-2024-002',
            'customer_name': '',  # Missing name
            'product': 'macbook pro',
            'quantity': -1,  # Invalid quantity
            'price': -100,  # Invalid price
            'order_date': '2025-12-31',  # Future date
            'source': 'app',
            'customer_email': 'invalid-email',
            'discount': 50.0,
            'store_location': 'los angeles ca'
        },
        {
            'order_id': 'ORD-2024-003',
            'customer_name': 'Jane Smith',
            'product': 'airpods pro',
            'quantity': 2,
            'price': 249.99,
            'order_date': '2024-01-16',
            'source': 'retail',
            'customer_email': 'jane@example.com',
            'discount': np.nan,
            'store_location': 'chicago il'
        }
    ])
    
    # Test cleaner
    cleaner = DataCleaner()
    result = cleaner.clean_orders(test_data)
    
    print("Data Cleaning Test Results:")
    print(f"Success: {result.success}")
    print(f"Original Records: {result.original_records}")
    print(f"Cleaned Records: {result.cleaned_records}")
    print(f"Removed Records: {result.removed_records}")
    print(f"Retention Rate: {safe_divide(result.cleaned_records, result.original_records, 0) * 100:.1f}%")
    print(f"Cleaning Time: {result.cleaning_time:.2f}s")
    
    if result.data is not None:
        print(f"\nCleaned Data Sample:")
        print(result.data[['customer_name', 'product', 'source', 'store_location']].head())
    
    # Generate report
    report = cleaner.generate_cleaning_report(result)
    print(f"\nCleaning report generated ({len(report)} characters)")
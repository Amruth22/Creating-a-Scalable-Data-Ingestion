"""
Data enrichment module for adding calculated fields and business intelligence
Enhances data with additional insights, categorizations, and derived metrics
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..utils.helpers import safe_divide
from ..utils.constants import DataSourceType

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EnrichmentResult:
    """Data enrichment result container"""
    success: bool
    original_fields: int
    enriched_fields: int
    added_fields: int
    enrichment_time: float
    data: Optional[pd.DataFrame] = None
    errors: List[str] = None

class DataEnricher:
    """Data enricher for adding calculated fields and business intelligence"""
    
    def __init__(self):
        """Initialize data enricher"""
        self.product_categories = self._load_product_categories()
        self.customer_segments = self._load_customer_segments()
        self.seasonal_factors = self._load_seasonal_factors()
        
        logger.info("Data enricher initialized")
    
    def _load_product_categories(self) -> Dict[str, Dict[str, Any]]:
        """Load product categorization data"""
        return {
            'iPhone 15': {
                'category': 'Electronics',
                'subcategory': 'Smartphones',
                'brand': 'Apple',
                'price_tier': 'Premium',
                'popularity_score': 95
            },
            'MacBook Pro': {
                'category': 'Electronics',
                'subcategory': 'Laptops',
                'brand': 'Apple',
                'price_tier': 'Premium',
                'popularity_score': 90
            },
            'AirPods Pro': {
                'category': 'Electronics',
                'subcategory': 'Audio',
                'brand': 'Apple',
                'price_tier': 'Premium',
                'popularity_score': 88
            },
            'iPad Air': {
                'category': 'Electronics',
                'subcategory': 'Tablets',
                'brand': 'Apple',
                'price_tier': 'Premium',
                'popularity_score': 85
            },
            'Apple Watch': {
                'category': 'Electronics',
                'subcategory': 'Wearables',
                'brand': 'Apple',
                'price_tier': 'Premium',
                'popularity_score': 82
            },
            'Samsung Galaxy S24': {
                'category': 'Electronics',
                'subcategory': 'Smartphones',
                'brand': 'Samsung',
                'price_tier': 'Premium',
                'popularity_score': 80
            },
            'Nintendo Switch': {
                'category': 'Gaming',
                'subcategory': 'Consoles',
                'brand': 'Nintendo',
                'price_tier': 'Mid-range',
                'popularity_score': 85
            },
            'PlayStation 5': {
                'category': 'Gaming',
                'subcategory': 'Consoles',
                'brand': 'Sony',
                'price_tier': 'Premium',
                'popularity_score': 92
            },
            'Xbox Series X': {
                'category': 'Gaming',
                'subcategory': 'Consoles',
                'brand': 'Microsoft',
                'price_tier': 'Premium',
                'popularity_score': 88
            },
            'Kindle Paperwhite': {
                'category': 'Electronics',
                'subcategory': 'E-readers',
                'brand': 'Amazon',
                'price_tier': 'Budget',
                'popularity_score': 75
            }
        }
    
    def _load_customer_segments(self) -> Dict[str, Dict[str, Any]]:
        """Load customer segmentation criteria"""
        return {
            'VIP': {
                'min_total_spent': 5000,
                'min_order_count': 10,
                'discount_rate': 0.15,
                'priority_level': 1
            },
            'Premium': {
                'min_total_spent': 2000,
                'min_order_count': 5,
                'discount_rate': 0.10,
                'priority_level': 2
            },
            'Standard': {
                'min_total_spent': 500,
                'min_order_count': 2,
                'discount_rate': 0.05,
                'priority_level': 3
            },
            'Budget': {
                'min_total_spent': 0,
                'min_order_count': 0,
                'discount_rate': 0.02,
                'priority_level': 4
            }
        }
    
    def _load_seasonal_factors(self) -> Dict[int, float]:
        """Load seasonal adjustment factors by month"""
        return {
            1: 0.85,   # January - Post-holiday low
            2: 0.90,   # February - Valentine's boost
            3: 0.95,   # March - Spring preparation
            4: 1.00,   # April - Normal
            5: 1.05,   # May - Mother's Day
            6: 1.10,   # June - Summer start
            7: 1.15,   # July - Summer peak
            8: 1.10,   # August - Back to school
            9: 1.05,   # September - Fall start
            10: 1.20,  # October - Halloween/preparation
            11: 1.40,  # November - Black Friday
            12: 1.50   # December - Holiday season
        }
    
    def enrich_orders(self, data: pd.DataFrame) -> EnrichmentResult:
        """
        Enrich order data with calculated fields and business intelligence
        
        Args:
            data (pd.DataFrame): Order data to enrich
            
        Returns:
            EnrichmentResult: Enrichment results with enhanced data
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting data enrichment for {len(data)} records")
            
            original_fields = len(data.columns)
            enriched_data = data.copy()
            
            # 1. Add calculated financial fields
            enriched_data = self._add_financial_calculations(enriched_data)
            
            # 2. Add product intelligence
            enriched_data = self._add_product_intelligence(enriched_data)
            
            # 3. Add customer segmentation
            enriched_data = self._add_customer_segmentation(enriched_data)
            
            # 4. Add temporal features
            enriched_data = self._add_temporal_features(enriched_data)
            
            # 5. Add business metrics
            enriched_data = self._add_business_metrics(enriched_data)
            
            # 6. Add risk assessment
            enriched_data = self._add_risk_assessment(enriched_data)
            
            # 7. Add seasonal analysis
            enriched_data = self._add_seasonal_analysis(enriched_data)
            
            # 8. Add enrichment metadata
            enriched_data = self._add_enrichment_metadata(enriched_data)
            
            enriched_fields = len(enriched_data.columns)
            added_fields = enriched_fields - original_fields
            enrichment_time = (datetime.now() - start_time).total_seconds()
            
            result = EnrichmentResult(
                success=True,
                original_fields=original_fields,
                enriched_fields=enriched_fields,
                added_fields=added_fields,
                enrichment_time=enrichment_time,
                data=enriched_data,
                errors=[]
            )
            
            logger.info(f"Data enrichment completed: {added_fields} fields added ({enrichment_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during data enrichment: {e}")
            return EnrichmentResult(
                success=False,
                original_fields=len(data.columns) if data is not None else 0,
                enriched_fields=0,
                added_fields=0,
                enrichment_time=(datetime.now() - start_time).total_seconds(),
                data=None,
                errors=[str(e)]
            )
    
    def _add_financial_calculations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add financial calculations and metrics"""
        # Calculate total amount if not present
        if 'total_amount' not in data.columns and all(col in data.columns for col in ['quantity', 'price']):
            data['total_amount'] = data['quantity'] * data['price']
            if 'discount' in data.columns:
                data['total_amount'] = data['total_amount'] - data['discount'].fillna(0)
        
        # Calculate unit price after discount
        if all(col in data.columns for col in ['total_amount', 'quantity']):
            data['unit_price_after_discount'] = safe_divide(data['total_amount'], data['quantity'], 0)
        
        # Calculate discount percentage
        if all(col in data.columns for col in ['discount', 'quantity', 'price']):
            original_total = data['quantity'] * data['price']
            data['discount_percentage'] = safe_divide(data['discount'].fillna(0), original_total, 0) * 100
        
        # Calculate profit margin (assuming 30% margin)
        if 'total_amount' in data.columns:
            data['estimated_profit'] = data['total_amount'] * 0.30
        
        return data
    
    def _add_product_intelligence(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add product intelligence and categorization"""
        if 'product' not in data.columns:
            return data
        
        # Initialize new columns
        data['product_brand'] = 'Unknown'
        data['product_subcategory'] = 'Unknown'
        data['price_tier'] = 'Unknown'
        data['popularity_score'] = 50  # Default score
        
        # Apply product mappings
        for product_name, product_info in self.product_categories.items():
            mask = data['product'].str.contains(product_name, case=False, na=False)
            if mask.any():
                data.loc[mask, 'product_brand'] = product_info['brand']
                data.loc[mask, 'product_subcategory'] = product_info['subcategory']
                data.loc[mask, 'price_tier'] = product_info['price_tier']
                data.loc[mask, 'popularity_score'] = product_info['popularity_score']
        
        # Add product category if not present
        if 'product_category' not in data.columns:
            data['product_category'] = 'Electronics'  # Default
            
            # Set specific categories based on product
            gaming_mask = data['product'].str.contains('Nintendo|PlayStation|Xbox', case=False, na=False)
            data.loc[gaming_mask, 'product_category'] = 'Gaming'
            
            book_mask = data['product'].str.contains('Kindle|Book', case=False, na=False)
            data.loc[book_mask, 'product_category'] = 'Books'
        
        return data
    
    def _add_customer_segmentation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add customer segmentation based on spending patterns"""
        if 'customer_name' not in data.columns:
            return data
        
        # Calculate customer metrics
        customer_metrics = data.groupby('customer_name').agg({
            'total_amount': ['sum', 'count', 'mean'],
            'order_date': ['min', 'max'] if 'order_date' in data.columns else ['count']
        }).round(2)
        
        # Flatten column names
        customer_metrics.columns = ['total_spent', 'order_count', 'avg_order_value', 'first_order', 'last_order']
        
        # Determine customer segment
        def determine_segment(row):
            total_spent = row['total_spent']
            order_count = row['order_count']
            
            for segment, criteria in self.customer_segments.items():
                if (total_spent >= criteria['min_total_spent'] and 
                    order_count >= criteria['min_order_count']):
                    return segment
            return 'Budget'
        
        customer_metrics['customer_segment'] = customer_metrics.apply(determine_segment, axis=1)
        
        # Add segment information to main data
        segment_mapping = customer_metrics['customer_segment'].to_dict()
        data['customer_segment'] = data['customer_name'].map(segment_mapping).fillna('Budget')
        
        # Add customer lifetime value estimate
        ltv_mapping = customer_metrics['total_spent'].to_dict()
        data['customer_ltv'] = data['customer_name'].map(ltv_mapping).fillna(0)
        
        # Add customer priority
        priority_mapping = {
            segment: info['priority_level'] 
            for segment, info in self.customer_segments.items()
        }
        data['customer_priority'] = data['customer_segment'].map(priority_mapping).fillna(4)
        
        return data
    
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features and date-based insights"""
        if 'order_date' not in data.columns:
            return data
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data['order_date']):
            data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')
        
        # Extract date components
        data['order_year'] = data['order_date'].dt.year
        data['order_month'] = data['order_date'].dt.month
        data['order_day'] = data['order_date'].dt.day
        data['order_weekday'] = data['order_date'].dt.day_name()
        data['order_week_of_year'] = data['order_date'].dt.isocalendar().week
        data['order_quarter'] = data['order_date'].dt.quarter
        
        # Add business day indicators
        data['is_weekend'] = data['order_date'].dt.weekday >= 5
        data['is_month_end'] = data['order_date'].dt.is_month_end
        data['is_month_start'] = data['order_date'].dt.is_month_start
        
        # Add time since order
        current_date = datetime.now()
        data['days_since_order'] = (current_date - data['order_date']).dt.days
        
        # Categorize order timing
        data['order_timing'] = 'Regular'
        data.loc[data['is_weekend'], 'order_timing'] = 'Weekend'
        data.loc[data['order_month'].isin([11, 12]), 'order_timing'] = 'Holiday Season'
        data.loc[data['order_month'].isin([6, 7, 8]), 'order_timing'] = 'Summer'
        
        return data
    
    def _add_business_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add business-specific metrics and KPIs"""
        # Order size categorization
        if 'total_amount' in data.columns:
            data['order_size_category'] = pd.cut(
                data['total_amount'],
                bins=[0, 100, 500, 1000, 2000, float('inf')],
                labels=['Small', 'Medium', 'Large', 'XLarge', 'Enterprise'],
                include_lowest=True
            )
        
        # Quantity categorization
        if 'quantity' in data.columns:
            data['quantity_category'] = pd.cut(
                data['quantity'],
                bins=[0, 1, 3, 5, 10, float('inf')],
                labels=['Single', 'Few', 'Several', 'Many', 'Bulk'],
                include_lowest=True
            )
        
        # Revenue contribution
        if 'total_amount' in data.columns:
            total_revenue = data['total_amount'].sum()
            data['revenue_contribution_pct'] = (data['total_amount'] / total_revenue * 100).round(2)
        
        # Add source channel insights
        if 'source' in data.columns:
            channel_mapping = {
                'website': 'Digital',
                'mobile_app': 'Digital',
                'store': 'Physical',
                'phone': 'Traditional',
                'partner': 'Channel',
                'unknown': 'Other'
            }
            data['channel_type'] = data['source'].map(channel_mapping).fillna('Other')
        
        return data
    
    def _add_risk_assessment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add risk assessment and fraud indicators"""
        # Initialize risk score
        data['risk_score'] = 50  # Neutral score
        
        # High value order risk
        if 'total_amount' in data.columns:
            high_value_threshold = data['total_amount'].quantile(0.95)
            data.loc[data['total_amount'] > high_value_threshold, 'risk_score'] += 20
        
        # High quantity risk
        if 'quantity' in data.columns:
            high_quantity_threshold = data['quantity'].quantile(0.95)
            data.loc[data['quantity'] > high_quantity_threshold, 'risk_score'] += 15
        
        # New customer risk
        if 'customer_segment' in data.columns:
            data.loc[data['customer_segment'] == 'Budget', 'risk_score'] += 10
        
        # Weekend order risk (slightly higher)
        if 'is_weekend' in data.columns:
            data.loc[data['is_weekend'], 'risk_score'] += 5
        
        # Discount abuse risk
        if 'discount_percentage' in data.columns:
            high_discount_mask = data['discount_percentage'] > 20
            data.loc[high_discount_mask, 'risk_score'] += 15
        
        # Normalize risk score to 0-100 range
        data['risk_score'] = data['risk_score'].clip(0, 100)
        
        # Categorize risk level
        data['risk_level'] = pd.cut(
            data['risk_score'],
            bins=[0, 30, 60, 80, 100],
            labels=['Low', 'Medium', 'High', 'Critical'],
            include_lowest=True
        )
        
        return data
    
    def _add_seasonal_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal analysis and adjustments"""
        if 'order_month' not in data.columns:
            return data
        
        # Add seasonal factor
        data['seasonal_factor'] = data['order_month'].map(self.seasonal_factors).fillna(1.0)
        
        # Calculate seasonally adjusted metrics
        if 'total_amount' in data.columns:
            data['seasonally_adjusted_amount'] = data['total_amount'] / data['seasonal_factor']
        
        # Add season category
        season_mapping = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        data['season'] = data['order_month'].map(season_mapping)
        
        # Add holiday proximity
        data['is_holiday_season'] = data['order_month'].isin([11, 12])
        data['is_summer_season'] = data['order_month'].isin([6, 7, 8])
        
        return data
    
    def _add_enrichment_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add metadata about the enrichment process"""
        data['enriched_at'] = datetime.now().isoformat()
        data['enrichment_version'] = '1.0'
        
        return data
    
    def generate_enrichment_report(self, result: EnrichmentResult) -> str:
        """Generate detailed enrichment report"""
        report = []
        report.append("# Data Enrichment Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Original Fields**: {result.original_fields}")
        report.append(f"- **Enriched Fields**: {result.enriched_fields}")
        report.append(f"- **Added Fields**: {result.added_fields}")
        report.append(f"- **Enrichment Time**: {result.enrichment_time:.2f} seconds")
        report.append(f"- **Status**: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        report.append("")
        
        # Added fields breakdown
        if result.data is not None:
            new_fields = [col for col in result.data.columns if col not in ['order_id', 'customer_name', 'product', 'quantity', 'price', 'order_date']]
            
            report.append("## Added Fields")
            field_categories = {
                'Financial': [f for f in new_fields if any(term in f.lower() for term in ['amount', 'price', 'profit', 'discount'])],
                'Product': [f for f in new_fields if any(term in f.lower() for term in ['product', 'brand', 'category', 'tier'])],
                'Customer': [f for f in new_fields if any(term in f.lower() for term in ['customer', 'segment', 'ltv', 'priority'])],
                'Temporal': [f for f in new_fields if any(term in f.lower() for term in ['date', 'time', 'day', 'month', 'year', 'season'])],
                'Business': [f for f in new_fields if any(term in f.lower() for term in ['size', 'category', 'contribution', 'channel'])],
                'Risk': [f for f in new_fields if any(term in f.lower() for term in ['risk', 'score', 'level'])],
                'Other': []
            }
            
            # Categorize remaining fields
            categorized_fields = set()
            for category_fields in field_categories.values():
                categorized_fields.update(category_fields)
            
            field_categories['Other'] = [f for f in new_fields if f not in categorized_fields]
            
            for category, fields in field_categories.items():
                if fields:
                    report.append(f"### {category} Fields ({len(fields)})")
                    for field in fields:
                        report.append(f"- {field}")
                    report.append("")
        
        # Errors
        if result.errors:
            report.append("## Errors")
            for i, error in enumerate(result.errors, 1):
                report.append(f"{i}. {error}")
            report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test data enricher
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
            'discount': 0.0
        },
        {
            'order_id': 'ORD-2024-002',
            'customer_name': 'Jane Smith',
            'product': 'MacBook Pro',
            'quantity': 1,
            'price': 1999.99,
            'order_date': '2024-11-15',
            'source': 'store',
            'discount': 100.0
        },
        {
            'order_id': 'ORD-2024-003',
            'customer_name': 'John Doe',
            'product': 'AirPods Pro',
            'quantity': 2,
            'price': 249.99,
            'order_date': '2024-07-16',
            'source': 'mobile_app',
            'discount': 0.0
        }
    ])
    
    # Test enricher
    enricher = DataEnricher()
    result = enricher.enrich_orders(test_data)
    
    print("Data Enrichment Test Results:")
    print(f"Success: {result.success}")
    print(f"Original Fields: {result.original_fields}")
    print(f"Enriched Fields: {result.enriched_fields}")
    print(f"Added Fields: {result.added_fields}")
    print(f"Enrichment Time: {result.enrichment_time:.2f}s")
    
    if result.data is not None:
        print(f"\nSample Enriched Data:")
        sample_cols = ['customer_name', 'product', 'customer_segment', 'product_brand', 'risk_level', 'season']
        available_cols = [col for col in sample_cols if col in result.data.columns]
        print(result.data[available_cols].head())
    
    # Generate report
    report = enricher.generate_enrichment_report(result)
    print(f"\nEnrichment report generated ({len(report)} characters)")
"""
Data enrichment module for adding calculated fields and business intelligence
Enhances data with derived fields, categorizations, and business metrics
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
    enrichment_operations: List[str]
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
        """Load product category mappings with metadata"""
        return {
            'iPhone 15': {
                'category': 'Electronics',
                'subcategory': 'Smartphones',
                'brand': 'Apple',
                'avg_price': 999.99,
                'margin_category': 'High',
                'popularity_score': 95
            },
            'MacBook Pro': {
                'category': 'Electronics',
                'subcategory': 'Laptops',
                'brand': 'Apple',
                'avg_price': 1999.99,
                'margin_category': 'High',
                'popularity_score': 90
            },
            'AirPods Pro': {
                'category': 'Electronics',
                'subcategory': 'Audio',
                'brand': 'Apple',
                'avg_price': 249.99,
                'margin_category': 'High',
                'popularity_score': 88
            },
            'iPad Air': {
                'category': 'Electronics',
                'subcategory': 'Tablets',
                'brand': 'Apple',
                'avg_price': 599.99,
                'margin_category': 'High',
                'popularity_score': 85
            },
            'Apple Watch': {
                'category': 'Electronics',
                'subcategory': 'Wearables',
                'brand': 'Apple',
                'avg_price': 399.99,
                'margin_category': 'High',
                'popularity_score': 82
            },
            'Samsung Galaxy S24': {
                'category': 'Electronics',
                'subcategory': 'Smartphones',
                'brand': 'Samsung',
                'avg_price': 899.99,
                'margin_category': 'Medium',
                'popularity_score': 80
            },
            'Nintendo Switch': {
                'category': 'Gaming',
                'subcategory': 'Consoles',
                'brand': 'Nintendo',
                'avg_price': 299.99,
                'margin_category': 'Medium',
                'popularity_score': 85
            },
            'PlayStation 5': {
                'category': 'Gaming',
                'subcategory': 'Consoles',
                'brand': 'Sony',
                'avg_price': 499.99,
                'margin_category': 'Low',
                'popularity_score': 92
            },
            'Xbox Series X': {
                'category': 'Gaming',
                'subcategory': 'Consoles',
                'brand': 'Microsoft',
                'avg_price': 499.99,
                'margin_category': 'Low',
                'popularity_score': 88
            },
            'Kindle Paperwhite': {
                'category': 'Books',
                'subcategory': 'E-Readers',
                'brand': 'Amazon',
                'avg_price': 139.99,
                'margin_category': 'Medium',
                'popularity_score': 75
            }
        }
    
    def _load_customer_segments(self) -> Dict[str, Dict[str, Any]]:
        """Load customer segmentation rules"""
        return {
            'VIP': {
                'min_order_value': 2000,
                'description': 'High-value customers',
                'discount_eligible': True,
                'priority_shipping': True
            },
            'Premium': {
                'min_order_value': 1000,
                'description': 'Premium customers',
                'discount_eligible': True,
                'priority_shipping': False
            },
            'Standard': {
                'min_order_value': 100,
                'description': 'Standard customers',
                'discount_eligible': False,
                'priority_shipping': False
            },
            'Budget': {
                'min_order_value': 0,
                'description': 'Budget-conscious customers',
                'discount_eligible': False,
                'priority_shipping': False
            }
        }
    
    def _load_seasonal_factors(self) -> Dict[int, Dict[str, float]]:
        """Load seasonal adjustment factors by month"""
        return {
            1: {'Electronics': 0.8, 'Gaming': 0.7, 'Books': 1.0},  # January - post-holiday
            2: {'Electronics': 0.9, 'Gaming': 0.8, 'Books': 1.0},  # February
            3: {'Electronics': 1.0, 'Gaming': 0.9, 'Books': 1.0},  # March
            4: {'Electronics': 1.0, 'Gaming': 1.0, 'Books': 1.0},  # April
            5: {'Electronics': 1.1, 'Gaming': 1.0, 'Books': 1.0},  # May
            6: {'Electronics': 1.0, 'Gaming': 1.0, 'Books': 0.9},  # June
            7: {'Electronics': 1.0, 'Gaming': 1.1, 'Books': 0.9},  # July
            8: {'Electronics': 1.1, 'Gaming': 1.0, 'Books': 1.1},  # August - back to school
            9: {'Electronics': 1.2, 'Gaming': 1.0, 'Books': 1.2},  # September - back to school
            10: {'Electronics': 1.1, 'Gaming': 1.1, 'Books': 1.0}, # October
            11: {'Electronics': 1.3, 'Gaming': 1.4, 'Books': 1.1}, # November - Black Friday
            12: {'Electronics': 1.5, 'Gaming': 1.6, 'Books': 1.3}  # December - Holiday season
        }
    
    def enrich_orders(self, data: pd.DataFrame) -> EnrichmentResult:
        """
        Enrich order data with calculated fields and business intelligence
        
        Args:
            data (pd.DataFrame): Cleaned order data
            
        Returns:
            EnrichmentResult: Enrichment results with enhanced data
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting data enrichment for {len(data)} records")
            
            original_fields = len(data.columns)
            enriched_data = data.copy()
            operations = []
            
            # 1. Calculate basic financial metrics
            enriched_data = self._calculate_financial_metrics(enriched_data)
            operations.append("Calculated financial metrics")
            
            # 2. Add product intelligence
            enriched_data = self._add_product_intelligence(enriched_data)
            operations.append("Added product intelligence")
            
            # 3. Add customer segmentation
            enriched_data = self._add_customer_segmentation(enriched_data)
            operations.append("Added customer segmentation")
            
            # 4. Add temporal features
            enriched_data = self._add_temporal_features(enriched_data)
            operations.append("Added temporal features")
            
            # 5. Add business intelligence metrics
            enriched_data = self._add_business_metrics(enriched_data)
            operations.append("Added business intelligence metrics")
            
            # 6. Add order categorization
            enriched_data = self._add_order_categorization(enriched_data)
            operations.append("Added order categorization")
            
            # 7. Add risk assessment
            enriched_data = self._add_risk_assessment(enriched_data)
            operations.append("Added risk assessment")
            
            # 8. Add seasonal adjustments
            enriched_data = self._add_seasonal_adjustments(enriched_data)
            operations.append("Added seasonal adjustments")
            
            # 9. Add enrichment metadata
            enriched_data = self._add_enrichment_metadata(enriched_data)
            operations.append("Added enrichment metadata")
            
            enrichment_time = (datetime.now() - start_time).total_seconds()
            enriched_fields = len(enriched_data.columns)
            
            result = EnrichmentResult(
                success=True,
                original_fields=original_fields,
                enriched_fields=enriched_fields,
                added_fields=enriched_fields - original_fields,
                enrichment_operations=operations,
                enrichment_time=enrichment_time,
                data=enriched_data,
                errors=[]
            )
            
            logger.info(f"Data enrichment completed: added {enriched_fields - original_fields} fields ({enrichment_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during data enrichment: {e}")
            return EnrichmentResult(
                success=False,
                original_fields=len(data.columns) if data is not None else 0,
                enriched_fields=0,
                added_fields=0,
                enrichment_operations=[],
                enrichment_time=(datetime.now() - start_time).total_seconds(),
                data=None,
                errors=[str(e)]
            )
    
    def _calculate_financial_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate financial metrics"""
        # Calculate total amount if not present
        if 'total_amount' not in data.columns and all(col in data.columns for col in ['price', 'quantity']):
            discount = data.get('discount', 0)
            data['total_amount'] = (data['price'] * data['quantity']) - discount
        
        # Calculate unit discount percentage
        if 'discount' in data.columns and 'price' in data.columns:
            data['discount_percentage'] = safe_divide(data['discount'], data['price'], 0) * 100
        
        # Calculate revenue per unit
        if 'total_amount' in data.columns and 'quantity' in data.columns:
            data['revenue_per_unit'] = safe_divide(data['total_amount'], data['quantity'], 0)
        
        # Calculate profit margin estimate (assuming 30% average margin)
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
        data['margin_category'] = 'Unknown'
        data['popularity_score'] = 50
        data['price_vs_avg'] = 'Unknown'
        
        # Apply product mappings
        for product_name, product_info in self.product_categories.items():
            mask = data['product'] == product_name
            if mask.any():
                data.loc[mask, 'product_brand'] = product_info['brand']
                data.loc[mask, 'product_subcategory'] = product_info['subcategory']
                data.loc[mask, 'margin_category'] = product_info['margin_category']
                data.loc[mask, 'popularity_score'] = product_info['popularity_score']
                
                # Compare price to average
                if 'price' in data.columns:
                    avg_price = product_info['avg_price']
                    price_ratio = data.loc[mask, 'price'] / avg_price
                    data.loc[mask & (price_ratio > 1.1), 'price_vs_avg'] = 'Above Average'
                    data.loc[mask & (price_ratio < 0.9), 'price_vs_avg'] = 'Below Average'
                    data.loc[mask & (price_ratio >= 0.9) & (price_ratio <= 1.1), 'price_vs_avg'] = 'Average'
        
        return data
    
    def _add_customer_segmentation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add customer segmentation based on order value"""
        if 'total_amount' not in data.columns:
            return data
        
        # Initialize customer segment
        data['customer_segment'] = 'Budget'
        data['segment_description'] = 'Budget-conscious customers'
        data['discount_eligible'] = False
        data['priority_shipping'] = False
        
        # Apply segmentation rules
        for segment, rules in self.customer_segments.items():
            mask = data['total_amount'] >= rules['min_order_value']
            if mask.any():
                data.loc[mask, 'customer_segment'] = segment
                data.loc[mask, 'segment_description'] = rules['description']
                data.loc[mask, 'discount_eligible'] = rules['discount_eligible']
                data.loc[mask, 'priority_shipping'] = rules['priority_shipping']
        
        return data
    
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features from order date"""
        if 'order_date' not in data.columns:
            return data
        
        # Ensure order_date is datetime
        data['order_date'] = pd.to_datetime(data['order_date'])
        
        # Extract temporal components
        data['order_year'] = data['order_date'].dt.year
        data['order_month'] = data['order_date'].dt.month
        data['order_day'] = data['order_date'].dt.day
        data['order_weekday'] = data['order_date'].dt.day_name()
        data['order_quarter'] = data['order_date'].dt.quarter
        data['order_week_of_year'] = data['order_date'].dt.isocalendar().week
        
        # Add business day indicators
        data['is_weekend'] = data['order_date'].dt.weekday >= 5
        data['is_month_end'] = data['order_date'].dt.is_month_end
        data['is_month_start'] = data['order_date'].dt.is_month_start
        
        # Add season
        data['season'] = data['order_month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Add holiday proximity (simplified)
        data['is_holiday_season'] = data['order_month'].isin([11, 12])
        data['is_back_to_school'] = data['order_month'].isin([8, 9])
        
        # Calculate days since order
        current_date = datetime.now()
        data['days_since_order'] = (current_date - data['order_date']).dt.days
        
        return data
    
    def _add_business_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add business intelligence metrics"""
        # Order size categorization
        if 'total_amount' in data.columns:
            data['order_size_category'] = pd.cut(
                data['total_amount'],
                bins=[0, 50, 200, 500, 1000, float('inf')],
                labels=['Micro', 'Small', 'Medium', 'Large', 'XLarge'],
                include_lowest=True
            )
        
        # Quantity categorization
        if 'quantity' in data.columns:
            data['quantity_category'] = pd.cut(
                data['quantity'],
                bins=[0, 1, 2, 5, 10, float('inf')],
                labels=['Single', 'Pair', 'Few', 'Multiple', 'Bulk'],
                include_lowest=True
            )
        
        # Revenue impact
        if 'total_amount' in data.columns:
            total_revenue = data['total_amount'].sum()
            data['revenue_contribution'] = safe_divide(data['total_amount'], total_revenue, 0) * 100
            
            # Cumulative revenue contribution
            data_sorted = data.sort_values('total_amount', ascending=False)
            data_sorted['cumulative_revenue_pct'] = data_sorted['total_amount'].cumsum() / total_revenue * 100
            data['cumulative_revenue_pct'] = data_sorted['cumulative_revenue_pct']
        
        return data
    
    def _add_order_categorization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add order categorization and flags"""
        # High-value order flag
        if 'total_amount' in data.columns:
            high_value_threshold = data['total_amount'].quantile(0.9)
            data['is_high_value'] = data['total_amount'] >= high_value_threshold
        
        # Bulk order flag
        if 'quantity' in data.columns:
            bulk_threshold = data['quantity'].quantile(0.8)
            data['is_bulk_order'] = data['quantity'] >= bulk_threshold
        
        # Discounted order flag
        if 'discount' in data.columns:
            data['is_discounted'] = data['discount'] > 0
        
        # New customer flag (simplified - based on email domain)
        if 'customer_email' in data.columns:
            # This is a simplified approach - in reality, you'd check against customer database
            common_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
            data['likely_new_customer'] = ~data['customer_email'].str.split('@').str[1].isin(common_domains)
        
        # Rush order flag (orders placed on weekends might be rush orders)
        if 'is_weekend' in data.columns:
            data['potential_rush_order'] = data['is_weekend']
        
        return data
    
    def _add_risk_assessment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add risk assessment metrics"""
        data['risk_score'] = 0
        data['risk_factors'] = ''
        
        risk_factors = []
        
        # High quantity risk
        if 'quantity' in data.columns:
            high_qty_mask = data['quantity'] > 10
            data.loc[high_qty_mask, 'risk_score'] += 10
            risk_factors.append('High Quantity')
        
        # High value risk
        if 'total_amount' in data.columns:
            high_value_threshold = data['total_amount'].quantile(0.95)
            high_value_mask = data['total_amount'] >= high_value_threshold
            data.loc[high_value_mask, 'risk_score'] += 15
            risk_factors.append('High Value')
        
        # Weekend order risk
        if 'is_weekend' in data.columns:
            weekend_mask = data['is_weekend']
            data.loc[weekend_mask, 'risk_score'] += 5
            risk_factors.append('Weekend Order')
        
        # New customer risk
        if 'likely_new_customer' in data.columns:
            new_customer_mask = data['likely_new_customer']
            data.loc[new_customer_mask, 'risk_score'] += 8
            risk_factors.append('New Customer')
        
        # Risk level categorization
        data['risk_level'] = pd.cut(
            data['risk_score'],
            bins=[0, 5, 15, 25, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical'],
            include_lowest=True
        )
        
        return data
    
    def _add_seasonal_adjustments(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal adjustment factors"""
        if 'order_month' not in data.columns or 'product_category' not in data.columns:
            return data
        
        data['seasonal_factor'] = 1.0
        data['seasonally_adjusted_amount'] = data.get('total_amount', 0)
        
        for month, factors in self.seasonal_factors.items():
            month_mask = data['order_month'] == month
            
            for category, factor in factors.items():
                category_mask = data['product_category'] == category
                combined_mask = month_mask & category_mask
                
                if combined_mask.any():
                    data.loc[combined_mask, 'seasonal_factor'] = factor
                    if 'total_amount' in data.columns:
                        data.loc[combined_mask, 'seasonally_adjusted_amount'] = (
                            data.loc[combined_mask, 'total_amount'] / factor
                        )
        
        return data
    
    def _add_enrichment_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add metadata about the enrichment process"""
        data['enriched_at'] = datetime.now().isoformat()
        data['enrichment_version'] = '1.0'
        
        # Calculate enrichment completeness score
        enrichment_fields = [
            'product_brand', 'customer_segment', 'order_size_category',
            'seasonal_factor', 'risk_level'
        ]
        
        present_fields = [field for field in enrichment_fields if field in data.columns]
        data['enrichment_completeness'] = len(present_fields) / len(enrichment_fields) * 100
        
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
        
        # Operations performed
        if result.enrichment_operations:
            report.append("## Enrichment Operations")
            for i, operation in enumerate(result.enrichment_operations, 1):
                report.append(f"{i}. {operation}")
            report.append("")
        
        # Field analysis
        if result.data is not None:
            report.append("## Added Fields Analysis")
            
            # Customer segments
            if 'customer_segment' in result.data.columns:
                segment_counts = result.data['customer_segment'].value_counts()
                report.append("### Customer Segments")
                for segment, count in segment_counts.items():
                    pct = (count / len(result.data)) * 100
                    report.append(f"- **{segment}**: {count:,} ({pct:.1f}%)")
                report.append("")
            
            # Order categories
            if 'order_size_category' in result.data.columns:
                size_counts = result.data['order_size_category'].value_counts()
                report.append("### Order Size Distribution")
                for size, count in size_counts.items():
                    pct = (count / len(result.data)) * 100
                    report.append(f"- **{size}**: {count:,} ({pct:.1f}%)")
                report.append("")
            
            # Risk levels
            if 'risk_level' in result.data.columns:
                risk_counts = result.data['risk_level'].value_counts()
                report.append("### Risk Level Distribution")
                for risk, count in risk_counts.items():
                    pct = (count / len(result.data)) * 100
                    report.append(f"- **{risk}**: {count:,} ({pct:.1f}%)")
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
            'customer_email': 'john@example.com',
            'discount': 0.0,
            'total_amount': 999.99
        },
        {
            'order_id': 'ORD-2024-002',
            'customer_name': 'Jane Smith',
            'product': 'MacBook Pro',
            'quantity': 1,
            'price': 1999.99,
            'order_date': '2024-12-15',
            'source': 'store',
            'customer_email': 'jane@company.com',
            'discount': 100.0,
            'total_amount': 1899.99
        },
        {
            'order_id': 'ORD-2024-003',
            'customer_name': 'Bob Wilson',
            'product': 'Nintendo Switch',
            'quantity': 5,
            'price': 299.99,
            'order_date': '2024-11-25',
            'source': 'mobile_app',
            'customer_email': 'bob@gmail.com',
            'discount': 0.0,
            'total_amount': 1499.95
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
        print(f"\nSample Enriched Fields:")
        enriched_fields = ['customer_segment', 'product_brand', 'order_size_category', 'risk_level', 'season']
        available_fields = [field for field in enriched_fields if field in result.data.columns]
        if available_fields:
            print(result.data[available_fields].head())
    
    # Generate report
    report = enricher.generate_enrichment_report(result)
    print(f"\nEnrichment report generated ({len(report)} characters)")
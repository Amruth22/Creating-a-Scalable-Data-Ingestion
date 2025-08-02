"""
API ingestion module for collecting data from REST APIs
Handles HTTP requests, authentication, rate limiting, and error handling
"""

import requests
import pandas as pd
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from ..utils.config import config
from ..utils.helpers import retry_operation
from ..utils.constants import DataSourceType, HTTPStatus

# Configure logging
logger = logging.getLogger(__name__)

class APIIngestion:
    """API ingestion class for collecting data from REST APIs"""
    
    def __init__(self):
        """Initialize API ingestion"""
        self.base_url = config.api.base_url
        self.timeout = config.api.timeout
        self.retry_attempts = config.api.retry_attempts
        self.retry_delay = config.api.retry_delay
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DataIngestionPipeline/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        logger.info(f"API ingestion initialized with base URL: {self.base_url}")
    
    def _wait_for_rate_limit(self):
        """Implement simple rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def make_api_request(self, endpoint: str, method: str = 'GET', 
                        params: Optional[Dict] = None, 
                        data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request to API endpoint
        
        Args:
            endpoint (str): API endpoint
            method (str): HTTP method
            params (Dict, optional): Query parameters
            data (Dict, optional): Request body data
            
        Returns:
            Dict[str, Any]: API response data and metadata
        """
        result = {
            'success': False,
            'data': None,
            'status_code': None,
            'error_message': None,
            'response_time': 0,
            'url': None
        }
        
        try:
            # Construct full URL
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            result['url'] = url
            
            # Apply rate limiting
            self._wait_for_rate_limit()
            
            # Make request with retry logic
            def _make_request():
                start_time = time.time()
                
                if method.upper() == 'GET':
                    response = self.session.get(url, params=params, timeout=self.timeout)
                elif method.upper() == 'POST':
                    response = self.session.post(url, params=params, json=data, timeout=self.timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response_time = time.time() - start_time
                return response, response_time
            
            response, response_time = retry_operation(
                _make_request, 
                max_attempts=self.retry_attempts, 
                delay=self.retry_delay
            )
            
            result['status_code'] = response.status_code
            result['response_time'] = response_time
            
            # Check if request was successful
            if response.status_code == HTTPStatus.OK:
                try:
                    json_data = response.json()
                    result['success'] = True
                    result['data'] = json_data
                    logger.debug(f"Successful API request: {url} ({response_time:.2f}s)")
                except json.JSONDecodeError as e:
                    result['error_message'] = f"Invalid JSON response: {str(e)}"
                    logger.error(f"JSON decode error for {url}: {e}")
            else:
                result['error_message'] = f"HTTP {response.status_code}: {response.reason}"
                logger.error(f"API request failed: {url} - {result['error_message']}")
            
        except requests.exceptions.Timeout:
            result['error_message'] = f"Request timeout after {self.timeout} seconds"
            logger.error(f"API request timeout: {url}")
        except requests.exceptions.ConnectionError as e:
            result['error_message'] = f"Connection error: {str(e)}"
            logger.error(f"API connection error: {url} - {e}")
        except Exception as e:
            result['error_message'] = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected API error: {url} - {e}")
        
        return result
    
    def fetch_orders_from_api(self, limit: int = 100) -> Dict[str, Any]:
        """
        Fetch orders from API (using JSONPlaceholder posts as demo data)
        
        Args:
            limit (int): Maximum number of records to fetch
            
        Returns:
            Dict[str, Any]: Processing results
        """
        result = {
            'success': False,
            'records_count': 0,
            'data': None,
            'error_message': None,
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Fetching orders from API (limit: {limit})")
            
            # Make API request
            api_response = self.make_api_request('posts', params={'_limit': limit})
            
            if not api_response['success']:
                result['error_message'] = api_response['error_message']
                return result
            
            posts_data = api_response['data']
            
            if not posts_data:
                result['error_message'] = "No data received from API"
                return result
            
            # Transform API data to order-like structure
            orders = []
            for i, post in enumerate(posts_data):
                # Create realistic order data from post data
                order = {
                    'order_id': f"API-{post.get('id', i+1):04d}",
                    'customer_name': f"Customer {post.get('userId', 1)}",
                    'customer_email': f"customer{post.get('userId', 1)}@example.com",
                    'product': self._generate_product_from_title(post.get('title', 'Unknown Product')),
                    'product_category': 'Electronics',
                    'quantity': 1,
                    'price': round(50 + (post.get('id', 1) % 20) * 25.99, 2),
                    'discount': 0.0,
                    'order_date': datetime.now().strftime('%Y-%m-%d'),
                    'source': 'api',
                    'api_post_id': post.get('id'),
                    'source_type': DataSourceType.API_REST.value,
                    'ingested_at': datetime.now().isoformat()
                }
                
                # Calculate total amount
                order['total_amount'] = round(order['price'] * order['quantity'] - order['discount'], 2)
                
                orders.append(order)
            
            # Convert to DataFrame
            data = pd.DataFrame(orders)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result.update({
                'success': True,
                'records_count': len(data),
                'data': data,
                'processing_time': processing_time,
                'api_response_time': api_response['response_time']
            })
            
            logger.info(f"Successfully fetched {len(data)} orders from API in {processing_time:.2f}s")
            
        except Exception as e:
            result['error_message'] = f"Error fetching orders from API: {str(e)}"
            logger.error(f"Error in fetch_orders_from_api: {e}")
        
        return result
    
    def _generate_product_from_title(self, title: str) -> str:
        """
        Generate realistic product name from post title
        
        Args:
            title (str): Post title
            
        Returns:
            str: Product name
        """
        # Simple mapping based on keywords in title
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['phone', 'mobile', 'call']):
            return 'iPhone 15'
        elif any(word in title_lower for word in ['computer', 'laptop', 'tech']):
            return 'MacBook Pro'
        elif any(word in title_lower for word in ['music', 'audio', 'sound']):
            return 'AirPods Pro'
        elif any(word in title_lower for word in ['watch', 'time']):
            return 'Apple Watch'
        elif any(word in title_lower for word in ['tablet', 'pad']):
            return 'iPad Air'
        elif any(word in title_lower for word in ['game', 'play']):
            return 'Nintendo Switch'
        elif any(word in title_lower for word in ['book', 'read']):
            return 'Kindle Paperwhite'
        else:
            # Default products
            products = ['iPhone 15', 'MacBook Pro', 'AirPods Pro', 'iPad Air', 'Apple Watch']
            return products[hash(title) % len(products)]
    
    def fetch_users_from_api(self) -> Dict[str, Any]:
        """
        Fetch user data from API for customer information
        
        Returns:
            Dict[str, Any]: Processing results
        """
        result = {
            'success': False,
            'records_count': 0,
            'data': None,
            'error_message': None,
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            logger.info("Fetching users from API")
            
            # Make API request
            api_response = self.make_api_request('users')
            
            if not api_response['success']:
                result['error_message'] = api_response['error_message']
                return result
            
            users_data = api_response['data']
            
            if not users_data:
                result['error_message'] = "No user data received from API"
                return result
            
            # Transform API data to customer structure
            customers = []
            for user in users_data:
                customer = {
                    'customer_id': f"CUST-{user.get('id', 0):03d}",
                    'name': user.get('name', 'Unknown Customer'),
                    'email': user.get('email', 'unknown@example.com'),
                    'phone': user.get('phone', ''),
                    'address': self._format_address(user.get('address', {})),
                    'company': user.get('company', {}).get('name', ''),
                    'website': user.get('website', ''),
                    'source_type': DataSourceType.API_REST.value,
                    'ingested_at': datetime.now().isoformat()
                }
                customers.append(customer)
            
            # Convert to DataFrame
            data = pd.DataFrame(customers)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result.update({
                'success': True,
                'records_count': len(data),
                'data': data,
                'processing_time': processing_time,
                'api_response_time': api_response['response_time']
            })
            
            logger.info(f"Successfully fetched {len(data)} customers from API in {processing_time:.2f}s")
            
        except Exception as e:
            result['error_message'] = f"Error fetching users from API: {str(e)}"
            logger.error(f"Error in fetch_users_from_api: {e}")
        
        return result
    
    def _format_address(self, address_data: Dict) -> str:
        """
        Format address from API data
        
        Args:
            address_data (Dict): Address data from API
            
        Returns:
            str: Formatted address
        """
        if not address_data:
            return ""
        
        parts = []
        
        if address_data.get('street'):
            parts.append(address_data['street'])
        if address_data.get('suite'):
            parts.append(address_data['suite'])
        if address_data.get('city'):
            parts.append(address_data['city'])
        if address_data.get('zipcode'):
            parts.append(address_data['zipcode'])
        
        return ", ".join(parts)
    
    def test_api_connection(self) -> Dict[str, Any]:
        """
        Test API connection and availability
        
        Returns:
            Dict[str, Any]: Connection test results
        """
        result = {
            'success': False,
            'base_url': self.base_url,
            'response_time': 0,
            'status_code': None,
            'error_message': None
        }
        
        try:
            logger.info(f"Testing API connection to {self.base_url}")
            
            # Test with a simple endpoint
            api_response = self.make_api_request('posts/1')
            
            result.update({
                'success': api_response['success'],
                'response_time': api_response['response_time'],
                'status_code': api_response['status_code'],
                'error_message': api_response['error_message']
            })
            
            if result['success']:
                logger.info(f"API connection test successful ({result['response_time']:.2f}s)")
            else:
                logger.error(f"API connection test failed: {result['error_message']}")
            
        except Exception as e:
            result['error_message'] = f"Connection test error: {str(e)}"
            logger.error(f"API connection test error: {e}")
        
        return result
    
    def get_api_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive API health status
        
        Returns:
            Dict[str, Any]: Health status information
        """
        health_status = {
            'overall_status': 'unknown',
            'base_url': self.base_url,
            'timestamp': datetime.now().isoformat(),
            'endpoints': {},
            'performance': {},
            'issues': []
        }
        
        try:
            # Test multiple endpoints
            endpoints_to_test = ['posts/1', 'users/1', 'posts']
            
            total_response_time = 0
            successful_tests = 0
            
            for endpoint in endpoints_to_test:
                test_result = self.make_api_request(endpoint)
                
                health_status['endpoints'][endpoint] = {
                    'status': 'healthy' if test_result['success'] else 'unhealthy',
                    'response_time': test_result['response_time'],
                    'status_code': test_result['status_code'],
                    'error': test_result['error_message']
                }
                
                if test_result['success']:
                    successful_tests += 1
                    total_response_time += test_result['response_time']
                else:
                    health_status['issues'].append(f"{endpoint}: {test_result['error_message']}")
            
            # Calculate performance metrics
            if successful_tests > 0:
                health_status['performance'] = {
                    'average_response_time': total_response_time / successful_tests,
                    'success_rate': successful_tests / len(endpoints_to_test),
                    'successful_endpoints': successful_tests,
                    'total_endpoints': len(endpoints_to_test)
                }
            
            # Determine overall status
            success_rate = successful_tests / len(endpoints_to_test)
            if success_rate >= 0.8:
                health_status['overall_status'] = 'healthy'
            elif success_rate >= 0.5:
                health_status['overall_status'] = 'degraded'
            else:
                health_status['overall_status'] = 'unhealthy'
            
            logger.info(f"API health check completed: {health_status['overall_status']} ({success_rate:.1%} success rate)")
            
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['issues'].append(f"Health check error: {str(e)}")
            logger.error(f"API health check error: {e}")
        
        return health_status
    
    def close(self):
        """Close the session and cleanup resources"""
        try:
            self.session.close()
            logger.info("API ingestion session closed")
        except Exception as e:
            logger.error(f"Error closing API session: {e}")

if __name__ == "__main__":
    # Test API ingestion
    api_ingestion = APIIngestion()
    
    try:
        # Test connection
        print("Testing API connection...")
        connection_test = api_ingestion.test_api_connection()
        print(f"Connection test: {'✅ Success' if connection_test['success'] else '❌ Failed'}")
        if not connection_test['success']:
            print(f"Error: {connection_test['error_message']}")
        
        # Get health status
        print("\nChecking API health...")
        health_status = api_ingestion.get_api_health_status()
        print(f"Overall status: {health_status['overall_status']}")
        print(f"Success rate: {health_status['performance'].get('success_rate', 0):.1%}")
        
        # Fetch orders
        print("\nFetching orders from API...")
        orders_result = api_ingestion.fetch_orders_from_api(limit=10)
        if orders_result['success']:
            print(f"✅ Fetched {orders_result['records_count']} orders")
            print(f"Processing time: {orders_result['processing_time']:.2f}s")
            print(f"Sample order: {orders_result['data'].iloc[0].to_dict()}")
        else:
            print(f"❌ Failed to fetch orders: {orders_result['error_message']}")
        
        # Fetch users
        print("\nFetching users from API...")
        users_result = api_ingestion.fetch_users_from_api()
        if users_result['success']:
            print(f"✅ Fetched {users_result['records_count']} users")
            print(f"Processing time: {users_result['processing_time']:.2f}s")
        else:
            print(f"❌ Failed to fetch users: {users_result['error_message']}")
    
    finally:
        # Cleanup
        api_ingestion.close()
import os
import logging
from deriv_api import DerivAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_deriv_connection():
    """Test the Deriv API connection."""
    
    # Set API token
    api_token = 'V4FOmNuhORnxbHT'
    os.environ['DERIV_API_TOKEN'] = api_token
    
    logger.info("Testing Deriv API connection...")
    
    # Initialize API client
    deriv_api = DerivAPI(api_token)
    
    # Test connection
    connected = deriv_api.connect()
    
    if connected:
        logger.info("✓ Successfully connected to Deriv API")
        
        # Test symbol availability
        symbols = ['crash300', 'boom300', 'crash1000', 'boom1000']
        
        for symbol in symbols:
            available = deriv_api.is_symbol_available(symbol)
            logger.info(f"Symbol {symbol}: {'Available' if available else 'Not Available'}")
        
        # Test data fetch
        logger.info("Testing data fetch for crash300...")
        data = deriv_api.get_historical_data('crash300', '1m', 50)
        
        if data is not None and not data.empty:
            logger.info(f"✓ Successfully fetched {len(data)} data points")
            logger.info(f"Data columns: {list(data.columns)}")
            logger.info(f"Latest price: {data['close'].iloc[-1]}")
        else:
            logger.error("✗ Failed to fetch historical data")
        
        # Disconnect
        deriv_api.disconnect()
        logger.info("Disconnected from Deriv API")
        
    else:
        logger.error("✗ Failed to connect to Deriv API")
        return False
    
    return True

if __name__ == "__main__":
    test_deriv_connection()
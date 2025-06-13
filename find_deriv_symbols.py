import os
import json
import websocket
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DerivSymbolFinder:
    def __init__(self, api_token):
        self.api_token = api_token
        self.ws_url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
        self.ws = None
        self.is_connected = False
        self.symbols_found = []
        
    def connect(self):
        """Connect to Deriv WebSocket API."""
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Start WebSocket in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        # Wait for connection
        timeout = 10
        while not self.is_connected and timeout > 0:
            time.sleep(0.5)
            timeout -= 0.5
        
        return self.is_connected
    
    def _on_open(self, ws):
        """WebSocket connection opened."""
        logger.info("Connected to Deriv API")
        self.is_connected = True
        
        # Authorize
        auth_request = {
            "authorize": self.api_token,
            "req_id": 1
        }
        self._send_request(auth_request)
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            if 'authorize' in data:
                if 'error' not in data:
                    logger.info("Successfully authorized")
                    # Request active symbols
                    self._request_active_symbols()
                else:
                    logger.error(f"Authorization failed: {data['error']['message']}")
            
            elif 'active_symbols' in data:
                self._handle_active_symbols(data)
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        logger.info("Connection closed")
        self.is_connected = False
    
    def _send_request(self, request):
        """Send request to WebSocket."""
        if self.ws and self.is_connected:
            self.ws.send(json.dumps(request))
    
    def _request_active_symbols(self):
        """Request all active symbols."""
        request = {
            "active_symbols": "brief",
            "product_type": "basic",
            "req_id": 2
        }
        self._send_request(request)
    
    def _handle_active_symbols(self, data):
        """Handle active symbols response."""
        symbols = data['active_symbols']
        
        # Filter for synthetic indices
        synthetic_symbols = []
        
        for symbol in symbols:
            symbol_name = symbol.get('symbol', '')
            display_name = symbol.get('display_name', '')
            
            # Look for crash, boom, volatility indices
            if any(keyword in display_name.lower() for keyword in ['crash', 'boom', 'volatility']):
                synthetic_symbols.append({
                    'symbol': symbol_name,
                    'display_name': display_name,
                    'market': symbol.get('market', ''),
                    'submarket': symbol.get('submarket', '')
                })
        
        self.symbols_found = synthetic_symbols
        
        logger.info(f"Found {len(synthetic_symbols)} synthetic indices:")
        for sym in synthetic_symbols:
            logger.info(f"  {sym['symbol']} - {sym['display_name']}")
    
    def get_symbols(self):
        """Get the found symbols."""
        return self.symbols_found
    
    def disconnect(self):
        """Disconnect from WebSocket."""
        if self.ws:
            self.ws.close()
        self.is_connected = False

def find_deriv_symbols():
    """Find all available Deriv synthetic indices symbols."""
    api_token = 'V4FOmNuhORnxbHT'
    
    finder = DerivSymbolFinder(api_token)
    
    if finder.connect():
        # Wait for symbols to be retrieved
        time.sleep(5)
        
        symbols = finder.get_symbols()
        
        if symbols:
            print("\n=== DERIV SYNTHETIC INDICES SYMBOLS ===")
            for sym in symbols:
                print(f"Symbol: {sym['symbol']}")
                print(f"Name: {sym['display_name']}")
                print(f"Market: {sym['market']}")
                print(f"Submarket: {sym['submarket']}")
                print("-" * 40)
        else:
            print("No synthetic indices found")
        
        finder.disconnect()
        return symbols
    else:
        print("Failed to connect to Deriv API")
        return []

if __name__ == "__main__":
    find_deriv_symbols()
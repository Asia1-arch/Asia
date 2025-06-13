import os
import json
import websocket
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymbolDebugger:
    def __init__(self, api_token):
        self.api_token = api_token
        self.ws_url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
        self.ws = None
        self.is_connected = False
        self.received_data = {}
        
        # All possible symbol mappings to test
        self.symbol_variations = {
            'boom300': ['BOOM300N', 'BOOM300', 'boom300', 'Boom300', 'BOOM_300'],
            'crash1000': ['CRASH1000', 'CRASH1000N', 'crash1000', 'Crash1000', 'CRASH_1000'],
            'boom1000': ['BOOM1000', 'BOOM1000N', 'boom1000', 'Boom1000', 'BOOM_1000']
        }
        
    def test_symbol_variations(self, base_symbol):
        """Test all variations of a symbol to find the working one."""
        variations = self.symbol_variations.get(base_symbol, [base_symbol])
        working_symbols = []
        
        for variant in variations:
            logger.info(f"Testing symbol variation: {variant}")
            
            if self.connect():
                if self.test_single_symbol(variant):
                    working_symbols.append(variant)
                    logger.info(f"✓ {variant} works!")
                else:
                    logger.warning(f"✗ {variant} failed")
                self.disconnect()
                time.sleep(2)
            
        return working_symbols
    
    def connect(self):
        """Connect to Deriv WebSocket API."""
        try:
            self.is_connected = False
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
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False
    
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
                else:
                    logger.error(f"Authorization failed: {data['error']['message']}")
            
            elif 'candles' in data:
                self._handle_candle_data(data)
            
            elif 'error' in data:
                logger.error(f"API Error: {data['error']['message']}")
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def _handle_candle_data(self, data):
        """Handle candle data response."""
        try:
            candles = data['candles']
            if candles and len(candles) > 0:
                self.received_data['success'] = True
                self.received_data['count'] = len(candles)
                self.received_data['latest_price'] = candles[-1]['close']
                logger.info(f"Received {len(candles)} candles, latest price: {candles[-1]['close']}")
            else:
                self.received_data['success'] = False
                logger.warning("No candle data received")
                
        except Exception as e:
            logger.error(f"Error processing candle data: {str(e)}")
            self.received_data['success'] = False
    
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
    
    def test_single_symbol(self, symbol):
        """Test data retrieval for a specific symbol."""
        try:
            self.received_data = {'success': False}
            
            # Request historical candles
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": 10,
                "end": "latest",
                "start": 1,
                "style": "candles",
                "granularity": 60,  # 1 minute
                "req_id": 2
            }
            
            self._send_request(request)
            
            # Wait for data
            timeout = 10
            while timeout > 0 and not self.received_data.get('success', False):
                time.sleep(0.5)
                timeout -= 0.5
            
            return self.received_data.get('success', False)
            
        except Exception as e:
            logger.error(f"Error testing symbol {symbol}: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket."""
        try:
            if self.ws:
                self.ws.close()
            self.is_connected = False
        except:
            pass

def main():
    """Debug all symbol variations to find working ones."""
    api_token = 'V4FOmNuhORnxbHT'
    
    debugger = SymbolDebugger(api_token)
    
    symbols_to_test = ['boom300', 'crash1000', 'boom1000']
    working_mappings = {}
    
    for symbol in symbols_to_test:
        logger.info(f"\n=== Testing {symbol.upper()} ===")
        working_variants = debugger.test_symbol_variations(symbol)
        if working_variants:
            working_mappings[symbol] = working_variants[0]  # Use first working variant
            logger.info(f"✓ {symbol} -> {working_variants[0]}")
        else:
            logger.error(f"✗ No working variant found for {symbol}")
    
    print("\n=== FINAL WORKING SYMBOL MAPPINGS ===")
    print("Current confirmed working:")
    print("  crash300 -> CRASH300N")
    
    print("\nNew working mappings found:")
    for symbol, mapping in working_mappings.items():
        print(f"  {symbol} -> {mapping}")
    
    return working_mappings

if __name__ == "__main__":
    main()
import os
import json
import websocket
import threading
import time
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymbolTester:
    def __init__(self, api_token):
        self.api_token = api_token
        self.ws_url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
        self.ws = None
        self.is_connected = False
        self.test_results = {}
        
        # Test symbols with correct mappings
        self.test_symbols = {
            'crash300': 'CRASH300N',
            'boom300': 'BOOM300N', 
            'crash1000': 'CRASH1000',
            'boom1000': 'BOOM1000'
        }
        
    def test_all_symbols(self):
        """Test data retrieval for all symbols."""
        results = {}
        
        for app_symbol, deriv_symbol in self.test_symbols.items():
            logger.info(f"Testing {app_symbol} ({deriv_symbol})...")
            
            # Fresh connection for each symbol
            self.disconnect()
            time.sleep(2)
            
            if self.connect():
                result = self.test_symbol_data(deriv_symbol)
                results[app_symbol] = {
                    'deriv_symbol': deriv_symbol,
                    'success': result is not None,
                    'data_points': len(result) if result is not None else 0,
                    'latest_price': result.iloc[-1]['close'] if result is not None and len(result) > 0 else None
                }
            else:
                results[app_symbol] = {
                    'deriv_symbol': deriv_symbol,
                    'success': False,
                    'error': 'Connection failed'
                }
            
            self.disconnect()
            time.sleep(1)
        
        return results
    
    def connect(self):
        """Connect to Deriv WebSocket API."""
        try:
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
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def _handle_candle_data(self, data):
        """Handle candle data response."""
        try:
            candles = data['candles']
            
            if candles:
                df_data = []
                for candle in candles:
                    df_data.append({
                        'timestamp': datetime.fromtimestamp(candle['epoch']),
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': 1000  # Synthetic volume
                    })
                
                self.current_data = pd.DataFrame(df_data)
                logger.info(f"Received {len(df_data)} candles")
            else:
                logger.warning("No candle data received")
                self.current_data = None
                
        except Exception as e:
            logger.error(f"Error processing candle data: {str(e)}")
            self.current_data = None
    
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
    
    def test_symbol_data(self, symbol):
        """Test data retrieval for a specific symbol."""
        try:
            self.current_data = None
            
            # Request historical candles
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": 50,
                "end": "latest",
                "start": 1,
                "style": "candles",
                "granularity": 60,  # 1 minute
                "req_id": 2
            }
            
            self._send_request(request)
            
            # Wait for data
            timeout = 15
            while timeout > 0 and self.current_data is None:
                time.sleep(0.5)
                timeout -= 0.5
            
            return self.current_data
            
        except Exception as e:
            logger.error(f"Error testing symbol {symbol}: {str(e)}")
            return None
    
    def disconnect(self):
        """Disconnect from WebSocket."""
        try:
            if self.ws:
                self.ws.close()
            self.is_connected = False
        except:
            pass

def main():
    """Test all synthetic indices symbols."""
    api_token = 'V4FOmNuhORnxbHT'
    
    tester = SymbolTester(api_token)
    results = tester.test_all_symbols()
    
    print("\n=== SYMBOL TEST RESULTS ===")
    for symbol, result in results.items():
        print(f"\n{symbol.upper()}:")
        print(f"  Deriv Symbol: {result['deriv_symbol']}")
        print(f"  Success: {result['success']}")
        if result['success']:
            print(f"  Data Points: {result['data_points']}")
            print(f"  Latest Price: {result['latest_price']}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    main()
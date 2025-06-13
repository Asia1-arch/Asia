import json
import websocket
import threading
import time
import logging
from datetime import datetime
import pandas as pd
from typing import Optional, Callable, Dict, List

class RealtimePriceStreamer:
    """
    Real-time price streaming for synthetic indices with bid/ask spreads.
    Provides live tick data updates for dynamic chart visualization.
    """
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.ws_url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
        self.ws = None
        self.is_connected = False
        self.is_streaming = False
        
        # Price data storage
        self.live_prices = {}
        self.price_history = {}
        self.bid_ask_spreads = {}
        
        # Callbacks for price updates
        self.price_callbacks = {}
        
        # Symbol mappings
        self.symbol_map = {
            'crash300': 'CRASH300N',
            'boom300': 'BOOM300N', 
            'crash1000': 'CRASH1000',
            'boom1000': 'BOOM1000'
        }
        
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Connect to Deriv WebSocket API for streaming."""
        try:
            if self.ws:
                self.ws.close()
                time.sleep(1)
            
            self.is_connected = False
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in separate thread
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
            self.logger.error(f"Connection error: {str(e)}")
            return False
    
    def _on_open(self, ws):
        """WebSocket connection opened."""
        self.logger.info("Connected to Deriv streaming API")
        self.is_connected = True
        
        # Authorize for streaming
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
                    self.logger.info("Successfully authorized for streaming")
                else:
                    self.logger.error(f"Authorization failed: {data['error']['message']}")
            
            elif 'tick' in data:
                self._handle_tick_data(data['tick'])
            
            elif 'proposal' in data:
                self._handle_proposal_data(data['proposal'])
                
        except Exception as e:
            self.logger.error(f"Error processing streaming message: {str(e)}")
    
    def _handle_tick_data(self, tick):
        """Handle real-time tick data."""
        try:
            symbol = tick.get('symbol', '')
            price = float(tick.get('quote', 0))
            timestamp = datetime.fromtimestamp(tick.get('epoch', time.time()))
            
            # Find app symbol from deriv symbol
            app_symbol = None
            for app_sym, deriv_sym in self.symbol_map.items():
                if deriv_sym == symbol:
                    app_symbol = app_sym
                    break
            
            if app_symbol:
                # Update live price
                self.live_prices[app_symbol] = {
                    'price': price,
                    'timestamp': timestamp,
                    'symbol': symbol
                }
                
                # Add to price history
                if app_symbol not in self.price_history:
                    self.price_history[app_symbol] = []
                
                self.price_history[app_symbol].append({
                    'timestamp': timestamp,
                    'price': price
                })
                
                # Keep only recent history (last 1000 ticks)
                if len(self.price_history[app_symbol]) > 1000:
                    self.price_history[app_symbol] = self.price_history[app_symbol][-1000:]
                
                # Calculate bid/ask spread (simulate spread for synthetic indices)
                spread = self._calculate_spread(price, app_symbol)
                self.bid_ask_spreads[app_symbol] = spread
                
                # Trigger callbacks
                if app_symbol in self.price_callbacks:
                    for callback in self.price_callbacks[app_symbol]:
                        try:
                            callback(app_symbol, price, spread, timestamp)
                        except Exception as e:
                            self.logger.error(f"Error in price callback: {str(e)}")
                            
        except Exception as e:
            self.logger.error(f"Error handling tick data: {str(e)}")
    
    def _handle_proposal_data(self, proposal):
        """Handle proposal data for bid/ask information."""
        try:
            # Extract bid/ask information from proposals if available
            display_value = proposal.get('display_value', '')
            symbol = proposal.get('underlying', '')
            
            # Process proposal data for more accurate spreads
            pass
            
        except Exception as e:
            self.logger.error(f"Error handling proposal data: {str(e)}")
    
    def _calculate_spread(self, price: float, symbol: str) -> Dict[str, float]:
        """Calculate realistic bid/ask spread for synthetic indices."""
        # Typical spreads for synthetic indices (in points)
        spread_configs = {
            'crash300': 0.001,   # 0.1% spread
            'boom300': 0.001,
            'crash1000': 0.002,  # 0.2% spread
            'boom1000': 0.002
        }
        
        spread_percentage = spread_configs.get(symbol, 0.001)
        spread_amount = price * spread_percentage
        
        return {
            'bid': price - (spread_amount / 2),
            'ask': price + (spread_amount / 2),
            'spread': spread_amount
        }
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        self.logger.error(f"Streaming WebSocket error: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        self.logger.info("Streaming connection closed")
        self.is_connected = False
        self.is_streaming = False
    
    def _send_request(self, request):
        """Send request to WebSocket."""
        if self.ws and self.is_connected:
            self.ws.send(json.dumps(request))
    
    def start_streaming(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """Start real-time price streaming for a symbol."""
        if not self.is_connected:
            if not self.connect():
                return False
        
        deriv_symbol = self.symbol_map.get(symbol)
        if not deriv_symbol:
            self.logger.error(f"Unknown symbol: {symbol}")
            return False
        
        # Register callback
        if callback:
            if symbol not in self.price_callbacks:
                self.price_callbacks[symbol] = []
            self.price_callbacks[symbol].append(callback)
        
        # Subscribe to real-time ticks
        tick_request = {
            "ticks": deriv_symbol,
            "subscribe": 1,
            "req_id": f"tick_{symbol}"
        }
        
        self._send_request(tick_request)
        self.is_streaming = True
        self.logger.info(f"Started streaming for {symbol} ({deriv_symbol})")
        return True
    
    def stop_streaming(self, symbol: str):
        """Stop streaming for a symbol."""
        deriv_symbol = self.symbol_map.get(symbol)
        if deriv_symbol:
            forget_request = {
                "forget": f"tick_{symbol}"
            }
            self._send_request(forget_request)
            self.logger.info(f"Stopped streaming for {symbol}")
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get latest price data for a symbol."""
        return self.live_prices.get(symbol)
    
    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent price history for a symbol."""
        history = self.price_history.get(symbol, [])
        return history[-limit:] if history else []
    
    def get_bid_ask_spread(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask spread for a symbol."""
        return self.bid_ask_spreads.get(symbol)
    
    def disconnect(self):
        """Disconnect from streaming API."""
        self.is_streaming = False
        if self.ws:
            self.ws.close()
        self.is_connected = False
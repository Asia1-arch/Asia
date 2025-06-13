import websocket
import json
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional, Callable

class DerivAPI:
    """
    Deriv API client for fetching real-time synthetic indices data.
    Connects to Deriv's WebSocket API to get live market data.
    """
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv('DERIV_API_TOKEN')
        self.ws_url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
        self.ws = None
        self.is_connected = False
        self.subscriptions = {}
        self.data_cache = {}
        self.callbacks = {}
        
        # Symbol mapping for Deriv synthetic indices (verified working symbols)
        self.symbol_map = {
            'crash300': 'CRASH300N',
            'boom300': 'BOOM300N', 
            'crash1000': 'CRASH1000',
            'boom1000': 'BOOM1000',
            'crash500': 'CRASH500',
            'boom500': 'BOOM500'
        }
        
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
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
            self.logger.error(f"Error connecting to Deriv API: {str(e)}")
            return False
    
    def _on_open(self, ws):
        """WebSocket connection opened."""
        self.logger.info("Connected to Deriv API")
        self.is_connected = True
        
        # Authorize if token is provided
        if self.api_token:
            self._authorize()
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if 'tick' in data:
                self._handle_tick_data(data)
            elif 'candles' in data:
                self._handle_candle_data(data)
            elif 'ohlc' in data:
                self._handle_ohlc_data(data)
            elif 'authorize' in data:
                self._handle_auth_response(data)
            elif 'error' in data:
                self._handle_error(data)
                
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        self.logger.error(f"WebSocket error: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        self.logger.info("Deriv API connection closed")
        self.is_connected = False
    
    def _authorize(self):
        """Authorize with API token."""
        auth_request = {
            "authorize": self.api_token,
            "req_id": 1
        }
        self._send_request(auth_request)
    
    def _send_request(self, request: dict):
        """Send request to WebSocket."""
        if self.ws and self.is_connected:
            self.ws.send(json.dumps(request))
    
    def _handle_auth_response(self, data):
        """Handle authorization response."""
        if 'error' in data:
            self.logger.error(f"Authorization failed: {data['error']['message']}")
        else:
            self.logger.info("Successfully authorized with Deriv API")
    
    def _handle_tick_data(self, data):
        """Handle real-time tick data."""
        tick = data['tick']
        symbol = tick.get('symbol', '')
        
        if symbol in self.data_cache:
            # Update latest tick
            self.data_cache[symbol]['latest_tick'] = {
                'price': tick['quote'],
                'timestamp': datetime.fromtimestamp(tick['epoch'])
            }
            
            # Call registered callbacks
            if symbol in self.callbacks:
                for callback in self.callbacks[symbol]:
                    try:
                        callback(tick)
                    except Exception as e:
                        self.logger.error(f"Error in callback: {str(e)}")
    
    def _handle_candle_data(self, data):
        """Handle historical candle data."""
        candles = data['candles']
        req_id = data.get('req_id')
        
        if req_id in self.subscriptions:
            symbol = self.subscriptions[req_id]['symbol']
            
            # Convert to DataFrame
            df_data = []
            for candle in candles:
                df_data.append({
                    'timestamp': datetime.fromtimestamp(candle['epoch']),
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close'],
                    'volume': 1000  # Synthetic indices don't have real volume
                })
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df = df.sort_values('timestamp').reset_index(drop=True)
                self.data_cache[symbol] = {
                    'historical_data': df,
                    'last_updated': datetime.now()
                }
    
    def _handle_ohlc_data(self, data):
        """Handle OHLC subscription data."""
        ohlc = data['ohlc']
        symbol = ohlc.get('symbol', '')
        
        if symbol not in self.data_cache:
            self.data_cache[symbol] = {'ohlc_data': []}
        
        # Add new OHLC data
        new_ohlc = {
            'timestamp': datetime.fromtimestamp(ohlc['epoch']),
            'open': ohlc['open'],
            'high': ohlc['high'],
            'low': ohlc['low'],
            'close': ohlc['close'],
            'volume': 1000
        }
        
        self.data_cache[symbol]['ohlc_data'].append(new_ohlc)
        
        # Keep only recent data (last 500 candles)
        if len(self.data_cache[symbol]['ohlc_data']) > 500:
            self.data_cache[symbol]['ohlc_data'] = self.data_cache[symbol]['ohlc_data'][-500:]
    
    def _handle_error(self, data):
        """Handle API errors."""
        error = data['error']
        self.logger.error(f"API Error: {error['message']} (Code: {error['code']})")
    
    def get_historical_data(self, symbol: str, timeframe: str = '1m', count: int = 200) -> Optional[pd.DataFrame]:
        """
        Get historical candle data for a symbol.
        
        Args:
            symbol (str): Symbol name (e.g., 'crash300', 'boom1000')
            timeframe (str): Timeframe ('1m', '5m', '15m', '1h')
            count (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        if not self.is_connected:
            if not self.connect():
                return None
        
        # Map symbol to Deriv format
        deriv_symbol = self._get_deriv_symbol(symbol)
        if not deriv_symbol:
            self.logger.error(f"Unknown symbol: {symbol}")
            return None
        
        # Map timeframe to granularity
        granularity = self._get_granularity(timeframe)
        if not granularity:
            self.logger.error(f"Unknown timeframe: {timeframe}")
            return None
        
        # Calculate start time
        end_time = int(time.time())
        start_time = end_time - (count * granularity)
        
        # Create request
        req_id = int(time.time() * 1000)
        request = {
            "ticks_history": deriv_symbol,
            "start": start_time,
            "end": end_time,
            "granularity": granularity,
            "style": "candles",
            "count": count,
            "req_id": req_id
        }
        
        # Store subscription info
        self.subscriptions[req_id] = {
            'symbol': symbol,
            'type': 'historical'
        }
        
        # Send request
        self._send_request(request)
        
        # Wait for response with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Reconnect for retry attempts
                    self.disconnect()
                    time.sleep(2)
                    if not self.connect():
                        continue
                    self._send_request(request)
                
                timeout = 15
                while timeout > 0:
                    if symbol in self.data_cache and 'historical_data' in self.data_cache[symbol]:
                        data = self.data_cache[symbol]['historical_data']
                        self.logger.info(f"Retrieved {len(data)} candles for {symbol}")
                        return data
                    time.sleep(0.5)
                    timeout -= 0.5
                
                self.logger.warning(f"Timeout on attempt {attempt + 1} for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt + 1} for {symbol}: {str(e)}")
        
        self.logger.error(f"Failed to retrieve data for {symbol} after {max_retries} attempts")
        return None
    
    def subscribe_ticks(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to real-time tick data.
        
        Args:
            symbol (str): Symbol to subscribe to
            callback (Callable): Optional callback function for tick data
            
        Returns:
            bool: True if subscription successful
        """
        if not self.is_connected:
            if not self.connect():
                return False
        
        deriv_symbol = self._get_deriv_symbol(symbol)
        if not deriv_symbol:
            return False
        
        # Register callback
        if callback:
            if symbol not in self.callbacks:
                self.callbacks[symbol] = []
            self.callbacks[symbol].append(callback)
        
        # Subscribe to ticks
        request = {
            "ticks": deriv_symbol,
            "subscribe": 1
        }
        
        self._send_request(request)
        return True
    
    def subscribe_ohlc(self, symbol: str, timeframe: str = '1m') -> bool:
        """
        Subscribe to real-time OHLC data.
        
        Args:
            symbol (str): Symbol to subscribe to
            timeframe (str): Timeframe for OHLC data
            
        Returns:
            bool: True if subscription successful
        """
        if not self.is_connected:
            if not self.connect():
                return False
        
        deriv_symbol = self._get_deriv_symbol(symbol)
        granularity = self._get_granularity(timeframe)
        
        if not deriv_symbol or not granularity:
            return False
        
        request = {
            "ticks_history": deriv_symbol,
            "granularity": granularity,
            "style": "candles",
            "subscribe": 1
        }
        
        self._send_request(request)
        return True
    
    def _get_deriv_symbol(self, symbol: str) -> Optional[str]:
        """Convert app symbol to Deriv symbol."""
        symbol_lower = symbol.lower().replace(" ", "")
        
        # Direct mapping
        if symbol_lower in self.symbol_map:
            return self.symbol_map[symbol_lower]
        
        # Alternative mappings
        symbol_mappings = {
            'crash300': 'R_10',
            'boom300': 'R_25',
            'crash1000': 'R_100', 
            'boom1000': 'R_75',
            'crash500': 'R_50',
            'boom500': 'R_50',
            'volatility10': 'R_10',
            'volatility25': 'R_25',
            'volatility50': 'R_50',
            'volatility75': 'R_75',
            'volatility100': 'R_100'
        }
        
        return symbol_mappings.get(symbol_lower)
    
    def _get_granularity(self, timeframe: str) -> Optional[int]:
        """Convert timeframe to Deriv granularity (seconds)."""
        timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        
        return timeframe_map.get(timeframe.lower())
    
    def get_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached data for a symbol."""
        if symbol in self.data_cache:
            if 'historical_data' in self.data_cache[symbol]:
                return self.data_cache[symbol]['historical_data']
            elif 'ohlc_data' in self.data_cache[symbol]:
                return pd.DataFrame(self.data_cache[symbol]['ohlc_data'])
        
        return None
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        if symbol in self.data_cache and 'latest_tick' in self.data_cache[symbol]:
            return self.data_cache[symbol]['latest_tick']['price']
        
        return None
    
    def disconnect(self):
        """Disconnect from WebSocket."""
        if self.ws:
            self.ws.close()
        self.is_connected = False
    
    def is_symbol_available(self, symbol: str) -> bool:
        """Check if symbol is available."""
        return self._get_deriv_symbol(symbol) is not None
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        return list(self.symbol_map.keys())
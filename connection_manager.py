import json
import websocket
import threading
import time
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, List
from queue import Queue
import os

class StableConnectionManager:
    """
    Stable WebSocket connection manager for smooth real-time trading data.
    Maintains persistent connections with intelligent reconnection and heartbeat.
    """
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv('DERIV_API_TOKEN')
        self.ws_url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
        self.ws = None
        self.is_connected = False
        self.is_authenticated = False
        
        # Connection stability features
        self.connection_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 2  # seconds
        self.heartbeat_interval = 30  # seconds
        self.last_heartbeat = None
        
        # Threading
        self.ws_thread = None
        self.heartbeat_thread = None
        self.reconnect_thread = None
        self._stop_event = threading.Event()
        
        # Data management
        self.message_queue = Queue()
        self.subscribers = {}  # {symbol: [callbacks]}
        self.active_subscriptions = set()
        
        # Symbol mappings for synthetic indices
        self.symbol_map = {
            'crash300': 'R_10',
            'boom300': 'R_25', 
            'crash1000': 'R_50',
            'boom1000': 'R_75',
            'crash500': 'R_100'
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def start(self) -> bool:
        """Start the stable connection manager."""
        if self.is_connected:
            return True
            
        self._stop_event.clear()
        return self._establish_connection()
    
    def stop(self):
        """Stop all connections and threads."""
        self._stop_event.set()
        self.is_connected = False
        self.is_authenticated = False
        
        if self.ws:
            self.ws.close()
            
        # Wait for threads to finish
        for thread in [self.ws_thread, self.heartbeat_thread, self.reconnect_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2)
    
    def _establish_connection(self) -> bool:
        """Establish WebSocket connection with retry logic."""
        while self.connection_attempts < self.max_reconnect_attempts and not self._stop_event.is_set():
            try:
                self.logger.info(f"Connecting to Deriv API (attempt {self.connection_attempts + 1})")
                
                # Close existing connection
                if self.ws:
                    self.ws.close()
                    time.sleep(1)
                
                # Create new WebSocket connection
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                
                # Start WebSocket in separate thread
                self.ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={
                    'ping_interval': 20,
                    'ping_timeout': 10
                })
                self.ws_thread.daemon = True
                self.ws_thread.start()
                
                # Wait for connection with timeout
                timeout = 15
                while not self.is_connected and timeout > 0 and not self._stop_event.is_set():
                    time.sleep(0.5)
                    timeout -= 0.5
                
                if self.is_connected:
                    self.connection_attempts = 0
                    self._start_heartbeat()
                    self.logger.info("Successfully connected to Deriv API")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Connection error: {str(e)}")
                
            self.connection_attempts += 1
            if self.connection_attempts < self.max_reconnect_attempts:
                delay = min(self.reconnect_delay * (2 ** self.connection_attempts), 30)
                self.logger.info(f"Retrying connection in {delay} seconds...")
                time.sleep(delay)
        
        self.logger.error("Max reconnection attempts reached")
        return False
    
    def _on_open(self, ws):
        """WebSocket connection opened."""
        self.is_connected = True
        self.last_heartbeat = datetime.now()
        self.logger.info("WebSocket connection established")
        
        # Skip authorization if no token (use public feeds)
        if self.api_token:
            self._authenticate()
        else:
            self.is_authenticated = True
            self._resubscribe_all()
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Handle authentication
            if 'authorize' in data:
                if 'error' not in data:
                    self.is_authenticated = True
                    self.logger.info("Successfully authenticated")
                    self._resubscribe_all()
                else:
                    self.logger.warning(f"Authentication failed: {data['error']['message']}")
                    # Continue with public feeds only
                    self.is_authenticated = True
                    self._resubscribe_all()
            
            # Handle tick data
            elif 'tick' in data:
                self._handle_tick_data(data)
            
            # Handle heartbeat responses
            elif 'ping' in data:
                self.last_heartbeat = datetime.now()
                
            # Handle errors
            elif 'error' in data:
                error_code = data['error'].get('code', '')
                if error_code not in ['InvalidToken']:  # Don't log token errors repeatedly
                    self.logger.warning(f"API Error: {data['error']['message']}")
                    
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        self.logger.warning(f"WebSocket error: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        self.is_connected = False
        self.is_authenticated = False
        self.logger.info("WebSocket connection closed")
        
        # Auto-reconnect if not intentionally stopped
        if not self._stop_event.is_set():
            self._schedule_reconnect()
    
    def _authenticate(self):
        """Authenticate with API token."""
        if self.api_token and self.is_connected:
            auth_request = {
                "authorize": self.api_token,
                "req_id": "auth"
            }
            self._send_request(auth_request)
    
    def _start_heartbeat(self):
        """Start heartbeat thread to maintain connection."""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
            
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
    
    def _heartbeat_loop(self):
        """Heartbeat loop to keep connection alive."""
        while not self._stop_event.is_set() and self.is_connected:
            try:
                # Send ping every 30 seconds
                time.sleep(self.heartbeat_interval)
                
                if self.is_connected and not self._stop_event.is_set():
                    ping_request = {
                        "ping": 1,
                        "req_id": "heartbeat"
                    }
                    self._send_request(ping_request)
                    
                    # Check if connection is stale
                    if self.last_heartbeat:
                        time_since_heartbeat = (datetime.now() - self.last_heartbeat).seconds
                        if time_since_heartbeat > 60:  # 1 minute timeout
                            self.logger.warning("Connection appears stale, reconnecting...")
                            self._schedule_reconnect()
                            break
                            
            except Exception as e:
                self.logger.error(f"Heartbeat error: {str(e)}")
    
    def _schedule_reconnect(self):
        """Schedule automatic reconnection."""
        if self.reconnect_thread and self.reconnect_thread.is_alive():
            return
            
        self.reconnect_thread = threading.Thread(target=self._reconnect_loop)
        self.reconnect_thread.daemon = True
        self.reconnect_thread.start()
    
    def _reconnect_loop(self):
        """Automatic reconnection loop."""
        if self._stop_event.is_set():
            return
            
        self.logger.info("Attempting to reconnect...")
        time.sleep(2)  # Brief delay before reconnect
        
        if self._establish_connection():
            self.logger.info("Reconnection successful")
        else:
            self.logger.error("Reconnection failed")
    
    def _send_request(self, request: dict):
        """Send request to WebSocket safely."""
        try:
            if self.ws and self.is_connected:
                self.ws.send(json.dumps(request))
        except Exception as e:
            self.logger.error(f"Error sending request: {str(e)}")
    
    def _handle_tick_data(self, data):
        """Handle incoming tick data."""
        try:
            tick = data.get('tick', {})
            symbol = tick.get('symbol', '')
            price = float(tick.get('quote', 0))
            timestamp = datetime.fromtimestamp(tick.get('epoch', time.time()))
            
            # Find app symbol
            app_symbol = None
            for app_sym, deriv_sym in self.symbol_map.items():
                if deriv_sym == symbol:
                    app_symbol = app_sym
                    break
            
            if app_symbol and app_symbol in self.subscribers:
                # Notify all subscribers
                for callback in self.subscribers[app_symbol]:
                    try:
                        callback(app_symbol, price, timestamp)
                    except Exception as e:
                        self.logger.error(f"Callback error: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error handling tick data: {str(e)}")
    
    def subscribe_symbol(self, symbol: str, callback: Callable) -> bool:
        """Subscribe to real-time data for a symbol."""
        if symbol not in self.symbol_map:
            self.logger.error(f"Unknown symbol: {symbol}")
            return False
        
        # Add callback to subscribers
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
        
        # Subscribe if connected
        if self.is_connected:
            return self._subscribe_symbol(symbol)
        else:
            # Will be subscribed when connection is established
            return True
    
    def _subscribe_symbol(self, symbol: str) -> bool:
        """Internal method to subscribe to a symbol."""
        deriv_symbol = self.symbol_map.get(symbol)
        if not deriv_symbol:
            return False
        
        subscribe_request = {
            "ticks": deriv_symbol,
            "subscribe": 1,
            "req_id": f"tick_{symbol}"
        }
        
        self._send_request(subscribe_request)
        self.active_subscriptions.add(symbol)
        self.logger.info(f"Subscribed to {symbol} ({deriv_symbol})")
        return True
    
    def _resubscribe_all(self):
        """Resubscribe to all active symbols after reconnection."""
        for symbol in list(self.subscribers.keys()):
            self._subscribe_symbol(symbol)
    
    def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from a symbol."""
        if symbol in self.active_subscriptions:
            forget_request = {
                "forget": f"tick_{symbol}",
                "req_id": f"forget_{symbol}"
            }
            self._send_request(forget_request)
            self.active_subscriptions.discard(symbol)
        
        # Remove from subscribers
        if symbol in self.subscribers:
            del self.subscribers[symbol]
    
    def is_ready(self) -> bool:
        """Check if connection is ready for use."""
        return self.is_connected and self.is_authenticated
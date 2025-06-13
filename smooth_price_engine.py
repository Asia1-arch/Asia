import threading
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from connection_manager import StableConnectionManager

@dataclass
class PricePoint:
    """Single price point with metadata."""
    price: float
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: int = 0

@dataclass 
class MarketState:
    """Current market state for a symbol."""
    symbol: str
    current_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    last_update: Optional[datetime] = None
    price_history: List[PricePoint] = field(default_factory=list)
    trend: str = "neutral"  # up, down, neutral
    volatility: float = 0.0
    daily_change: float = 0.0
    daily_change_pct: float = 0.0

class SmoothPriceEngine:
    """
    Smooth real-time price engine for trading platforms.
    Provides seamless price updates without connection interruptions.
    """
    
    def __init__(self):
        self.connection_manager = StableConnectionManager()
        self.market_states: Dict[str, MarketState] = {}
        self.price_callbacks: Dict[str, List[Callable]] = {}
        
        # Price smoothing settings
        self.smoothing_enabled = True
        self.interpolation_steps = 5
        self.update_frequency = 200  # milliseconds
        
        # Threading
        self.price_thread = None
        self.smoothing_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Price generation for smooth movement
        self.pending_updates = {}
        self.interpolated_prices = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Symbol configurations
        self.symbol_configs = {
            'crash300': {
                'base_spread': 0.001,
                'volatility_factor': 1.2,
                'min_move': 0.001
            },
            'boom300': {
                'base_spread': 0.001,
                'volatility_factor': 1.1,
                'min_move': 0.001
            },
            'crash1000': {
                'base_spread': 0.002,
                'volatility_factor': 1.5,
                'min_move': 0.002
            },
            'boom1000': {
                'base_spread': 0.002,
                'volatility_factor': 1.3,
                'min_move': 0.002
            }
        }
    
    def start(self) -> bool:
        """Start the smooth price engine."""
        if not self.connection_manager.start():
            self.logger.error("Failed to start connection manager")
            return False
        
        self._stop_event.clear()
        
        # Start price smoothing thread
        self.smoothing_thread = threading.Thread(target=self._smoothing_loop)
        self.smoothing_thread.daemon = True
        self.smoothing_thread.start()
        
        self.logger.info("Smooth price engine started")
        return True
    
    def stop(self):
        """Stop the price engine."""
        self._stop_event.set()
        self.connection_manager.stop()
        
        if self.smoothing_thread and self.smoothing_thread.is_alive():
            self.smoothing_thread.join(timeout=2)
    
    def subscribe_symbol(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """Subscribe to smooth price updates for a symbol."""
        with self._lock:
            # Initialize market state
            if symbol not in self.market_states:
                self.market_states[symbol] = MarketState(symbol=symbol)
            
            # Register callback
            if callback:
                if symbol not in self.price_callbacks:
                    self.price_callbacks[symbol] = []
                self.price_callbacks[symbol].append(callback)
        
        # Subscribe to raw data
        return self.connection_manager.subscribe_symbol(symbol, self._handle_raw_price)
    
    def _handle_raw_price(self, symbol: str, price: float, timestamp: datetime):
        """Handle raw price data from connection manager."""
        with self._lock:
            if symbol not in self.market_states:
                return
            
            market_state = self.market_states[symbol]
            config = self.symbol_configs.get(symbol, self.symbol_configs['crash300'])
            
            # Calculate bid/ask spread
            spread_amount = price * config['base_spread']
            bid = price - (spread_amount / 2)
            ask = price + (spread_amount / 2)
            
            # Create price point
            price_point = PricePoint(
                price=price,
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                spread=spread_amount
            )
            
            # Update market state
            previous_price = market_state.current_price
            market_state.current_price = price
            market_state.bid = bid
            market_state.ask = ask
            market_state.spread = spread_amount
            market_state.last_update = timestamp
            
            # Add to history
            market_state.price_history.append(price_point)
            if len(market_state.price_history) > 1000:
                market_state.price_history = market_state.price_history[-1000:]
            
            # Calculate trend and statistics
            self._update_market_statistics(symbol, previous_price)
            
            # Setup smooth transition if enabled
            if self.smoothing_enabled and previous_price > 0:
                self._setup_smooth_transition(symbol, previous_price, price, timestamp)
            else:
                # Direct update without smoothing
                self._notify_callbacks(symbol, market_state)
    
    def _setup_smooth_transition(self, symbol: str, start_price: float, end_price: float, timestamp: datetime):
        """Setup smooth price transition between two points."""
        if abs(end_price - start_price) < self.symbol_configs.get(symbol, {}).get('min_move', 0.001):
            return  # Price change too small to smooth
        
        # Create interpolation steps
        steps = []
        for i in range(1, self.interpolation_steps + 1):
            ratio = i / self.interpolation_steps
            interpolated_price = start_price + (end_price - start_price) * ratio
            steps.append(interpolated_price)
        
        self.pending_updates[symbol] = {
            'steps': steps,
            'current_step': 0,
            'start_time': datetime.now(),
            'target_price': end_price,
            'timestamp': timestamp
        }
    
    def _smoothing_loop(self):
        """Main smoothing loop for price interpolation."""
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    for symbol in list(self.pending_updates.keys()):
                        update_info = self.pending_updates[symbol]
                        
                        if update_info['current_step'] < len(update_info['steps']):
                            # Get next interpolated price
                            price = update_info['steps'][update_info['current_step']]
                            update_info['current_step'] += 1
                            
                            # Update market state with interpolated price
                            market_state = self.market_states.get(symbol)
                            if market_state:
                                config = self.symbol_configs.get(symbol, self.symbol_configs['crash300'])
                                spread_amount = price * config['base_spread']
                                
                                market_state.current_price = price
                                market_state.bid = price - (spread_amount / 2)
                                market_state.ask = price + (spread_amount / 2)
                                market_state.spread = spread_amount
                                
                                # Notify callbacks with smooth update
                                self._notify_callbacks(symbol, market_state)
                        else:
                            # Finished interpolation
                            del self.pending_updates[symbol]
                
                time.sleep(self.update_frequency / 1000.0)  # Convert to seconds
                
            except Exception as e:
                self.logger.error(f"Error in smoothing loop: {str(e)}")
    
    def _update_market_statistics(self, symbol: str, previous_price: float):
        """Update market statistics and trends."""
        market_state = self.market_states[symbol]
        
        # Calculate trend
        if previous_price > 0:
            if market_state.current_price > previous_price:
                market_state.trend = "up"
            elif market_state.current_price < previous_price:
                market_state.trend = "down"
            else:
                market_state.trend = "neutral"
        
        # Calculate volatility from recent history
        if len(market_state.price_history) >= 20:
            recent_prices = [p.price for p in market_state.price_history[-20:]]
            volatility_val = np.std(recent_prices)
            market_state.volatility = float(volatility_val) if volatility_val is not None else 0.0
        
        # Calculate daily change (mock for demonstration)
        if len(market_state.price_history) >= 100:
            start_price = market_state.price_history[-100].price
            market_state.daily_change = market_state.current_price - start_price
            market_state.daily_change_pct = (market_state.daily_change / start_price) * 100
    
    def _notify_callbacks(self, symbol: str, market_state: MarketState):
        """Notify all registered callbacks for a symbol."""
        callbacks = self.price_callbacks.get(symbol, [])
        for callback in callbacks:
            try:
                callback(symbol, market_state)
            except Exception as e:
                self.logger.error(f"Callback error for {symbol}: {str(e)}")
    
    def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """Get current market state for a symbol."""
        return self.market_states.get(symbol)
    
    def get_price_history(self, symbol: str, periods: int = 100) -> List[PricePoint]:
        """Get price history for a symbol."""
        market_state = self.market_states.get(symbol)
        if market_state:
            return market_state.price_history[-periods:]
        return []
    
    def is_connected(self) -> bool:
        """Check if the engine is connected and ready."""
        return self.connection_manager.is_ready()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status."""
        return {
            'connected': self.connection_manager.is_connected,
            'authenticated': self.connection_manager.is_authenticated,
            'active_symbols': list(self.market_states.keys()),
            'smoothing_enabled': self.smoothing_enabled,
            'pending_updates': len(self.pending_updates)
        }
    
    def set_smoothing(self, enabled: bool, interpolation_steps: int = 5, update_frequency: int = 200):
        """Configure price smoothing settings."""
        with self._lock:
            self.smoothing_enabled = enabled
            self.interpolation_steps = max(1, interpolation_steps)
            self.update_frequency = max(50, update_frequency)  # Minimum 50ms
    
    def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from a symbol."""
        with self._lock:
            self.connection_manager.unsubscribe_symbol(symbol)
            
            if symbol in self.market_states:
                del self.market_states[symbol]
            if symbol in self.price_callbacks:
                del self.price_callbacks[symbol]
            if symbol in self.pending_updates:
                del self.pending_updates[symbol]
import threading
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from connection_manager import StableConnectionManager
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@dataclass
class LiveCandle:
    """Real-time candlestick data."""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    timestamp: Optional[datetime] = None
    is_forming: bool = True
    tick_count: int = 0

@dataclass
class BidAskData:
    """Real-time bid/ask spread data."""
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    timestamp: Optional[datetime] = None

class RealtimeCandlestickEngine:
    """
    Real-time candlestick engine that shows smooth candle formation
    with live bid/ask lines, simulating professional trading platforms.
    """
    
    def __init__(self, timeframe_seconds: int = 60):
        self.connection_manager = StableConnectionManager()
        self.timeframe_seconds = timeframe_seconds
        
        # Candle data storage
        self.live_candles: Dict[str, List[LiveCandle]] = {}
        self.current_candles: Dict[str, LiveCandle] = {}
        self.bid_ask_data: Dict[str, BidAskData] = {}
        
        # Callbacks for real-time updates
        self.candle_callbacks: Dict[str, List[Callable]] = {}
        self.bid_ask_callbacks: Dict[str, List[Callable]] = {}
        
        # Threading
        self.candle_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Symbol configurations for realistic spreads
        self.symbol_configs = {
            'crash300': {
                'base_spread_pct': 0.0008,  # 0.08%
                'volatility_factor': 1.2,
                'min_tick': 0.0001
            },
            'boom300': {
                'base_spread_pct': 0.0008,
                'volatility_factor': 1.1,
                'min_tick': 0.0001
            },
            'crash1000': {
                'base_spread_pct': 0.0012,  # 0.12%
                'volatility_factor': 1.5,
                'min_tick': 0.0001
            },
            'boom1000': {
                'base_spread_pct': 0.0012,
                'volatility_factor': 1.3,
                'min_tick': 0.0001
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> bool:
        """Start the real-time candlestick engine."""
        if not self.connection_manager.start():
            self.logger.error("Failed to start connection manager")
            return False
        
        self._stop_event.clear()
        
        # Start candle formation thread
        self.candle_thread = threading.Thread(target=self._candle_formation_loop)
        self.candle_thread.daemon = True
        self.candle_thread.start()
        
        self.logger.info("Real-time candlestick engine started")
        return True
    
    def stop(self):
        """Stop the candlestick engine."""
        self._stop_event.set()
        self.connection_manager.stop()
        
        if self.candle_thread and self.candle_thread.is_alive():
            self.candle_thread.join(timeout=2)
    
    def subscribe_symbol(self, symbol: str, 
                        candle_callback: Optional[Callable] = None,
                        bid_ask_callback: Optional[Callable] = None) -> bool:
        """Subscribe to real-time candlestick and bid/ask data for a symbol."""
        with self._lock:
            # Initialize storage
            if symbol not in self.live_candles:
                self.live_candles[symbol] = []
                self.current_candles[symbol] = LiveCandle(timestamp=datetime.now())
                self.bid_ask_data[symbol] = BidAskData(timestamp=datetime.now())
            
            # Register callbacks
            if candle_callback:
                if symbol not in self.candle_callbacks:
                    self.candle_callbacks[symbol] = []
                self.candle_callbacks[symbol].append(candle_callback)
            
            if bid_ask_callback:
                if symbol not in self.bid_ask_callbacks:
                    self.bid_ask_callbacks[symbol] = []
                self.bid_ask_callbacks[symbol].append(bid_ask_callback)
        
        # Subscribe to raw price data
        return self.connection_manager.subscribe_symbol(symbol, self._handle_tick)
    
    def _handle_tick(self, symbol: str, price: float, timestamp: datetime):
        """Handle incoming tick data to form candles and update bid/ask."""
        with self._lock:
            if symbol not in self.current_candles:
                return
            
            current_candle = self.current_candles[symbol]
            config = self.symbol_configs.get(symbol, self.symbol_configs['crash300'])
            
            # Calculate current candle period
            candle_start = self._get_candle_start_time(timestamp)
            
            # Check if we need to start a new candle
            if (current_candle.timestamp is None or 
                candle_start > current_candle.timestamp):
                
                # Finalize previous candle
                if current_candle.timestamp is not None and current_candle.tick_count > 0:
                    current_candle.is_forming = False
                    self.live_candles[symbol].append(current_candle)
                    
                    # Keep only recent candles (last 200)
                    if len(self.live_candles[symbol]) > 200:
                        self.live_candles[symbol] = self.live_candles[symbol][-200:]
                
                # Start new candle
                current_candle = LiveCandle(
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    timestamp=candle_start,
                    is_forming=True,
                    tick_count=1
                )
                self.current_candles[symbol] = current_candle
            else:
                # Update current candle
                current_candle.high = max(current_candle.high, price)
                current_candle.low = min(current_candle.low, price)
                current_candle.close = price
                current_candle.tick_count += 1
            
            # Update bid/ask spread
            spread_amount = price * config['base_spread_pct']
            bid_ask = BidAskData(
                bid=price - (spread_amount / 2),
                ask=price + (spread_amount / 2),
                spread=spread_amount,
                timestamp=timestamp
            )
            self.bid_ask_data[symbol] = bid_ask
            
            # Notify callbacks
            self._notify_candle_callbacks(symbol, current_candle)
            self._notify_bid_ask_callbacks(symbol, bid_ask)
    
    def _get_candle_start_time(self, timestamp: datetime) -> datetime:
        """Get the start time for the current candle period."""
        # Round down to nearest timeframe interval
        seconds_since_epoch = timestamp.timestamp()
        interval_start = (seconds_since_epoch // self.timeframe_seconds) * self.timeframe_seconds
        return datetime.fromtimestamp(interval_start)
    
    def _candle_formation_loop(self):
        """Background loop for smooth candle formation updates."""
        while not self._stop_event.is_set():
            try:
                # Update candle formation progress every 100ms for smooth updates
                time.sleep(0.1)
                
                with self._lock:
                    current_time = datetime.now()
                    
                    for symbol in list(self.current_candles.keys()):
                        current_candle = self.current_candles[symbol]
                        
                        if current_candle.timestamp:
                            # Calculate candle formation progress
                            candle_end = current_candle.timestamp + timedelta(seconds=self.timeframe_seconds)
                            if current_time >= candle_end and current_candle.is_forming:
                                # Force candle completion
                                current_candle.is_forming = False
                                self.live_candles[symbol].append(current_candle)
                                
                                # Start new candle with last close as open
                                new_candle = LiveCandle(
                                    open=current_candle.close,
                                    high=current_candle.close,
                                    low=current_candle.close,
                                    close=current_candle.close,
                                    timestamp=self._get_candle_start_time(current_time),
                                    is_forming=True,
                                    tick_count=1
                                )
                                self.current_candles[symbol] = new_candle
                                
                                # Notify of new candle
                                self._notify_candle_callbacks(symbol, new_candle)
                        
            except Exception as e:
                self.logger.error(f"Error in candle formation loop: {str(e)}")
    
    def _notify_candle_callbacks(self, symbol: str, candle: LiveCandle):
        """Notify candle callbacks."""
        callbacks = self.candle_callbacks.get(symbol, [])
        for callback in callbacks:
            try:
                callback(symbol, candle)
            except Exception as e:
                self.logger.error(f"Candle callback error for {symbol}: {str(e)}")
    
    def _notify_bid_ask_callbacks(self, symbol: str, bid_ask: BidAskData):
        """Notify bid/ask callbacks."""
        callbacks = self.bid_ask_callbacks.get(symbol, [])
        for callback in callbacks:
            try:
                callback(symbol, bid_ask)
            except Exception as e:
                self.logger.error(f"Bid/ask callback error for {symbol}: {str(e)}")
    
    def get_candles_dataframe(self, symbol: str, include_forming: bool = True) -> pd.DataFrame:
        """Get candlestick data as DataFrame for charting."""
        with self._lock:
            candles = self.live_candles.get(symbol, []).copy()
            
            if include_forming and symbol in self.current_candles:
                current = self.current_candles[symbol]
                if current.timestamp:
                    candles.append(current)
            
            if not candles:
                return pd.DataFrame()
            
            data = []
            for candle in candles:
                data.append({
                    'timestamp': candle.timestamp,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume,
                    'is_forming': candle.is_forming
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
    
    def get_current_bid_ask(self, symbol: str) -> Optional[BidAskData]:
        """Get current bid/ask data for a symbol."""
        return self.bid_ask_data.get(symbol)
    
    def create_live_candlestick_chart(self, symbol: str) -> go.Figure:
        """Create real-time candlestick chart with bid/ask lines."""
        df = self.get_candles_dataframe(symbol, include_forming=True)
        bid_ask = self.get_current_bid_ask(symbol)
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Waiting for candle data...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Create candlestick chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=[f'{symbol.upper()} - Live Candlesticks', 'Volume'],
            row_width=[0.8, 0.2]
        )
        
        # Separate completed and forming candles
        completed_df = df[~df['is_forming']] if 'is_forming' in df.columns else df
        forming_df = df[df['is_forming']] if 'is_forming' in df.columns else pd.DataFrame()
        
        # Add completed candles
        if not completed_df.empty:
            fig.add_trace(
                go.Candlestick(
                    x=completed_df.index,
                    open=completed_df['open'],
                    high=completed_df['high'],
                    low=completed_df['low'],
                    close=completed_df['close'],
                    name='Completed Candles',
                    increasing_line_color='#00FF88',
                    decreasing_line_color='#FF4444',
                    opacity=0.9
                ),
                row=1, col=1
            )
        
        # Add forming candle with different style
        if not forming_df.empty:
            fig.add_trace(
                go.Candlestick(
                    x=forming_df.index,
                    open=forming_df['open'],
                    high=forming_df['high'],
                    low=forming_df['low'],
                    close=forming_df['close'],
                    name='Forming Candle',
                    increasing_line_color='#88FF88',
                    decreasing_line_color='#FF8888',
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Add bid/ask lines if available
        if bid_ask and not df.empty:
            x_range = [df.index.min(), df.index.max()]
            
            # Bid line
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[bid_ask.bid, bid_ask.bid],
                    mode='lines',
                    name='Bid',
                    line=dict(color='#FF6B6B', width=2, dash='dot'),
                    opacity=0.8
                ),
                row=1, col=1
            )
            
            # Ask line
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[bid_ask.ask, bid_ask.ask],
                    mode='lines',
                    name='Ask',
                    line=dict(color='#4ECDC4', width=2, dash='dot'),
                    opacity=0.8
                ),
                row=1, col=1
            )
            
            # Spread annotation
            fig.add_annotation(
                x=df.index[-1] if not df.empty else datetime.now(),
                y=bid_ask.ask,
                text=f"Spread: {bid_ask.spread:.5f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#666666",
                bordercolor="#666666",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.8)"
            )
        
        # Add volume bars
        if not df.empty:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color='rgba(158,202,225,0.5)',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Update layout for professional appearance
        fig.update_layout(
            title=f"{symbol.upper()} - Real-time Candlesticks with Bid/Ask",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            height=600,
            template="plotly_dark",
            font=dict(color="white"),
            plot_bgcolor='rgba(17,17,17,1)',
            paper_bgcolor='rgba(17,17,17,1)'
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        )
        fig.update_yaxes(
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        )
        
        return fig
    
    def is_connected(self) -> bool:
        """Check if the engine is connected."""
        return self.connection_manager.is_ready()
    
    def get_connection_status(self) -> Dict[str, any]:
        """Get connection status."""
        return {
            'connected': self.connection_manager.is_connected,
            'authenticated': self.connection_manager.is_authenticated,
            'active_symbols': list(self.current_candles.keys()),
            'timeframe_seconds': self.timeframe_seconds
        }
    
    def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from a symbol."""
        with self._lock:
            self.connection_manager.unsubscribe_symbol(symbol)
            
            # Clean up data
            if symbol in self.live_candles:
                del self.live_candles[symbol]
            if symbol in self.current_candles:
                del self.current_candles[symbol]
            if symbol in self.bid_ask_data:
                del self.bid_ask_data[symbol]
            if symbol in self.candle_callbacks:
                del self.candle_callbacks[symbol]
            if symbol in self.bid_ask_callbacks:
                del self.bid_ask_callbacks[symbol]
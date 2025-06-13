import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator for trading signal generation.
    Implements RSI, MACD, Bollinger Bands, Moving Averages, and other key indicators.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_indicators(self, price_data: pd.DataFrame) -> Dict:
        """
        Calculate all technical indicators for the given price data.
        
        Args:
            price_data (pd.DataFrame): OHLCV price data
            
        Returns:
            Dict: Dictionary containing all calculated indicators
        """
        if price_data is None or price_data.empty:
            self.logger.warning("No price data provided for indicator calculation")
            return {}
        
        try:
            indicators = {}
            
            # Moving Averages
            indicators.update(self._calculate_moving_averages(price_data))
            
            # RSI
            indicators['rsi'] = self.calculate_rsi(price_data['close'])
            
            # MACD
            macd_data = self.calculate_macd(price_data['close'])
            indicators.update(macd_data)
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(price_data['close'])
            indicators.update(bb_data)
            
            # Stochastic Oscillator
            stoch_data = self.calculate_stochastic(price_data)
            indicators.update(stoch_data)
            
            # Average True Range (ATR)
            indicators['atr'] = self.calculate_atr(price_data)
            
            # Williams %R
            indicators['williams_r'] = self.calculate_williams_r(price_data)
            
            # Commodity Channel Index (CCI)
            indicators['cci'] = self.calculate_cci(price_data)
            
            # Volume indicators (if volume data available)
            if 'volume' in price_data.columns:
                volume_indicators = self._calculate_volume_indicators(price_data)
                indicators.update(volume_indicators)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}
    
    def calculate_rsi(self, close_prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        if len(close_prices) < period + 1:
            return pd.Series(index=close_prices.index, dtype=float)
        
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, close_prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(close_prices) < slow + signal:
            return {
                'macd_line': pd.Series(index=close_prices.index, dtype=float),
                'macd_signal': pd.Series(index=close_prices.index, dtype=float),
                'macd_histogram': pd.Series(index=close_prices.index, dtype=float)
            }
        
        ema_fast = close_prices.ewm(span=fast).mean()
        ema_slow = close_prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        
        return {
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
    
    def calculate_bollinger_bands(self, close_prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict:
        """Calculate Bollinger Bands."""
        if len(close_prices) < period:
            return {
                'bb_middle': pd.Series(index=close_prices.index, dtype=float),
                'bb_upper': pd.Series(index=close_prices.index, dtype=float),
                'bb_lower': pd.Series(index=close_prices.index, dtype=float),
                'bb_width': pd.Series(index=close_prices.index, dtype=float)
            }
        
        sma = close_prices.rolling(window=period).mean()
        std = close_prices.rolling(window=period).std()
        
        bb_upper = sma + (std * std_dev)
        bb_lower = sma - (std * std_dev)
        bb_width = (bb_upper - bb_lower) / sma * 100
        
        return {
            'bb_middle': sma,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_width': bb_width
        }
    
    def calculate_stochastic(self, price_data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict:
        """Calculate Stochastic Oscillator."""
        if len(price_data) < k_period:
            return {
                'stoch_k': pd.Series(index=price_data.index, dtype=float),
                'stoch_d': pd.Series(index=price_data.index, dtype=float)
            }
        
        high_max = price_data['high'].rolling(window=k_period).max()
        low_min = price_data['low'].rolling(window=k_period).min()
        
        stoch_k = 100 * ((price_data['close'] - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return {
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        }
    
    def calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        if len(price_data) < 2:
            return pd.Series(index=price_data.index, dtype=float)
        
        high_low = price_data['high'] - price_data['low']
        high_close_prev = abs(price_data['high'] - price_data['close'].shift(1))
        low_close_prev = abs(price_data['low'] - price_data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_williams_r(self, price_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        if len(price_data) < period:
            return pd.Series(index=price_data.index, dtype=float)
        
        high_max = price_data['high'].rolling(window=period).max()
        low_min = price_data['low'].rolling(window=period).min()
        
        williams_r = -100 * ((high_max - price_data['close']) / (high_max - low_min))
        
        return williams_r
    
    def calculate_cci(self, price_data: pd.DataFrame, period: int = 20, constant: float = 0.015) -> pd.Series:
        """Calculate Commodity Channel Index (CCI)."""
        if len(price_data) < period:
            return pd.Series(index=price_data.index, dtype=float)
        
        typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma_tp) / (constant * mad)
        
        return cci
    
    def _calculate_moving_averages(self, price_data: pd.DataFrame) -> Dict:
        """Calculate various moving averages."""
        close_prices = price_data['close']
        
        moving_averages = {}
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(close_prices) >= period:
                moving_averages[f'sma_{period}'] = close_prices.rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [5, 10, 20, 50, 100]:
            if len(close_prices) >= period:
                moving_averages[f'ema_{period}'] = close_prices.ewm(span=period).mean()
        
        # Weighted Moving Average
        if len(close_prices) >= 20:
            weights = np.arange(1, 21)
            moving_averages['wma_20'] = close_prices.rolling(window=20).apply(
                lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
            )
        
        return moving_averages
    
    def _calculate_volume_indicators(self, price_data: pd.DataFrame) -> Dict:
        """Calculate volume-based indicators."""
        if 'volume' not in price_data.columns:
            return {}
        
        volume_indicators = {}
        
        # On-Balance Volume (OBV)
        price_change = price_data['close'].diff()
        volume_direction = np.where(price_change > 0, price_data['volume'],
                                  np.where(price_change < 0, -price_data['volume'], 0))
        volume_indicators['obv'] = pd.Series(volume_direction, index=price_data.index).cumsum()
        
        # Volume Rate of Change
        volume_indicators['volume_roc'] = price_data['volume'].pct_change(periods=10) * 100
        
        # Volume Moving Average
        volume_indicators['volume_sma'] = price_data['volume'].rolling(window=20).mean()
        
        # Price Volume Trend (PVT)
        pvt = ((price_data['close'] - price_data['close'].shift(1)) / price_data['close'].shift(1)) * price_data['volume']
        volume_indicators['pvt'] = pvt.cumsum()
        
        return volume_indicators
    
    def calculate_momentum_oscillator(self, close_prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Momentum Oscillator."""
        if len(close_prices) < period + 1:
            return pd.Series(index=close_prices.index, dtype=float)
        
        momentum = close_prices / close_prices.shift(period) * 100
        return momentum
    
    def calculate_rate_of_change(self, close_prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Rate of Change (ROC)."""
        if len(close_prices) < period + 1:
            return pd.Series(index=close_prices.index, dtype=float)
        
        roc = ((close_prices - close_prices.shift(period)) / close_prices.shift(period)) * 100
        return roc

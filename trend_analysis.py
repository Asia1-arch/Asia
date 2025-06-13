import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
import logging

class TrendAnalyzer:
    """
    Advanced trend analysis for identifying market trends and trend lines.
    Implements mathematical algorithms for trend detection, support/resistance,
    and trend strength calculation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_trend_length = 10
        self.trend_tolerance = 0.02  # 2% tolerance for trend validation
    
    def analyze_trends(self, price_data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive trend analysis on price data.
        
        Args:
            price_data (pd.DataFrame): OHLCV price data
            
        Returns:
            Dict: Comprehensive trend analysis results
        """
        if price_data is None or len(price_data) < self.min_trend_length:
            self.logger.warning("Insufficient data for trend analysis")
            return {}
        
        try:
            analysis_results = {}
            
            # Overall trend direction
            overall_trend = self._determine_overall_trend(price_data)
            analysis_results.update(overall_trend)
            
            # Trend lines
            trend_lines = self._detect_trend_lines(price_data)
            analysis_results['trend_lines'] = trend_lines
            
            # Support and resistance levels
            support_resistance = self._calculate_support_resistance(price_data)
            analysis_results.update(support_resistance)
            
            # Trend strength indicators
            trend_strength = self._calculate_trend_strength(price_data)
            analysis_results.update(trend_strength)
            
            # Multi-timeframe trend analysis
            mtf_analysis = self._multi_timeframe_analysis(price_data)
            analysis_results['multi_timeframe'] = mtf_analysis
            
            # Trend reversal signals
            reversal_signals = self._detect_trend_reversals(price_data)
            analysis_results['reversal_signals'] = reversal_signals
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return {}
    
    def _determine_overall_trend(self, price_data: pd.DataFrame) -> Dict:
        """Determine the overall trend direction using multiple methods."""
        close_prices = price_data['close'].values
        
        # Method 1: Linear regression on entire dataset
        x = np.arange(len(close_prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, close_prices)
        
        # Method 2: Moving average comparison
        short_ma = pd.Series(close_prices).rolling(window=min(20, len(close_prices)//4)).mean()
        long_ma = pd.Series(close_prices).rolling(window=min(50, len(close_prices)//2)).mean()
        
        ma_trend = 'uptrend' if short_ma.iloc[-1] > long_ma.iloc[-1] else 'downtrend'
        
        # Method 3: Higher highs and higher lows analysis
        hl_trend = self._analyze_higher_highs_lows(price_data)
        
        # Combine methods for final determination
        slope_trend = 'uptrend' if slope > 0 else 'downtrend'
        
        # Weight the different methods
        trend_votes = [slope_trend, ma_trend, hl_trend]
        uptrend_votes = trend_votes.count('uptrend')
        downtrend_votes = trend_votes.count('downtrend')
        
        if uptrend_votes > downtrend_votes:
            current_trend = 'uptrend'
        elif downtrend_votes > uptrend_votes:
            current_trend = 'downtrend'
        else:
            current_trend = 'sideways'
        
        # Calculate trend strength
        trend_strength = min(abs(r_value), 1.0)  # R-value indicates strength
        
        return {
            'current_trend': current_trend,
            'trend_strength': trend_strength,
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'ma_trend': ma_trend,
            'hl_trend': hl_trend
        }
    
    def _analyze_higher_highs_lows(self, price_data: pd.DataFrame) -> str:
        """Analyze higher highs/higher lows pattern."""
        highs = price_data['high'].values
        lows = price_data['low'].values
        
        # Find recent peaks and troughs
        recent_data_length = min(50, len(price_data))
        recent_highs = highs[-recent_data_length:]
        recent_lows = lows[-recent_data_length:]
        
        # Simple approach: compare first and last quarters
        quarter_length = recent_data_length // 4
        
        first_quarter_high = np.max(recent_highs[:quarter_length])
        last_quarter_high = np.max(recent_highs[-quarter_length:])
        
        first_quarter_low = np.min(recent_lows[:quarter_length])
        last_quarter_low = np.min(recent_lows[-quarter_length:])
        
        higher_highs = last_quarter_high > first_quarter_high
        higher_lows = last_quarter_low > first_quarter_low
        
        if higher_highs and higher_lows:
            return 'uptrend'
        elif last_quarter_high < first_quarter_high and last_quarter_low < first_quarter_low:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _detect_trend_lines(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect significant trend lines in the price data."""
        trend_lines = []
        
        # Detect support trend lines (connecting lows)
        support_lines = self._find_trend_lines(price_data, 'support')
        trend_lines.extend(support_lines)
        
        # Detect resistance trend lines (connecting highs)
        resistance_lines = self._find_trend_lines(price_data, 'resistance')
        trend_lines.extend(resistance_lines)
        
        # Filter and rank trend lines by significance
        significant_lines = [line for line in trend_lines if line['touches'] >= 3 and line['r_squared'] >= 0.7]
        significant_lines.sort(key=lambda x: x['r_squared'], reverse=True)
        
        return significant_lines[:10]  # Return top 10 most significant lines
    
    def _find_trend_lines(self, price_data: pd.DataFrame, line_type: str) -> List[Dict]:
        """Find trend lines of a specific type (support or resistance)."""
        trend_lines = []
        
        # Select appropriate price series
        if line_type == 'support':
            price_series = price_data['low']
        else:  # resistance
            price_series = price_data['high']
        
        # Find local extrema
        extrema_indices = self._find_local_extrema(price_series.values, line_type)
        
        if len(extrema_indices) < 2:
            return trend_lines
        
        # Try different combinations of extrema to find trend lines
        for i in range(len(extrema_indices)):
            for j in range(i + 1, len(extrema_indices)):
                idx1, idx2 = extrema_indices[i], extrema_indices[j]
                
                if idx2 - idx1 < self.min_trend_length:
                    continue
                
                # Calculate trend line
                x1, y1 = idx1, price_series.iloc[idx1]
                x2, y2 = idx2, price_series.iloc[idx2]
                
                # Calculate slope and intercept
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # Check how many points touch this line
                touches, r_squared = self._validate_trend_line(
                    price_series, slope, intercept, idx1, idx2, line_type
                )
                
                if touches >= 2 and r_squared >= 0.5:
                    trend_line = {
                        'type': line_type,
                        'slope': slope,
                        'intercept': intercept,
                        'start_idx': idx1,
                        'end_idx': idx2,
                        'touches': touches,
                        'r_squared': r_squared,
                        'strength': r_squared * touches / 10,  # Normalize strength
                        'start_price': y1,
                        'end_price': y2,
                        'current_value': slope * (len(price_series) - 1) + intercept
                    }
                    trend_lines.append(trend_line)
        
        return trend_lines
    
    def _find_local_extrema(self, prices: np.ndarray, extrema_type: str) -> List[int]:
        """Find local extrema (peaks or troughs) in price data."""
        extrema = []
        window_size = max(3, len(prices) // 20)  # Adaptive window size
        
        for i in range(window_size, len(prices) - window_size):
            is_extremum = False
            
            if extrema_type == 'support':  # Find local minima
                if all(prices[i] <= prices[i-k] for k in range(1, window_size + 1)) and \
                   all(prices[i] <= prices[i+k] for k in range(1, window_size + 1)):
                    is_extremum = True
            else:  # Find local maxima for resistance
                if all(prices[i] >= prices[i-k] for k in range(1, window_size + 1)) and \
                   all(prices[i] >= prices[i+k] for k in range(1, window_size + 1)):
                    is_extremum = True
            
            if is_extremum:
                extrema.append(i)
        
        return extrema
    
    def _validate_trend_line(self, price_series: pd.Series, slope: float, intercept: float, 
                           start_idx: int, end_idx: int, line_type: str) -> Tuple[int, float]:
        """Validate a trend line by counting touches and calculating R-squared."""
        touches = 0
        prices_near_line = []
        
        for i in range(start_idx, end_idx + 1):
            line_value = slope * i + intercept
            price_value = price_series.iloc[i]
            
            # Calculate distance to line
            distance = abs(price_value - line_value) / line_value
            
            # Count as touch if within tolerance
            if distance <= self.trend_tolerance:
                touches += 1
                prices_near_line.append(price_value)
        
        # Calculate R-squared for line fit
        if len(prices_near_line) >= 2:
            x_values = np.arange(len(prices_near_line))
            predicted_values = [slope * (start_idx + i) + intercept for i in x_values]
            
            # Calculate R-squared
            ss_res = np.sum((np.array(prices_near_line) - np.array(predicted_values)) ** 2)
            ss_tot = np.sum((np.array(prices_near_line) - np.mean(prices_near_line)) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r_squared = max(0, min(1, r_squared))  # Clamp between 0 and 1
        else:
            r_squared = 0
        
        return touches, r_squared
    
    def _calculate_support_resistance(self, price_data: pd.DataFrame) -> Dict:
        """Calculate current support and resistance levels."""
        highs = price_data['high'].values
        lows = price_data['low'].values
        closes = price_data['close'].values
        
        # Recent price action (last 20% of data)
        recent_length = max(10, len(price_data) // 5)
        recent_highs = highs[-recent_length:]
        recent_lows = lows[-recent_length:]
        
        # Calculate levels using different methods
        
        # Method 1: Recent swing levels
        swing_resistance = np.max(recent_highs)
        swing_support = np.min(recent_lows)
        
        # Method 2: Psychological levels (round numbers)
        current_price = closes[-1]
        psychological_levels = self._find_psychological_levels(current_price)
        
        # Method 3: Moving average levels
        ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
        ma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price
        
        # Determine which MA acts as support/resistance
        if current_price > ma_20:
            ma_support = ma_20
            ma_resistance = max(ma_50, swing_resistance)
        else:
            ma_support = min(ma_50, swing_support)
            ma_resistance = ma_20
        
        # Combine and validate levels
        support_levels = [swing_support, ma_support] + [level for level in psychological_levels if level < current_price]
        resistance_levels = [swing_resistance, ma_resistance] + [level for level in psychological_levels if level > current_price]
        
        # Select strongest levels
        primary_support = max(support_levels) if support_levels else current_price * 0.95
        primary_resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
        
        return {
            'primary_support': primary_support,
            'primary_resistance': primary_resistance,
            'support_levels': sorted(support_levels, reverse=True)[:3],
            'resistance_levels': sorted(resistance_levels)[:3],
            'psychological_levels': psychological_levels
        }
    
    def _find_psychological_levels(self, current_price: float) -> List[float]:
        """Find psychological support/resistance levels (round numbers)."""
        levels = []
        
        # Determine the appropriate step size based on price magnitude
        if current_price < 1:
            step = 0.1
        elif current_price < 10:
            step = 1
        elif current_price < 100:
            step = 10
        elif current_price < 1000:
            step = 50
        else:
            step = 100
        
        # Find round numbers around current price
        base = int(current_price / step) * step
        
        for i in range(-3, 4):
            level = base + (i * step)
            if level > 0:
                levels.append(level)
        
        return levels
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> Dict:
        """Calculate various trend strength indicators."""
        closes = price_data['close'].values
        
        # ADX-like calculation (simplified)
        high_low_diff = price_data['high'] - price_data['low']
        close_prev_close = abs(price_data['close'] - price_data['close'].shift(1))
        
        true_range = np.maximum(high_low_diff, close_prev_close.fillna(0))
        
        # Directional movement
        plus_dm = np.where(
            (price_data['high'] - price_data['high'].shift(1)) > (price_data['low'].shift(1) - price_data['low']),
            np.maximum(price_data['high'] - price_data['high'].shift(1), 0),
            0
        )
        
        minus_dm = np.where(
            (price_data['low'].shift(1) - price_data['low']) > (price_data['high'] - price_data['high'].shift(1)),
            np.maximum(price_data['low'].shift(1) - price_data['low'], 0),
            0
        )
        
        # Calculate smoothed values
        period = min(14, len(price_data) // 2)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / pd.Series(true_range).rolling(window=period).mean()
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / pd.Series(true_range).rolling(window=period).mean()
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        current_adx = adx.iloc[-1] if not adx.empty and not pd.isna(adx.iloc[-1]) else 25
        
        # Trend consistency
        price_changes = pd.Series(closes).pct_change().dropna()
        positive_changes = len(price_changes[price_changes > 0])
        total_changes = len(price_changes)
        
        trend_consistency = abs(positive_changes / total_changes - 0.5) * 2 if total_changes > 0 else 0
        
        # Volatility-adjusted trend strength
        volatility = price_changes.std() if len(price_changes) > 1 else 0
        returns_magnitude = abs(price_changes.mean()) if len(price_changes) > 0 else 0
        
        volatility_adjusted_strength = returns_magnitude / volatility if volatility > 0 else 0
        
        return {
            'adx': current_adx / 100,  # Normalize to 0-1
            'trend_consistency': trend_consistency,
            'volatility_adjusted_strength': min(volatility_adjusted_strength, 1.0),
            'plus_di': plus_di.iloc[-1] / 100 if not plus_di.empty and not pd.isna(plus_di.iloc[-1]) else 0.5,
            'minus_di': minus_di.iloc[-1] / 100 if not minus_di.empty and not pd.isna(minus_di.iloc[-1]) else 0.5
        }
    
    def _multi_timeframe_analysis(self, price_data: pd.DataFrame) -> Dict:
        """Analyze trends across multiple timeframes."""
        analysis = {}
        
        # Simulate different timeframes by using different periods
        timeframes = {
            'short_term': len(price_data) // 4,  # Recent quarter
            'medium_term': len(price_data) // 2,  # Recent half
            'long_term': len(price_data)  # All data
        }
        
        for tf_name, periods in timeframes.items():
            if periods < 5:
                continue
                
            tf_data = price_data.tail(periods)
            tf_trend = self._determine_overall_trend(tf_data)
            
            analysis[tf_name] = {
                'trend': tf_trend['current_trend'],
                'strength': tf_trend['trend_strength'],
                'slope': tf_trend['slope']
            }
        
        # Determine overall alignment
        trends = [analysis[tf]['trend'] for tf in analysis.keys()]
        if all(trend == 'uptrend' for trend in trends):
            alignment = 'bullish_aligned'
        elif all(trend == 'downtrend' for trend in trends):
            alignment = 'bearish_aligned'
        else:
            alignment = 'mixed'
        
        analysis['alignment'] = alignment
        
        return analysis
    
    def _detect_trend_reversals(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect potential trend reversal signals."""
        reversals = []
        
        if len(price_data) < 20:
            return reversals
        
        closes = price_data['close'].values
        highs = price_data['high'].values
        lows = price_data['low'].values
        
        # Look for divergences and reversal patterns
        recent_periods = min(20, len(price_data))
        
        for i in range(recent_periods, len(price_data)):
            # Check for price exhaustion
            recent_range = highs[i-recent_periods:i+1].max() - lows[i-recent_periods:i+1].min()
            current_range = highs[i] - lows[i]
            
            # Volume spike detection (if volume data available)
            volume_spike = False
            if 'volume' in price_data.columns:
                avg_volume = price_data['volume'].iloc[i-recent_periods:i].mean()
                current_volume = price_data['volume'].iloc[i]
                volume_spike = current_volume > avg_volume * 1.5
            
            # Momentum divergence (simplified)
            price_momentum = closes[i] - closes[i-5] if i >= 5 else 0
            prev_price_momentum = closes[i-5] - closes[i-10] if i >= 10 else 0
            
            momentum_divergence = False
            if price_momentum * prev_price_momentum < 0:  # Opposite signs
                momentum_divergence = True
            
            # Check for reversal conditions
            if (current_range > recent_range * 0.8 and  # Significant range
                (volume_spike or momentum_divergence)):
                
                reversal_type = 'bearish_reversal' if price_momentum > 0 else 'bullish_reversal'
                confidence = 0.6  # Base confidence
                
                if volume_spike:
                    confidence += 0.2
                if momentum_divergence:
                    confidence += 0.2
                
                reversal = {
                    'type': reversal_type,
                    'timestamp': price_data.iloc[i]['timestamp'] if 'timestamp' in price_data.columns else None,
                    'confidence': min(confidence, 1.0),
                    'price': closes[i],
                    'indicators': {
                        'volume_spike': volume_spike,
                        'momentum_divergence': momentum_divergence,
                        'range_expansion': current_range / recent_range
                    }
                }
                reversals.append(reversal)
        
        return reversals[-5:]  # Return last 5 potential reversals

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import logging

class PatternDetector:
    """
    Advanced pattern detection for identifying trading patterns in price data.
    Implements mathematical algorithms for pattern recognition including:
    - Head and Shoulders
    - Double Tops/Bottoms
    - Triangles
    - Support/Resistance levels
    - Candlestick patterns
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_pattern_length = 10
        self.pattern_tolerance = 0.02  # 2% tolerance for pattern matching
    
    def detect_patterns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all trading patterns in the given price data.
        
        Args:
            price_data (pd.DataFrame): OHLCV price data
            
        Returns:
            pd.DataFrame: Detected patterns with confidence scores
        """
        if price_data is None or len(price_data) < self.min_pattern_length:
            self.logger.warning("Insufficient data for pattern detection")
            return pd.DataFrame()
        
        try:
            all_patterns = []
            
            # Detect reversal patterns
            reversal_patterns = self._detect_reversal_patterns(price_data)
            all_patterns.extend(reversal_patterns)
            
            # Detect continuation patterns
            continuation_patterns = self._detect_continuation_patterns(price_data)
            all_patterns.extend(continuation_patterns)
            
            # Detect support and resistance levels
            support_resistance = self._detect_support_resistance(price_data)
            all_patterns.extend(support_resistance)
            
            # Detect candlestick patterns
            candlestick_patterns = self._detect_candlestick_patterns(price_data)
            all_patterns.extend(candlestick_patterns)
            
            # Convert to DataFrame and sort by confidence
            patterns_df = pd.DataFrame(all_patterns)
            if not patterns_df.empty:
                patterns_df = patterns_df.sort_values('confidence', ascending=False)
                patterns_df = patterns_df.reset_index(drop=True)
            
            return patterns_df
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            return pd.DataFrame()
    
    def _detect_reversal_patterns(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect reversal patterns like Head and Shoulders, Double Tops/Bottoms."""
        patterns = []
        
        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(price_data['close'])
        
        # Head and Shoulders detection
        hs_patterns = self._detect_head_shoulders(price_data, peaks, troughs)
        patterns.extend(hs_patterns)
        
        # Double Top/Bottom detection
        double_patterns = self._detect_double_patterns(price_data, peaks, troughs)
        patterns.extend(double_patterns)
        
        # Triple Top/Bottom detection
        triple_patterns = self._detect_triple_patterns(price_data, peaks, troughs)
        patterns.extend(triple_patterns)
        
        return patterns
    
    def _detect_continuation_patterns(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect continuation patterns like triangles, flags, pennants."""
        patterns = []
        
        # Triangle patterns
        triangle_patterns = self._detect_triangles(price_data)
        patterns.extend(triangle_patterns)
        
        # Flag and pennant patterns
        flag_patterns = self._detect_flags_pennants(price_data)
        patterns.extend(flag_patterns)
        
        # Rectangle/channel patterns
        channel_patterns = self._detect_channels(price_data)
        patterns.extend(channel_patterns)
        
        return patterns
    
    def _detect_support_resistance(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect support and resistance levels."""
        patterns = []
        
        # Find significant price levels
        levels = self._find_significant_levels(price_data)
        
        for level in levels:
            pattern = {
                'pattern_type': 'support' if level['type'] == 'support' else 'resistance',
                'timestamp': level['timestamp'],
                'price_level': level['price'],
                'confidence': level['strength'],
                'bullish': level['type'] == 'support',
                'bearish': level['type'] == 'resistance',
                'strength': level['strength'],
                'touches': level['touches']
            }
            patterns.append(pattern)
        
        return patterns
    
    def _detect_candlestick_patterns(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect candlestick patterns."""
        patterns = []
        
        if len(price_data) < 3:
            return patterns
        
        # Doji patterns
        doji_patterns = self._detect_doji(price_data)
        patterns.extend(doji_patterns)
        
        # Hammer and hanging man
        hammer_patterns = self._detect_hammer_hanging_man(price_data)
        patterns.extend(hammer_patterns)
        
        # Engulfing patterns
        engulfing_patterns = self._detect_engulfing(price_data)
        patterns.extend(engulfing_patterns)
        
        # Morning/Evening star
        star_patterns = self._detect_star_patterns(price_data)
        patterns.extend(star_patterns)
        
        return patterns
    
    def _find_peaks_and_troughs(self, price_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks and troughs in price data using signal processing."""
        if len(price_series) < 5:
            return np.array([]), np.array([])
        
        # Find peaks (local maxima)
        peaks, _ = signal.find_peaks(price_series.values, distance=5, prominence=price_series.std() * 0.5)
        
        # Find troughs (local minima) by inverting the series
        troughs, _ = signal.find_peaks(-price_series.values, distance=5, prominence=price_series.std() * 0.5)
        
        return peaks, troughs
    
    def _detect_head_shoulders(self, price_data: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray) -> List[Dict]:
        """Detect Head and Shoulders patterns."""
        patterns = []
        
        if len(peaks) < 3 or len(troughs) < 2:
            return patterns
        
        close_prices = price_data['close'].values
        
        # Look for three consecutive peaks with specific characteristics
        for i in range(len(peaks) - 2):
            left_peak = peaks[i]
            head_peak = peaks[i + 1]
            right_peak = peaks[i + 2]
            
            # Find relevant troughs
            left_trough = None
            right_trough = None
            
            # Find trough between left peak and head
            for trough in troughs:
                if left_peak < trough < head_peak:
                    left_trough = trough
                    break
            
            # Find trough between head and right peak
            for trough in troughs:
                if head_peak < trough < right_peak:
                    right_trough = trough
                    break
            
            if left_trough is None or right_trough is None:
                continue
            
            # Check Head and Shoulders criteria
            left_height = close_prices[left_peak]
            head_height = close_prices[head_peak]
            right_height = close_prices[right_peak]
            
            left_low = close_prices[left_trough]
            right_low = close_prices[right_trough]
            
            # Head should be higher than both shoulders
            if head_height > left_height and head_height > right_height:
                # Shoulders should be approximately equal
                shoulder_diff = abs(left_height - right_height) / max(left_height, right_height)
                
                # Neckline should be approximately level
                neckline_diff = abs(left_low - right_low) / max(left_low, right_low)
                
                if shoulder_diff < self.pattern_tolerance and neckline_diff < self.pattern_tolerance:
                    confidence = 1 - max(shoulder_diff, neckline_diff)
                    
                    pattern = {
                        'pattern_type': 'head_and_shoulders',
                        'timestamp': price_data.iloc[head_peak]['timestamp'] if 'timestamp' in price_data.columns else pd.Timestamp.now(),
                        'confidence': confidence,
                        'bullish': False,
                        'bearish': True,
                        'strength': confidence,
                        'neckline': (left_low + right_low) / 2,
                        'head_price': head_height,
                        'target_price': (left_low + right_low) / 2 - (head_height - (left_low + right_low) / 2)
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_double_patterns(self, price_data: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray) -> List[Dict]:
        """Detect Double Top and Double Bottom patterns."""
        patterns = []
        
        close_prices = price_data['close'].values
        
        # Double Tops
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                for j in range(i + 1, len(peaks)):
                    peak1_price = close_prices[peaks[i]]
                    peak2_price = close_prices[peaks[j]]
                    
                    # Check if peaks are approximately equal
                    price_diff = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
                    
                    if price_diff < self.pattern_tolerance:
                        # Find the trough between peaks
                        trough_between = None
                        min_price = float('inf')
                        
                        for k in range(peaks[i], peaks[j]):
                            if close_prices[k] < min_price:
                                min_price = close_prices[k]
                                trough_between = k
                        
                        if trough_between is not None:
                            confidence = 1 - price_diff
                            
                            pattern = {
                                'pattern_type': 'double_top',
                                'timestamp': price_data.iloc[peaks[j]]['timestamp'] if 'timestamp' in price_data.columns else pd.Timestamp.now(),
                                'confidence': confidence,
                                'bullish': False,
                                'bearish': True,
                                'strength': confidence,
                                'support_level': min_price,
                                'peak_price': (peak1_price + peak2_price) / 2
                            }
                            patterns.append(pattern)
        
        # Double Bottoms
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                for j in range(i + 1, len(troughs)):
                    trough1_price = close_prices[troughs[i]]
                    trough2_price = close_prices[troughs[j]]
                    
                    # Check if troughs are approximately equal
                    price_diff = abs(trough1_price - trough2_price) / max(trough1_price, trough2_price)
                    
                    if price_diff < self.pattern_tolerance:
                        # Find the peak between troughs
                        peak_between = None
                        max_price = 0
                        
                        for k in range(troughs[i], troughs[j]):
                            if close_prices[k] > max_price:
                                max_price = close_prices[k]
                                peak_between = k
                        
                        if peak_between is not None:
                            confidence = 1 - price_diff
                            
                            pattern = {
                                'pattern_type': 'double_bottom',
                                'timestamp': price_data.iloc[troughs[j]]['timestamp'] if 'timestamp' in price_data.columns else pd.Timestamp.now(),
                                'confidence': confidence,
                                'bullish': True,
                                'bearish': False,
                                'strength': confidence,
                                'resistance_level': max_price,
                                'bottom_price': (trough1_price + trough2_price) / 2
                            }
                            patterns.append(pattern)
        
        return patterns
    
    def _detect_triple_patterns(self, price_data: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray) -> List[Dict]:
        """Detect Triple Top and Triple Bottom patterns."""
        patterns = []
        
        close_prices = price_data['close'].values
        
        # Triple Tops
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                peak1_price = close_prices[peaks[i]]
                peak2_price = close_prices[peaks[i + 1]]
                peak3_price = close_prices[peaks[i + 2]]
                
                # Check if all three peaks are approximately equal
                prices = [peak1_price, peak2_price, peak3_price]
                avg_price = np.mean(prices)
                max_diff = max([abs(p - avg_price) / avg_price for p in prices])
                
                if max_diff < self.pattern_tolerance:
                    confidence = 1 - max_diff
                    
                    pattern = {
                        'pattern_type': 'triple_top',
                        'timestamp': price_data.iloc[peaks[i + 2]]['timestamp'] if 'timestamp' in price_data.columns else pd.Timestamp.now(),
                        'confidence': confidence,
                        'bullish': False,
                        'bearish': True,
                        'strength': confidence,
                        'peak_price': avg_price
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_triangles(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect triangle patterns (ascending, descending, symmetrical)."""
        patterns = []
        
        if len(price_data) < 20:
            return patterns
        
        # Use a sliding window to detect triangles
        window_size = 20
        
        for i in range(len(price_data) - window_size):
            window_data = price_data.iloc[i:i + window_size]
            
            # Find trend lines for highs and lows
            high_trend = self._calculate_trend_line(window_data['high'].values)
            low_trend = self._calculate_trend_line(window_data['low'].values)
            
            if high_trend is not None and low_trend is not None:
                high_slope = high_trend['slope']
                low_slope = low_trend['slope']
                
                # Classify triangle type based on slopes
                triangle_type = None
                bullish = None
                bearish = None
                
                if abs(high_slope) < 0.0001 and low_slope > 0.001:  # Ascending triangle
                    triangle_type = 'ascending_triangle'
                    bullish = True
                    bearish = False
                elif abs(low_slope) < 0.0001 and high_slope < -0.001:  # Descending triangle
                    triangle_type = 'descending_triangle'
                    bullish = False
                    bearish = True
                elif high_slope < -0.001 and low_slope > 0.001:  # Symmetrical triangle
                    triangle_type = 'symmetrical_triangle'
                    bullish = None  # Direction depends on breakout
                    bearish = None
                
                if triangle_type:
                    confidence = min(high_trend['r_squared'], low_trend['r_squared'])
                    
                    pattern = {
                        'pattern_type': triangle_type,
                        'timestamp': window_data.iloc[-1]['timestamp'] if 'timestamp' in window_data.columns else pd.Timestamp.now(),
                        'confidence': confidence,
                        'bullish': bullish,
                        'bearish': bearish,
                        'strength': confidence,
                        'upper_trend_slope': high_slope,
                        'lower_trend_slope': low_slope
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_flags_pennants(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect flag and pennant patterns."""
        patterns = []
        
        if len(price_data) < 15:
            return patterns
        
        # Look for strong moves followed by consolidation
        for i in range(10, len(price_data) - 5):
            # Check for strong move (pole)
            pole_start = max(0, i - 10)
            pole_data = price_data.iloc[pole_start:i]
            
            if len(pole_data) < 5:
                continue
            
            price_change = (pole_data.iloc[-1]['close'] - pole_data.iloc[0]['close']) / pole_data.iloc[0]['close']
            
            # Require significant price movement (>3%)
            if abs(price_change) > 0.03:
                # Check for consolidation (flag/pennant)
                consolidation_data = price_data.iloc[i:i + 5]
                
                if len(consolidation_data) >= 5:
                    consolidation_range = (consolidation_data['high'].max() - consolidation_data['low'].min()) / consolidation_data['close'].mean()
                    
                    # Flag/pennant should have narrow range
                    if consolidation_range < 0.02:
                        pattern_type = 'bullish_flag' if price_change > 0 else 'bearish_flag'
                        confidence = min(abs(price_change) * 10, 1.0)  # Higher confidence for stronger moves
                        
                        pattern = {
                            'pattern_type': pattern_type,
                            'timestamp': consolidation_data.iloc[-1]['timestamp'] if 'timestamp' in consolidation_data.columns else pd.Timestamp.now(),
                            'confidence': confidence,
                            'bullish': price_change > 0,
                            'bearish': price_change < 0,
                            'strength': confidence,
                            'pole_size': abs(price_change),
                            'consolidation_range': consolidation_range
                        }
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_channels(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect channel/rectangle patterns."""
        patterns = []
        
        if len(price_data) < 20:
            return patterns
        
        window_size = 20
        
        for i in range(len(price_data) - window_size):
            window_data = price_data.iloc[i:i + window_size]
            
            # Check if price is moving in a channel
            high_values = window_data['high'].values
            low_values = window_data['low'].values
            
            # Calculate support and resistance levels
            resistance_level = np.percentile(high_values, 90)
            support_level = np.percentile(low_values, 10)
            
            # Check how well price respects these levels
            resistance_touches = np.sum(high_values >= resistance_level * 0.995)
            support_touches = np.sum(low_values <= support_level * 1.005)
            
            if resistance_touches >= 2 and support_touches >= 2:
                channel_width = (resistance_level - support_level) / ((resistance_level + support_level) / 2)
                
                # Channel should be reasonably wide but not too wide
                if 0.02 < channel_width < 0.15:
                    confidence = min((resistance_touches + support_touches) / 10, 1.0)
                    
                    pattern = {
                        'pattern_type': 'channel',
                        'timestamp': window_data.iloc[-1]['timestamp'] if 'timestamp' in window_data.columns else pd.Timestamp.now(),
                        'confidence': confidence,
                        'bullish': None,  # Neutral until breakout
                        'bearish': None,
                        'strength': confidence,
                        'resistance_level': resistance_level,
                        'support_level': support_level,
                        'channel_width': channel_width
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _find_significant_levels(self, price_data: pd.DataFrame) -> List[Dict]:
        """Find significant support and resistance levels."""
        levels = []
        
        if len(price_data) < 10:
            return levels
        
        # Combine high and low prices
        all_prices = pd.concat([price_data['high'], price_data['low']])
        
        # Find price levels that appear frequently
        price_counts = {}
        tolerance = all_prices.std() * 0.01  # 1% of standard deviation
        
        for price in all_prices:
            found_level = False
            for level_price in price_counts:
                if abs(price - level_price) <= tolerance:
                    price_counts[level_price] += 1
                    found_level = True
                    break
            
            if not found_level:
                price_counts[price] = 1
        
        # Filter levels with multiple touches
        for level_price, count in price_counts.items():
            if count >= 3:  # At least 3 touches
                # Determine if it's support or resistance
                recent_prices = price_data['close'].tail(10).values
                avg_recent_price = np.mean(recent_prices)
                
                level_type = 'support' if level_price < avg_recent_price else 'resistance'
                strength = min(count / 10, 1.0)  # Normalize strength
                
                level = {
                    'price': level_price,
                    'type': level_type,
                    'strength': strength,
                    'touches': count,
                    'timestamp': price_data.iloc[-1]['timestamp'] if 'timestamp' in price_data.columns else pd.Timestamp.now()
                }
                levels.append(level)
        
        return levels
    
    def _detect_doji(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect Doji candlestick patterns."""
        patterns = []
        
        for i in range(len(price_data)):
            row = price_data.iloc[i]
            
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # Doji: open and close are very close
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range > 0 and body_size / total_range < 0.1:  # Body is <10% of total range
                confidence = 1 - (body_size / total_range) * 10
                
                pattern = {
                    'pattern_type': 'doji',
                    'timestamp': row['timestamp'] if 'timestamp' in price_data.columns else pd.Timestamp.now(),
                    'confidence': confidence,
                    'bullish': None,  # Doji is neutral, depends on context
                    'bearish': None,
                    'strength': confidence
                }
                patterns.append(pattern)
        
        return patterns
    
    def _detect_hammer_hanging_man(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect Hammer and Hanging Man patterns."""
        patterns = []
        
        for i in range(len(price_data)):
            row = price_data.iloc[i]
            
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            total_range = high_price - low_price
            
            # Hammer/Hanging man: small body, long lower shadow, small upper shadow
            if (total_range > 0 and 
                body_size / total_range < 0.3 and  # Small body
                lower_shadow / total_range > 0.6 and  # Long lower shadow
                upper_shadow / total_range < 0.1):  # Small upper shadow
                
                # Determine trend context to classify as hammer or hanging man
                # For simplicity, we'll call it hammer (bullish reversal)
                confidence = (lower_shadow / total_range) * (1 - body_size / total_range)
                
                pattern = {
                    'pattern_type': 'hammer',
                    'timestamp': row['timestamp'] if 'timestamp' in price_data.columns else pd.Timestamp.now(),
                    'confidence': confidence,
                    'bullish': True,
                    'bearish': False,
                    'strength': confidence
                }
                patterns.append(pattern)
        
        return patterns
    
    def _detect_engulfing(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect Bullish and Bearish Engulfing patterns."""
        patterns = []
        
        for i in range(1, len(price_data)):
            prev_row = price_data.iloc[i - 1]
            curr_row = price_data.iloc[i]
            
            prev_open = prev_row['open']
            prev_close = prev_row['close']
            curr_open = curr_row['open']
            curr_close = curr_row['close']
            
            prev_body_size = abs(prev_close - prev_open)
            curr_body_size = abs(curr_close - curr_open)
            
            # Engulfing: current candle's body completely engulfs previous candle's body
            if (curr_body_size > prev_body_size and
                max(curr_open, curr_close) > max(prev_open, prev_close) and
                min(curr_open, curr_close) < min(prev_open, prev_close)):
                
                # Bullish engulfing: previous red, current green
                if prev_close < prev_open and curr_close > curr_open:
                    confidence = (curr_body_size / prev_body_size) * 0.8
                    
                    pattern = {
                        'pattern_type': 'bullish_engulfing',
                        'timestamp': curr_row['timestamp'] if 'timestamp' in price_data.columns else pd.Timestamp.now(),
                        'confidence': min(confidence, 1.0),
                        'bullish': True,
                        'bearish': False,
                        'strength': min(confidence, 1.0)
                    }
                    patterns.append(pattern)
                
                # Bearish engulfing: previous green, current red
                elif prev_close > prev_open and curr_close < curr_open:
                    confidence = (curr_body_size / prev_body_size) * 0.8
                    
                    pattern = {
                        'pattern_type': 'bearish_engulfing',
                        'timestamp': curr_row['timestamp'] if 'timestamp' in price_data.columns else pd.Timestamp.now(),
                        'confidence': min(confidence, 1.0),
                        'bullish': False,
                        'bearish': True,
                        'strength': min(confidence, 1.0)
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_star_patterns(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect Morning Star and Evening Star patterns."""
        patterns = []
        
        for i in range(2, len(price_data)):
            first_candle = price_data.iloc[i - 2]
            star_candle = price_data.iloc[i - 1]
            third_candle = price_data.iloc[i]
            
            # Morning Star: bearish candle, small star (gap down), bullish candle (gap up)
            if (first_candle['close'] < first_candle['open'] and  # First candle bearish
                abs(star_candle['close'] - star_candle['open']) < abs(first_candle['close'] - first_candle['open']) * 0.3 and  # Star small
                third_candle['close'] > third_candle['open'] and  # Third candle bullish
                third_candle['close'] > (first_candle['open'] + first_candle['close']) / 2):  # Recovery
                
                confidence = 0.7  # Base confidence for star patterns
                
                pattern = {
                    'pattern_type': 'morning_star',
                    'timestamp': third_candle['timestamp'] if 'timestamp' in price_data.columns else pd.Timestamp.now(),
                    'confidence': confidence,
                    'bullish': True,
                    'bearish': False,
                    'strength': confidence
                }
                patterns.append(pattern)
            
            # Evening Star: bullish candle, small star (gap up), bearish candle (gap down)
            elif (first_candle['close'] > first_candle['open'] and  # First candle bullish
                  abs(star_candle['close'] - star_candle['open']) < abs(first_candle['close'] - first_candle['open']) * 0.3 and  # Star small
                  third_candle['close'] < third_candle['open'] and  # Third candle bearish
                  third_candle['close'] < (first_candle['open'] + first_candle['close']) / 2):  # Decline
                
                confidence = 0.7  # Base confidence for star patterns
                
                pattern = {
                    'pattern_type': 'evening_star',
                    'timestamp': third_candle['timestamp'] if 'timestamp' in price_data.columns else pd.Timestamp.now(),
                    'confidence': confidence,
                    'bullish': False,
                    'bearish': True,
                    'strength': confidence
                }
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_trend_line(self, price_values: np.ndarray) -> Optional[Dict]:
        """Calculate trend line for given price values."""
        if len(price_values) < 2:
            return None
        
        x = np.arange(len(price_values))
        
        # Linear regression
        coefficients = np.polyfit(x, price_values, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((price_values - y_pred) ** 2)
        ss_tot = np.sum((price_values - np.mean(price_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': max(0, r_squared)  # Ensure non-negative
        }

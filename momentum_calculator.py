import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class MomentumCalculator:
    """
    Advanced momentum calculation module for trading signal generation.
    Implements various momentum indicators and oscillators for market analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_momentum_indicators(self, price_data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive momentum indicators.
        
        Args:
            price_data (pd.DataFrame): OHLCV price data
            
        Returns:
            Dict: Dictionary containing all momentum indicators
        """
        if price_data is None or price_data.empty:
            self.logger.warning("No price data provided for momentum calculation")
            return self._empty_momentum_data()
        
        try:
            momentum_data = {}
            
            # Basic momentum calculations
            basic_momentum = self._calculate_basic_momentum(price_data)
            momentum_data.update(basic_momentum)
            
            # Rate of Change (ROC) indicators
            roc_indicators = self._calculate_roc_indicators(price_data)
            momentum_data.update(roc_indicators)
            
            # Momentum oscillators
            oscillators = self._calculate_momentum_oscillators(price_data)
            momentum_data.update(oscillators)
            
            # Acceleration and velocity
            acceleration_data = self._calculate_acceleration(price_data)
            momentum_data.update(acceleration_data)
            
            # Volume-based momentum (if volume available)
            if 'volume' in price_data.columns:
                volume_momentum = self._calculate_volume_momentum(price_data)
                momentum_data.update(volume_momentum)
            
            # Composite momentum score
            composite_score = self._calculate_composite_momentum(momentum_data)
            momentum_data.update(composite_score)
            
            return momentum_data
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {str(e)}")
            return self._empty_momentum_data()
    
    def _calculate_basic_momentum(self, price_data: pd.DataFrame) -> Dict:
        """Calculate basic momentum indicators."""
        close_prices = price_data['close']
        momentum_data = {}
        
        # Simple momentum (current price vs N periods ago)
        periods = [3, 5, 10, 20]
        
        for period in periods:
            if len(close_prices) > period:
                momentum = close_prices / close_prices.shift(period) - 1
                momentum_data[f'momentum_{period}'] = momentum.iloc[-1] if not momentum.empty else 0
            else:
                momentum_data[f'momentum_{period}'] = 0
        
        # Current momentum (most recent)
        if len(close_prices) >= 5:
            momentum_data['current_momentum'] = momentum_data.get('momentum_5', 0)
        else:
            momentum_data['current_momentum'] = 0
        
        # Momentum change (momentum of momentum)
        if len(close_prices) >= 10:
            recent_momentum = momentum_data.get('momentum_5', 0)
            prev_momentum_5 = close_prices.iloc[-6] / close_prices.iloc[-11] - 1 if len(close_prices) >= 11 else 0
            momentum_data['momentum_change'] = recent_momentum - prev_momentum_5
        else:
            momentum_data['momentum_change'] = 0
        
        return momentum_data
    
    def _calculate_roc_indicators(self, price_data: pd.DataFrame) -> Dict:
        """Calculate Rate of Change indicators."""
        close_prices = price_data['close']
        roc_data = {}
        
        # Standard ROC periods
        roc_periods = [5, 10, 15, 20]
        
        for period in roc_periods:
            if len(close_prices) > period:
                roc = ((close_prices - close_prices.shift(period)) / close_prices.shift(period)) * 100
                roc_data[f'roc_{period}'] = roc.iloc[-1] if not roc.empty else 0
                
                # ROC moving average for smoothing
                roc_ma = roc.rolling(window=min(5, len(roc))).mean()
                roc_data[f'roc_{period}_ma'] = roc_ma.iloc[-1] if not roc_ma.empty else 0
            else:
                roc_data[f'roc_{period}'] = 0
                roc_data[f'roc_{period}_ma'] = 0
        
        # ROC trend (is ROC increasing or decreasing)
        if len(close_prices) >= 15:
            recent_roc = roc_data.get('roc_10', 0)
            prev_roc_data = ((close_prices.iloc[-6] - close_prices.iloc[-16]) / close_prices.iloc[-16]) * 100
            roc_data['roc_trend'] = 1 if recent_roc > prev_roc_data else -1
        else:
            roc_data['roc_trend'] = 0
        
        return roc_data
    
    def _calculate_momentum_oscillators(self, price_data: pd.DataFrame) -> Dict:
        """Calculate momentum oscillators."""
        oscillator_data = {}
        
        # Price Momentum Oscillator (PMO)
        pmo_data = self._calculate_pmo(price_data)
        oscillator_data.update(pmo_data)
        
        # Momentum Index
        momentum_index = self._calculate_momentum_index(price_data)
        oscillator_data.update(momentum_index)
        
        # Chande Momentum Oscillator
        cmo_data = self._calculate_cmo(price_data)
        oscillator_data.update(cmo_data)
        
        return oscillator_data
    
    def _calculate_pmo(self, price_data: pd.DataFrame, period1: int = 35, period2: int = 20) -> Dict:
        """Calculate Price Momentum Oscillator."""
        close_prices = price_data['close']
        
        if len(close_prices) < max(period1, period2) + 5:
            return {'pmo': 0, 'pmo_signal': 0}
        
        # Calculate rate of change
        roc = close_prices.pct_change() * 100
        
        # Double smoothing with EMA
        ema1 = roc.ewm(span=period1).mean()
        pmo = ema1.ewm(span=period2).mean()
        
        # PMO signal line
        pmo_signal = pmo.ewm(span=10).mean()
        
        return {
            'pmo': pmo.iloc[-1] if not pmo.empty else 0,
            'pmo_signal': pmo_signal.iloc[-1] if not pmo_signal.empty else 0
        }
    
    def _calculate_momentum_index(self, price_data: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate Momentum Index."""
        if len(price_data) < period + 5:
            return {'momentum_index': 50}
        
        close_prices = price_data['close']
        
        # Calculate momentum
        momentum = close_prices - close_prices.shift(period)
        
        # Positive and negative momentum sums
        positive_momentum = momentum.where(momentum > 0, 0).rolling(window=period).sum()
        negative_momentum = abs(momentum.where(momentum < 0, 0)).rolling(window=period).sum()
        
        # Momentum Index calculation
        momentum_index = 100 - (100 / (1 + positive_momentum / negative_momentum))
        
        return {
            'momentum_index': momentum_index.iloc[-1] if not momentum_index.empty and not pd.isna(momentum_index.iloc[-1]) else 50
        }
    
    def _calculate_cmo(self, price_data: pd.DataFrame, period: int = 14) -> Dict:
        """Calculate Chande Momentum Oscillator."""
        close_prices = price_data['close']
        
        if len(close_prices) < period + 1:
            return {'cmo': 0}
        
        # Price changes
        price_changes = close_prices.diff()
        
        # Positive and negative changes
        positive_changes = price_changes.where(price_changes > 0, 0)
        negative_changes = abs(price_changes.where(price_changes < 0, 0))
        
        # Sum over period
        positive_sum = positive_changes.rolling(window=period).sum()
        negative_sum = negative_changes.rolling(window=period).sum()
        
        # CMO calculation
        cmo = 100 * ((positive_sum - negative_sum) / (positive_sum + negative_sum))
        
        return {
            'cmo': cmo.iloc[-1] if not cmo.empty and not pd.isna(cmo.iloc[-1]) else 0
        }
    
    def _calculate_acceleration(self, price_data: pd.DataFrame) -> Dict:
        """Calculate price acceleration and velocity metrics."""
        close_prices = price_data['close']
        acceleration_data = {}
        
        if len(close_prices) < 10:
            return {
                'velocity': 0,
                'acceleration': 0,
                'velocity_ma': 0,
                'acceleration_ma': 0
            }
        
        # Velocity (first derivative of price)
        velocity = close_prices.diff()
        acceleration_data['velocity'] = velocity.iloc[-1] if not velocity.empty else 0
        
        # Acceleration (second derivative of price)
        acceleration = velocity.diff()
        acceleration_data['acceleration'] = acceleration.iloc[-1] if not acceleration.empty else 0
        
        # Smoothed velocity and acceleration
        velocity_ma = velocity.rolling(window=min(5, len(velocity))).mean()
        acceleration_ma = acceleration.rolling(window=min(5, len(acceleration))).mean()
        
        acceleration_data['velocity_ma'] = velocity_ma.iloc[-1] if not velocity_ma.empty else 0
        acceleration_data['acceleration_ma'] = acceleration_ma.iloc[-1] if not acceleration_ma.empty else 0
        
        # Jerk (third derivative) for advanced analysis
        if len(close_prices) >= 15:
            jerk = acceleration.diff()
            acceleration_data['jerk'] = jerk.iloc[-1] if not jerk.empty else 0
        else:
            acceleration_data['jerk'] = 0
        
        return acceleration_data
    
    def _calculate_volume_momentum(self, price_data: pd.DataFrame) -> Dict:
        """Calculate volume-based momentum indicators."""
        if 'volume' not in price_data.columns:
            return {}
        
        volume_momentum_data = {}
        close_prices = price_data['close']
        volume = price_data['volume']
        
        # Volume Rate of Change
        volume_roc = volume.pct_change(periods=5) * 100
        volume_momentum_data['volume_roc'] = volume_roc.iloc[-1] if not volume_roc.empty else 0
        
        # Price-Volume Momentum
        price_change = close_prices.pct_change()
        volume_change = volume.pct_change()
        
        # Combine price and volume momentum
        if len(price_change) > 0 and len(volume_change) > 0:
            pv_momentum = price_change * volume_change
            volume_momentum_data['price_volume_momentum'] = pv_momentum.iloc[-1] if not pv_momentum.empty else 0
        else:
            volume_momentum_data['price_volume_momentum'] = 0
        
        # Volume-weighted momentum
        if len(close_prices) >= 10:
            momentum_5 = (close_prices / close_prices.shift(5) - 1) * 100
            volume_weight = volume / volume.rolling(window=min(10, len(volume))).mean()
            
            vw_momentum = momentum_5 * volume_weight
            volume_momentum_data['volume_weighted_momentum'] = vw_momentum.iloc[-1] if not vw_momentum.empty else 0
        else:
            volume_momentum_data['volume_weighted_momentum'] = 0
        
        return volume_momentum_data
    
    def _calculate_composite_momentum(self, momentum_data: Dict) -> Dict:
        """Calculate composite momentum score from all indicators."""
        composite_data = {}
        
        # Collect all momentum values for scoring
        momentum_values = []
        
        # Basic momentum indicators
        for period in [3, 5, 10, 20]:
            key = f'momentum_{period}'
            if key in momentum_data:
                momentum_values.append(momentum_data[key])
        
        # ROC indicators
        for period in [5, 10, 15, 20]:
            key = f'roc_{period}'
            if key in momentum_data:
                momentum_values.append(momentum_data[key] / 100)  # Convert percentage to decimal
        
        # Oscillator values
        oscillator_keys = ['pmo', 'momentum_index', 'cmo']
        for key in oscillator_keys:
            if key in momentum_data:
                value = momentum_data[key]
                if key == 'momentum_index':
                    value = (value - 50) / 50  # Normalize to -1 to 1
                elif key == 'cmo':
                    value = value / 100  # Normalize to -1 to 1
                momentum_values.append(value)
        
        # Calculate composite score
        if momentum_values:
            # Remove outliers
            momentum_array = np.array(momentum_values)
            q75, q25 = np.percentile(momentum_array, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            filtered_values = momentum_array[(momentum_array >= lower_bound) & (momentum_array <= upper_bound)]
            
            if len(filtered_values) > 0:
                composite_score = np.mean(filtered_values)
            else:
                composite_score = np.mean(momentum_array)
        else:
            composite_score = 0
        
        composite_data['composite_momentum'] = composite_score
        
        # Momentum strength (absolute value)
        composite_data['momentum_strength'] = abs(composite_score)
        
        # Momentum direction
        if composite_score > 0.02:
            composite_data['momentum_direction'] = 'bullish'
        elif composite_score < -0.02:
            composite_data['momentum_direction'] = 'bearish'
        else:
            composite_data['momentum_direction'] = 'neutral'
        
        # Momentum quality score (consistency across indicators)
        if momentum_values:
            positive_count = sum(1 for v in momentum_values if v > 0)
            total_count = len(momentum_values)
            
            if positive_count / total_count > 0.7:
                quality_score = 0.8 + (positive_count / total_count - 0.7) * 0.67
            elif positive_count / total_count < 0.3:
                quality_score = 0.8 + (0.3 - positive_count / total_count) * 0.67
            else:
                quality_score = 0.4  # Mixed signals
            
            composite_data['momentum_quality'] = quality_score
        else:
            composite_data['momentum_quality'] = 0.5
        
        return composite_data
    
    def get_momentum_signals(self, momentum_data: Dict) -> List[Dict]:
        """Generate trading signals based on momentum analysis."""
        signals = []
        
        if not momentum_data:
            return signals
        
        try:
            # Current momentum values
            current_momentum = momentum_data.get('current_momentum', 0)
            momentum_change = momentum_data.get('momentum_change', 0)
            composite_momentum = momentum_data.get('composite_momentum', 0)
            momentum_quality = momentum_data.get('momentum_quality', 0.5)
            
            # Signal generation logic
            
            # Strong bullish momentum signal
            if (current_momentum > 0.03 and 
                momentum_change > 0 and 
                composite_momentum > 0.02 and 
                momentum_quality > 0.6):
                
                confidence = min(95, 70 + momentum_quality * 25)
                signals.append({
                    'signal_type': 'BUY',
                    'confidence': confidence,
                    'source': 'momentum',
                    'reasoning': 'Strong bullish momentum detected',
                    'indicators': {
                        'current_momentum': current_momentum,
                        'momentum_change': momentum_change,
                        'composite_momentum': composite_momentum
                    }
                })
            
            # Strong bearish momentum signal
            elif (current_momentum < -0.03 and 
                  momentum_change < 0 and 
                  composite_momentum < -0.02 and 
                  momentum_quality > 0.6):
                
                confidence = min(95, 70 + momentum_quality * 25)
                signals.append({
                    'signal_type': 'SELL',
                    'confidence': confidence,
                    'source': 'momentum',
                    'reasoning': 'Strong bearish momentum detected',
                    'indicators': {
                        'current_momentum': current_momentum,
                        'momentum_change': momentum_change,
                        'composite_momentum': composite_momentum
                    }
                })
            
            # Momentum reversal signals
            elif (abs(current_momentum) > 0.05 and 
                  momentum_change * current_momentum < 0):  # Momentum changing direction
                
                signal_type = 'SELL' if current_momentum > 0 else 'BUY'
                confidence = min(80, 50 + abs(momentum_change) * 300)
                
                signals.append({
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'source': 'momentum_reversal',
                    'reasoning': 'Momentum reversal detected',
                    'indicators': {
                        'current_momentum': current_momentum,
                        'momentum_change': momentum_change
                    }
                })
            
            # Momentum divergence signals (if oscillators available)
            pmo = momentum_data.get('pmo', 0)
            pmo_signal = momentum_data.get('pmo_signal', 0)
            
            if pmo != 0 and pmo_signal != 0:
                if pmo > pmo_signal and current_momentum > 0:
                    signals.append({
                        'signal_type': 'BUY',
                        'confidence': 65,
                        'source': 'pmo_crossover',
                        'reasoning': 'PMO bullish crossover',
                        'indicators': {'pmo': pmo, 'pmo_signal': pmo_signal}
                    })
                elif pmo < pmo_signal and current_momentum < 0:
                    signals.append({
                        'signal_type': 'SELL',
                        'confidence': 65,
                        'source': 'pmo_crossover',
                        'reasoning': 'PMO bearish crossover',
                        'indicators': {'pmo': pmo, 'pmo_signal': pmo_signal}
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating momentum signals: {str(e)}")
            return []
    
    def _empty_momentum_data(self) -> Dict:
        """Return empty momentum data structure."""
        return {
            'current_momentum': 0,
            'momentum_change': 0,
            'acceleration': 0,
            'velocity': 0,
            'composite_momentum': 0,
            'momentum_strength': 0,
            'momentum_direction': 'neutral',
            'momentum_quality': 0.5
        }
    
    def analyze_momentum_divergence(self, price_data: pd.DataFrame, momentum_data: Dict) -> Dict:
        """Analyze momentum divergence patterns."""
        if price_data is None or price_data.empty or not momentum_data:
            return {'divergence_detected': False}
        
        try:
            divergence_analysis = {'divergence_detected': False}
            
            close_prices = price_data['close']
            
            if len(close_prices) < 20:
                return divergence_analysis
            
            # Analyze recent price peaks and troughs
            recent_data = close_prices.tail(20)
            
            # Find local peaks and troughs
            peaks = []
            troughs = []
            
            for i in range(2, len(recent_data) - 2):
                if (recent_data.iloc[i] > recent_data.iloc[i-1] and 
                    recent_data.iloc[i] > recent_data.iloc[i+1] and
                    recent_data.iloc[i] > recent_data.iloc[i-2] and 
                    recent_data.iloc[i] > recent_data.iloc[i+2]):
                    peaks.append((i, recent_data.iloc[i]))
                
                if (recent_data.iloc[i] < recent_data.iloc[i-1] and 
                    recent_data.iloc[i] < recent_data.iloc[i+1] and
                    recent_data.iloc[i] < recent_data.iloc[i-2] and 
                    recent_data.iloc[i] < recent_data.iloc[i+2]):
                    troughs.append((i, recent_data.iloc[i]))
            
            # Check for divergence patterns
            current_momentum = momentum_data.get('current_momentum', 0)
            
            # Bullish divergence: price making lower lows, momentum making higher lows
            if len(troughs) >= 2:
                price_trend = troughs[-1][1] - troughs[-2][1]  # Latest trough vs previous
                if price_trend < 0 and current_momentum > -0.02:  # Price declining but momentum not as weak
                    divergence_analysis.update({
                        'divergence_detected': True,
                        'divergence_type': 'bullish',
                        'strength': min(abs(price_trend) * 100, 1.0),
                        'signal': 'potential_reversal_up'
                    })
            
            # Bearish divergence: price making higher highs, momentum making lower highs
            if len(peaks) >= 2:
                price_trend = peaks[-1][1] - peaks[-2][1]  # Latest peak vs previous
                if price_trend > 0 and current_momentum < 0.02:  # Price rising but momentum weakening
                    divergence_analysis.update({
                        'divergence_detected': True,
                        'divergence_type': 'bearish',
                        'strength': min(price_trend * 100, 1.0),
                        'signal': 'potential_reversal_down'
                    })
            
            return divergence_analysis
            
        except Exception as e:
            self.logger.error(f"Error in momentum divergence analysis: {str(e)}")
            return {'divergence_detected': False}

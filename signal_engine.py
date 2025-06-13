import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from technical_indicators import TechnicalIndicators
from pattern_detection import PatternDetector
from trend_analysis import TrendAnalyzer
from statistical_validation import StatisticalValidator
from momentum_calculator import MomentumCalculator
from data_processor import DataProcessor

class SignalEngine:
    """
    Advanced signal generation engine for synthetic indices trading.
    Combines multiple mathematical algorithms for accurate pattern detection.
    """
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        self.pattern_detector = PatternDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.statistical_validator = StatisticalValidator()
        self.momentum_calculator = MomentumCalculator()
        self.data_processor = DataProcessor()
        
        # Signal generation parameters
        self.signal_weights = {
            'technical': 0.30,
            'pattern': 0.25,
            'trend': 0.20,
            'momentum': 0.15,
            'statistical': 0.10
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_signals(self, symbol, timeframe, lookback_periods=200, confidence_threshold=75):
        """
        Generate comprehensive trading signals for the specified symbol and timeframe.
        
        Args:
            symbol (str): Trading symbol (e.g., 'crash300', 'boom1000')
            timeframe (str): Analysis timeframe ('1m', '5m', '15m', '1h')
            lookback_periods (int): Number of periods to analyze
            confidence_threshold (float): Minimum confidence level for signals
            
        Returns:
            dict: Comprehensive signal data including price data, signals, indicators, patterns
        """
        try:
            # Generate synthetic market data (in production, replace with real data feed)
            price_data = self.data_processor.get_market_data(symbol, timeframe, lookback_periods)
            
            if price_data is None or price_data.empty:
                self.logger.warning(f"No price data available for {symbol} on {timeframe}")
                return None
            
            # Calculate technical indicators
            indicators = self.technical_indicators.calculate_all_indicators(price_data)
            
            # Detect patterns
            patterns = self.pattern_detector.detect_patterns(price_data)
            
            # Analyze trends
            trend_analysis = self.trend_analyzer.analyze_trends(price_data)
            
            # Calculate momentum
            momentum_data = self.momentum_calculator.calculate_momentum_indicators(price_data)
            
            # Generate individual signal components
            technical_signals = self._generate_technical_signals(price_data, indicators)
            pattern_signals = self._generate_pattern_signals(patterns)
            trend_signals = self._generate_trend_signals(trend_analysis)
            momentum_signals = self._generate_momentum_signals(momentum_data)
            
            # Combine signals with weighted scoring
            combined_signals = self._combine_signals(
                technical_signals, pattern_signals, trend_signals, momentum_signals
            )
            
            # Statistical validation
            validated_signals = self.statistical_validator.validate_signals(
                combined_signals, price_data, confidence_threshold
            )
            
            # Calculate signal statistics
            statistics = self._calculate_signal_statistics(validated_signals, price_data)
            
            return {
                'price_data': price_data,
                'signals': validated_signals,
                'indicators': indicators,
                'patterns': patterns,
                'trend_lines': trend_analysis.get('trend_lines', []),
                'momentum': momentum_data,
                'statistics': statistics,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return None
    
    def _generate_technical_signals(self, price_data, indicators):
        """Generate signals based on technical indicators."""
        signals = []
        
        if indicators is None or price_data.empty:
            return pd.DataFrame(signals)
        
        for i in range(len(price_data)):
            if i < 20:  # Need sufficient data for analysis
                continue
                
            timestamp = price_data.iloc[i]['timestamp'] if 'timestamp' in price_data.columns else datetime.now()
            close_price = price_data.iloc[i]['close']
            
            # RSI signals
            rsi = indicators.get('rsi', pd.Series()).iloc[i] if i < len(indicators.get('rsi', [])) else 50
            rsi_signal = 'BUY' if rsi < 30 else ('SELL' if rsi > 70 else 'HOLD')
            rsi_strength = abs(50 - rsi) / 50
            
            # MACD signals
            macd_line = indicators.get('macd_line', pd.Series()).iloc[i] if i < len(indicators.get('macd_line', [])) else 0
            macd_signal_line = indicators.get('macd_signal', pd.Series()).iloc[i] if i < len(indicators.get('macd_signal', [])) else 0
            macd_signal = 'BUY' if macd_line > macd_signal_line else 'SELL'
            macd_strength = abs(macd_line - macd_signal_line) / max(abs(macd_line), abs(macd_signal_line), 0.001)
            
            # Moving average signals
            sma_20 = indicators.get('sma_20', pd.Series()).iloc[i] if i < len(indicators.get('sma_20', [])) else close_price
            sma_50 = indicators.get('sma_50', pd.Series()).iloc[i] if i < len(indicators.get('sma_50', [])) else close_price
            ma_signal = 'BUY' if sma_20 > sma_50 else 'SELL'
            ma_strength = abs(sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
            
            # Bollinger Bands signals
            bb_upper = indicators.get('bb_upper', pd.Series()).iloc[i] if i < len(indicators.get('bb_upper', [])) else close_price * 1.02
            bb_lower = indicators.get('bb_lower', pd.Series()).iloc[i] if i < len(indicators.get('bb_lower', [])) else close_price * 0.98
            bb_signal = 'SELL' if close_price > bb_upper else ('BUY' if close_price < bb_lower else 'HOLD')
            bb_strength = max(0, min(1, abs(close_price - (bb_upper + bb_lower) / 2) / ((bb_upper - bb_lower) / 2)))
            
            # Combine technical signals
            signal_scores = {
                'BUY': 0,
                'SELL': 0,
                'HOLD': 0
            }
            
            # Weight individual signals
            weights = {'rsi': 0.3, 'macd': 0.3, 'ma': 0.25, 'bb': 0.15}
            strengths = {'rsi': rsi_strength, 'macd': macd_strength, 'ma': ma_strength, 'bb': bb_strength}
            individual_signals = {'rsi': rsi_signal, 'macd': macd_signal, 'ma': ma_signal, 'bb': bb_signal}
            
            for indicator, signal in individual_signals.items():
                signal_scores[signal] += weights[indicator] * strengths[indicator]
            
            # Determine final technical signal
            final_signal = max(signal_scores, key=signal_scores.get)
            confidence = signal_scores[final_signal] * 100
            
            if confidence >= 20:  # Minimum threshold for technical signals
                signals.append({
                    'timestamp': timestamp,
                    'signal_type': final_signal,
                    'confidence': confidence,
                    'signal_source': 'technical',
                    'details': {
                        'rsi': rsi,
                        'macd_line': macd_line,
                        'macd_signal': macd_signal_line,
                        'price': close_price,
                        'sma_20': sma_20,
                        'sma_50': sma_50
                    }
                })
        
        return pd.DataFrame(signals)
    
    def _generate_pattern_signals(self, patterns):
        """Generate signals based on detected patterns."""
        signals = []
        
        if patterns is None or patterns.empty:
            return pd.DataFrame(signals)
        
        for _, pattern in patterns.iterrows():
            signal_type = 'BUY' if pattern.get('bullish', False) else ('SELL' if pattern.get('bearish', False) else 'HOLD')
            confidence = pattern.get('confidence', 0) * 100
            
            if confidence >= 30:  # Minimum threshold for pattern signals
                signals.append({
                    'timestamp': pattern.get('timestamp', datetime.now()),
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'signal_source': 'pattern',
                    'details': {
                        'pattern_type': pattern.get('pattern_type', 'unknown'),
                        'strength': pattern.get('strength', 0)
                    }
                })
        
        return pd.DataFrame(signals)
    
    def _generate_trend_signals(self, trend_analysis):
        """Generate signals based on trend analysis."""
        signals = []
        
        if not trend_analysis or 'current_trend' not in trend_analysis:
            return pd.DataFrame(signals)
        
        current_trend = trend_analysis['current_trend']
        trend_strength = trend_analysis.get('trend_strength', 0)
        
        if trend_strength >= 0.3:  # Minimum trend strength
            signal_type = 'BUY' if current_trend == 'uptrend' else ('SELL' if current_trend == 'downtrend' else 'HOLD')
            confidence = trend_strength * 100
            
            signals.append({
                'timestamp': datetime.now(),
                'signal_type': signal_type,
                'confidence': confidence,
                'signal_source': 'trend',
                'details': {
                    'trend_type': current_trend,
                    'strength': trend_strength,
                    'slope': trend_analysis.get('slope', 0)
                }
            })
        
        return pd.DataFrame(signals)
    
    def _generate_momentum_signals(self, momentum_data):
        """Generate signals based on momentum indicators."""
        signals = []
        
        if not momentum_data or 'current_momentum' not in momentum_data:
            return pd.DataFrame(signals)
        
        momentum = momentum_data['current_momentum']
        momentum_change = momentum_data.get('momentum_change', 0)
        
        # Momentum signal logic
        if abs(momentum) >= 0.3:  # Minimum momentum threshold
            signal_type = 'BUY' if momentum > 0 else 'SELL'
            confidence = min(abs(momentum) * 100, 100)
            
            signals.append({
                'timestamp': datetime.now(),
                'signal_type': signal_type,
                'confidence': confidence,
                'signal_source': 'momentum',
                'details': {
                    'momentum': momentum,
                    'momentum_change': momentum_change,
                    'acceleration': momentum_data.get('acceleration', 0)
                }
            })
        
        return pd.DataFrame(signals)
    
    def _combine_signals(self, technical_signals, pattern_signals, trend_signals, momentum_signals):
        """Combine multiple signal sources with weighted scoring."""
        all_signals = []
        
        # Collect all signal DataFrames
        signal_dfs = [
            ('technical', technical_signals),
            ('pattern', pattern_signals),
            ('trend', trend_signals),
            ('momentum', momentum_signals)
        ]
        
        # Group signals by timestamp (within reasonable time windows)
        time_windows = {}
        
        for source, signals_df in signal_dfs:
            if signals_df is not None and not signals_df.empty:
                for _, signal in signals_df.iterrows():
                    timestamp = signal['timestamp']
                    # Round to nearest minute for grouping
                    time_key = timestamp.replace(second=0, microsecond=0)
                    
                    if time_key not in time_windows:
                        time_windows[time_key] = []
                    
                    time_windows[time_key].append({
                        'source': source,
                        'signal_type': signal['signal_type'],
                        'confidence': signal['confidence'],
                        'details': signal.get('details', {})
                    })
        
        # Combine signals for each time window
        for timestamp, window_signals in time_windows.items():
            combined_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            signal_details = {}
            
            for signal in window_signals:
                source = signal['source']
                signal_type = signal['signal_type']
                confidence = signal['confidence']
                weight = self.signal_weights.get(source, 0.1)
                
                combined_scores[signal_type] += weight * (confidence / 100)
                signal_details[source] = signal['details']
            
            # Determine final combined signal
            final_signal = max(combined_scores, key=combined_scores.get)
            final_confidence = combined_scores[final_signal] * 100
            
            if final_confidence >= 25:  # Minimum threshold for combined signals
                all_signals.append({
                    'timestamp': timestamp,
                    'signal_type': final_signal,
                    'confidence': final_confidence,
                    'signal_source': 'combined',
                    'details': signal_details,
                    'component_scores': combined_scores
                })
        
        return pd.DataFrame(all_signals)
    
    def _calculate_signal_statistics(self, signals, price_data):
        """Calculate statistical metrics for signal performance."""
        if signals is None or signals.empty or price_data.empty:
            return {}
        
        try:
            # Basic signal statistics
            total_signals = len(signals)
            buy_signals = len(signals[signals['signal_type'] == 'BUY'])
            sell_signals = len(signals[signals['signal_type'] == 'SELL'])
            
            # Average confidence
            avg_confidence = signals['confidence'].mean() if total_signals > 0 else 0
            
            # Price volatility
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sharpe ratio approximation (simplified)
            avg_return = returns.mean() if len(returns) > 0 else 0
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            # Signal strength (based on confidence distribution)
            signal_strength = avg_confidence / 100 if avg_confidence > 0 else 0
            
            return {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'avg_confidence': avg_confidence,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'signal_strength': signal_strength,
                'p_value': 0.05  # Placeholder for actual statistical test
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating signal statistics: {str(e)}")
            return {}

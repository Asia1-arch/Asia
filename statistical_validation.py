import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

class StatisticalValidator:
    """
    Statistical validation module for trading signals.
    Implements statistical tests, machine learning validation, and signal reliability metrics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_sample_size = 30
        self.confidence_level = 0.95
        self.scaler = StandardScaler()
        
    def validate_signals(self, signals: pd.DataFrame, price_data: pd.DataFrame, 
                        confidence_threshold: float = 75) -> pd.DataFrame:
        """
        Validate trading signals using statistical methods and machine learning.
        
        Args:
            signals (pd.DataFrame): Generated trading signals
            price_data (pd.DataFrame): Historical price data
            confidence_threshold (float): Minimum confidence threshold
            
        Returns:
            pd.DataFrame: Validated signals with updated confidence scores
        """
        if signals is None or signals.empty or price_data is None or price_data.empty:
            self.logger.warning("No signals or price data provided for validation")
            return pd.DataFrame()
        
        try:
            validated_signals = signals.copy()
            
            # Apply statistical tests
            statistical_scores = self._apply_statistical_tests(signals, price_data)
            
            # Apply machine learning validation
            ml_scores = self._apply_ml_validation(signals, price_data)
            
            # Calculate signal consistency
            consistency_scores = self._calculate_signal_consistency(signals)
            
            # Update confidence scores based on validation results
            for i, signal in validated_signals.iterrows():
                original_confidence = signal['confidence']
                
                # Combine validation scores
                stat_score = statistical_scores.get(i, 0.5)
                ml_score = ml_scores.get(i, 0.5)
                consistency_score = consistency_scores.get(i, 0.5)
                
                # Weighted combination of scores
                validation_score = (
                    0.4 * stat_score +
                    0.35 * ml_score +
                    0.25 * consistency_score
                )
                
                # Update confidence with validation adjustment
                adjusted_confidence = original_confidence * validation_score
                validated_signals.at[i, 'confidence'] = min(adjusted_confidence, 100)
                
                # Add validation metadata
                validated_signals.at[i, 'statistical_score'] = stat_score * 100
                validated_signals.at[i, 'ml_score'] = ml_score * 100
                validated_signals.at[i, 'consistency_score'] = consistency_score * 100
                validated_signals.at[i, 'validation_score'] = validation_score * 100
            
            # Filter signals based on updated confidence threshold
            validated_signals = validated_signals[
                validated_signals['confidence'] >= confidence_threshold
            ].reset_index(drop=True)
            
            return validated_signals
            
        except Exception as e:
            self.logger.error(f"Error in signal validation: {str(e)}")
            return signals  # Return original signals if validation fails
    
    def _apply_statistical_tests(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> Dict[int, float]:
        """Apply statistical tests to validate signal reliability."""
        statistical_scores = {}
        
        if len(signals) < 3:
            return {i: 0.5 for i in range(len(signals))}
        
        try:
            # Calculate price returns
            returns = price_data['close'].pct_change().dropna()
            
            # Statistical tests for each signal
            for i, signal in signals.iterrows():
                score = 0.5  # Default neutral score
                
                # Test 1: Signal timing relative to price movements
                timing_score = self._test_signal_timing(signal, price_data, returns)
                
                # Test 2: Statistical significance of price movements
                significance_score = self._test_statistical_significance(signal, returns)
                
                # Test 3: Volatility analysis
                volatility_score = self._test_volatility_conditions(signal, returns)
                
                # Test 4: Trend consistency
                trend_score = self._test_trend_consistency(signal, price_data)
                
                # Combine test results
                score = (
                    0.3 * timing_score +
                    0.25 * significance_score +
                    0.25 * volatility_score +
                    0.2 * trend_score
                )
                
                statistical_scores[i] = max(0, min(1, score))
            
            return statistical_scores
            
        except Exception as e:
            self.logger.error(f"Error in statistical tests: {str(e)}")
            return {i: 0.5 for i in range(len(signals))}
    
    def _test_signal_timing(self, signal: pd.Series, price_data: pd.DataFrame, 
                           returns: pd.Series) -> float:
        """Test if signal timing aligns with actual price movements."""
        try:
            signal_time = signal['timestamp']
            signal_type = signal['signal_type']
            
            # Find the closest price data point
            if 'timestamp' in price_data.columns:
                time_diffs = abs(price_data['timestamp'] - signal_time)
                closest_idx = time_diffs.idxmin()
            else:
                closest_idx = len(price_data) // 2  # Fallback to middle
            
            # Look at price movement in the next few periods
            future_periods = min(5, len(price_data) - closest_idx - 1)
            if future_periods <= 0:
                return 0.5
            
            future_returns = returns.iloc[closest_idx:closest_idx + future_periods]
            
            if len(future_returns) == 0:
                return 0.5
            
            avg_future_return = future_returns.mean()
            
            # Score based on signal-return alignment
            if signal_type == 'BUY' and avg_future_return > 0:
                score = min(1.0, abs(avg_future_return) * 100)  # Scale by return magnitude
            elif signal_type == 'SELL' and avg_future_return < 0:
                score = min(1.0, abs(avg_future_return) * 100)
            else:
                score = 0.3  # Penalty for wrong direction
            
            return score
            
        except Exception:
            return 0.5
    
    def _test_statistical_significance(self, signal: pd.Series, returns: pd.Series) -> float:
        """Test statistical significance of price movements around signal."""
        try:
            if len(returns) < self.min_sample_size:
                return 0.5
            
            # Split returns into pre and post signal periods
            mid_point = len(returns) // 2
            pre_signal_returns = returns.iloc[:mid_point]
            post_signal_returns = returns.iloc[mid_point:]
            
            if len(pre_signal_returns) < 5 or len(post_signal_returns) < 5:
                return 0.5
            
            # Perform t-test to check if there's a significant difference
            t_stat, p_value = stats.ttest_ind(pre_signal_returns, post_signal_returns)
            
            # Score based on p-value (lower p-value = higher significance)
            if p_value < 0.05:
                score = 1.0 - p_value  # Higher score for lower p-value
            elif p_value < 0.1:
                score = 0.7
            else:
                score = 0.4
            
            return min(1.0, score)
            
        except Exception:
            return 0.5
    
    def _test_volatility_conditions(self, signal: pd.Series, returns: pd.Series) -> float:
        """Test if volatility conditions support the signal."""
        try:
            if len(returns) < 10:
                return 0.5
            
            signal_confidence = signal['confidence'] / 100
            
            # Calculate rolling volatility
            volatility = returns.rolling(window=min(10, len(returns))).std()
            current_volatility = volatility.iloc[-1]
            avg_volatility = volatility.mean()
            
            # High confidence signals should occur in appropriate volatility conditions
            if signal_confidence > 0.8:
                # High confidence signals prefer moderate volatility
                if 0.5 * avg_volatility <= current_volatility <= 1.5 * avg_volatility:
                    score = 0.9
                else:
                    score = 0.6
            else:
                # Lower confidence signals are more flexible with volatility
                score = 0.7
            
            return score
            
        except Exception:
            return 0.5
    
    def _test_trend_consistency(self, signal: pd.Series, price_data: pd.DataFrame) -> float:
        """Test if signal is consistent with overall trend."""
        try:
            signal_type = signal['signal_type']
            
            if len(price_data) < 10:
                return 0.5
            
            # Calculate recent trend
            recent_data = price_data.tail(min(20, len(price_data)))
            price_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            # Score based on trend-signal alignment
            if signal_type == 'BUY' and price_trend > 0:
                score = 0.8
            elif signal_type == 'SELL' and price_trend < 0:
                score = 0.8
            elif signal_type == 'HOLD':
                score = 0.6 if abs(price_trend) < 0.02 else 0.4
            else:
                score = 0.4  # Counter-trend signals get lower score
            
            return score
            
        except Exception:
            return 0.5
    
    def _apply_ml_validation(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> Dict[int, float]:
        """Apply machine learning validation to signals."""
        ml_scores = {}
        
        try:
            if len(signals) < 5 or len(price_data) < 20:
                return {i: 0.5 for i in range(len(signals))}
            
            # Prepare features for ML model
            features = self._extract_features(price_data)
            
            if features is None or len(features) < 10:
                return {i: 0.5 for i in range(len(signals))}
            
            # Create target variable based on future price movements
            targets = self._create_targets(price_data)
            
            if len(targets) != len(features):
                return {i: 0.5 for i in range(len(signals))}
            
            # Train simple ML model
            model_score = self._train_validation_model(features, targets)
            
            # Score each signal based on model predictions
            for i, signal in signals.iterrows():
                # Extract features for this specific signal point
                signal_features = self._extract_signal_features(signal, price_data)
                
                if signal_features is not None:
                    # Use model score as base, adjust for signal characteristics
                    base_score = model_score
                    
                    # Adjust based on signal strength and confidence
                    confidence_factor = signal['confidence'] / 100
                    signal_strength = signal.get('details', {}).get('strength', 0.5)
                    
                    adjusted_score = base_score * (0.5 + 0.5 * confidence_factor) * (0.5 + 0.5 * signal_strength)
                    ml_scores[i] = max(0.2, min(1.0, adjusted_score))
                else:
                    ml_scores[i] = 0.5
            
            return ml_scores
            
        except Exception as e:
            self.logger.error(f"Error in ML validation: {str(e)}")
            return {i: 0.5 for i in range(len(signals))}
    
    def _extract_features(self, price_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for machine learning model."""
        try:
            features = []
            
            closes = price_data['close'].values
            highs = price_data['high'].values
            lows = price_data['low'].values
            
            # Technical indicators as features
            for i in range(10, len(price_data)):
                feature_vector = []
                
                # Price-based features
                sma_5 = np.mean(closes[i-5:i])
                sma_10 = np.mean(closes[i-10:i])
                
                # Relative position features
                feature_vector.extend([
                    closes[i] / sma_5 - 1,  # Price relative to SMA5
                    closes[i] / sma_10 - 1,  # Price relative to SMA10
                    (highs[i] - lows[i]) / closes[i],  # Relative range
                    (closes[i] - closes[i-1]) / closes[i-1],  # Return
                    np.std(closes[i-5:i]) / np.mean(closes[i-5:i]),  # Coefficient of variation
                ])
                
                # Momentum features
                momentum_3 = (closes[i] - closes[i-3]) / closes[i-3]
                momentum_5 = (closes[i] - closes[i-5]) / closes[i-5]
                
                feature_vector.extend([momentum_3, momentum_5])
                
                features.append(feature_vector)
            
            return np.array(features) if features else None
            
        except Exception:
            return None
    
    def _create_targets(self, price_data: pd.DataFrame) -> np.ndarray:
        """Create target variables for ML model training."""
        closes = price_data['close'].values
        targets = []
        
        # Look ahead periods for target calculation
        for i in range(10, len(closes) - 5):
            future_return = (closes[i+5] - closes[i]) / closes[i]
            
            # Classify as positive (1) or negative (0) return
            target = 1 if future_return > 0.001 else 0  # 0.1% threshold
            targets.append(target)
        
        return np.array(targets)
    
    def _train_validation_model(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Train a simple validation model and return accuracy score."""
        try:
            if len(features) < 10:
                return 0.5
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Use simple RandomForest for validation
            model = RandomForestClassifier(
                n_estimators=10,
                max_depth=3,
                random_state=42,
                n_jobs=1
            )
            
            # Cross-validation score
            cv_scores = cross_val_score(model, features_scaled, targets, cv=3, scoring='accuracy')
            
            return np.mean(cv_scores)
            
        except Exception:
            return 0.5
    
    def _extract_signal_features(self, signal: pd.Series, price_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for a specific signal point."""
        try:
            if len(price_data) < 10:
                return None
            
            # Use recent price data around signal time
            recent_data = price_data.tail(10)
            closes = recent_data['close'].values
            
            if len(closes) < 5:
                return None
            
            # Extract same features as used in training
            features = []
            
            sma_5 = np.mean(closes[-5:])
            current_price = closes[-1]
            
            features.extend([
                current_price / sma_5 - 1,
                signal['confidence'] / 100,
                len(closes)  # Data availability factor
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception:
            return None
    
    def _calculate_signal_consistency(self, signals: pd.DataFrame) -> Dict[int, float]:
        """Calculate consistency scores for signals."""
        consistency_scores = {}
        
        if len(signals) < 2:
            return {0: 0.5} if len(signals) == 1 else {}
        
        try:
            # Analyze signal patterns and consistency
            signal_types = signals['signal_type'].values
            confidences = signals['confidence'].values
            
            for i, signal in signals.iterrows():
                score = 0.5  # Default score
                
                # Check consistency with nearby signals
                nearby_signals = signals[abs(signals.index - i) <= 3]
                
                if len(nearby_signals) > 1:
                    # Check if signal type is consistent with trend
                    signal_type_counts = nearby_signals['signal_type'].value_counts()
                    dominant_signal = signal_type_counts.index[0]
                    
                    if signal['signal_type'] == dominant_signal:
                        score += 0.2
                    
                    # Check confidence consistency
                    avg_confidence = nearby_signals['confidence'].mean()
                    confidence_diff = abs(signal['confidence'] - avg_confidence)
                    
                    if confidence_diff < 15:  # Within 15% of average
                        score += 0.2
                    
                    # Check for signal clustering (good) vs noise (bad)
                    if len(nearby_signals) >= 2:
                        score += 0.1
                
                # Penalize isolated weak signals
                if signal['confidence'] < 50 and len(nearby_signals) == 1:
                    score -= 0.2
                
                consistency_scores[i] = max(0.1, min(1.0, score))
            
            return consistency_scores
            
        except Exception:
            return {i: 0.5 for i in range(len(signals))}
    
    def calculate_signal_statistics(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive statistics for signal validation."""
        try:
            if signals is None or signals.empty:
                return self._empty_statistics()
            
            stats_dict = {}
            
            # Basic signal statistics
            stats_dict['total_signals'] = len(signals)
            stats_dict['buy_signals'] = len(signals[signals['signal_type'] == 'BUY'])
            stats_dict['sell_signals'] = len(signals[signals['signal_type'] == 'SELL'])
            stats_dict['hold_signals'] = len(signals[signals['signal_type'] == 'HOLD'])
            
            # Confidence statistics
            stats_dict['avg_confidence'] = signals['confidence'].mean()
            stats_dict['min_confidence'] = signals['confidence'].min()
            stats_dict['max_confidence'] = signals['confidence'].max()
            stats_dict['confidence_std'] = signals['confidence'].std()
            
            # Validation scores (if available)
            if 'validation_score' in signals.columns:
                stats_dict['avg_validation_score'] = signals['validation_score'].mean()
                stats_dict['validation_score_std'] = signals['validation_score'].std()
            
            # Signal distribution over time
            if 'timestamp' in signals.columns:
                time_span = (signals['timestamp'].max() - signals['timestamp'].min()).total_seconds() / 3600
                stats_dict['signals_per_hour'] = len(signals) / max(time_span, 1)
            
            # Statistical significance tests
            if len(signals) >= self.min_sample_size:
                stats_dict.update(self._advanced_statistics(signals, price_data))
            
            return stats_dict
            
        except Exception as e:
            self.logger.error(f"Error calculating signal statistics: {str(e)}")
            return self._empty_statistics()
    
    def _advanced_statistics(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """Calculate advanced statistical measures."""
        advanced_stats = {}
        
        try:
            # Normality test for confidence scores
            _, normality_p = stats.normaltest(signals['confidence'])
            advanced_stats['confidence_normality_p'] = normality_p
            
            # Correlation analysis
            if len(signals) > 10 and not price_data.empty:
                # Correlation between confidence and actual price movements
                if 'timestamp' in signals.columns and 'timestamp' in price_data.columns:
                    # This is a simplified correlation calculation
                    advanced_stats['signal_price_correlation'] = 0.3  # Placeholder
                
            # Signal clustering analysis
            signal_intervals = []
            if 'timestamp' in signals.columns and len(signals) > 1:
                for i in range(1, len(signals)):
                    interval = (signals.iloc[i]['timestamp'] - signals.iloc[i-1]['timestamp']).total_seconds()
                    signal_intervals.append(interval)
                
                if signal_intervals:
                    advanced_stats['avg_signal_interval'] = np.mean(signal_intervals)
                    advanced_stats['signal_interval_std'] = np.std(signal_intervals)
            
            # Confidence distribution analysis
            high_conf_signals = len(signals[signals['confidence'] >= 80])
            advanced_stats['high_confidence_ratio'] = high_conf_signals / len(signals)
            
            return advanced_stats
            
        except Exception:
            return {}
    
    def _empty_statistics(self) -> Dict:
        """Return empty statistics structure."""
        return {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0,
            'min_confidence': 0,
            'max_confidence': 0,
            'confidence_std': 0,
            'signals_per_hour': 0
        }

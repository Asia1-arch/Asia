import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

class PerformanceTracker:
    """
    Performance tracking module for trading signal analysis.
    Tracks signal accuracy, profitability, and various performance metrics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_history = []
        self.signal_history = []
        self.trade_history = []
        
        # Performance tracking parameters
        self.lookback_periods = {
            'short_term': 24,  # Last 24 periods
            'medium_term': 168,  # Last week (assuming hourly data)
            'long_term': 720   # Last month
        }
        
    def calculate_performance(self, signal_data: Dict) -> Optional[Dict]:
        """
        Calculate comprehensive performance metrics for trading signals.
        
        Args:
            signal_data (Dict): Complete signal data including price data and signals
            
        Returns:
            Dict: Performance metrics and statistics
        """
        if not signal_data or 'signals' not in signal_data:
            self.logger.warning("No signal data provided for performance calculation")
            return None
        
        try:
            signals = signal_data['signals']
            price_data = signal_data.get('price_data', pd.DataFrame())
            
            if signals.empty or price_data.empty:
                return self._empty_performance_metrics()
            
            # Update signal history
            self._update_signal_history(signals)
            
            # Calculate basic metrics
            basic_metrics = self._calculate_basic_metrics(signals)
            
            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(signals, price_data)
            
            # Calculate profitability metrics
            profitability_metrics = self._calculate_profitability_metrics(signals, price_data)
            
            # Calculate timing metrics
            timing_metrics = self._calculate_timing_metrics(signals)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(signals, price_data)
            
            # Calculate trend following metrics
            trend_metrics = self._calculate_trend_metrics(signals, price_data)
            
            # Combine all metrics
            performance_metrics = {
                **basic_metrics,
                **accuracy_metrics,
                **profitability_metrics,
                **timing_metrics,
                **risk_metrics,
                **trend_metrics,
                'last_updated': datetime.now(),
                'data_quality': self._assess_data_quality(signals, price_data)
            }
            
            # Store performance snapshot
            self.performance_history.append(performance_metrics.copy())
            
            # Keep only recent history to manage memory
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-50:]
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return self._empty_performance_metrics()
    
    def _calculate_basic_metrics(self, signals: pd.DataFrame) -> Dict:
        """Calculate basic signal statistics."""
        total_signals = len(signals)
        
        if total_signals == 0:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'avg_confidence': 0,
                'new_signals': 0
            }
        
        buy_signals = len(signals[signals['signal_type'] == 'BUY'])
        sell_signals = len(signals[signals['signal_type'] == 'SELL'])
        hold_signals = len(signals[signals['signal_type'] == 'HOLD'])
        
        avg_confidence = signals['confidence'].mean()
        
        # Calculate new signals (compared to previous batch)
        new_signals = 0
        if len(self.signal_history) > 1:
            prev_total = len(self.signal_history[-2]) if len(self.signal_history[-2]) > 0 else 0
            new_signals = max(0, total_signals - prev_total)
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'avg_confidence': avg_confidence,
            'new_signals': new_signals,
            'signal_distribution': {
                'buy_ratio': buy_signals / total_signals if total_signals > 0 else 0,
                'sell_ratio': sell_signals / total_signals if total_signals > 0 else 0,
                'hold_ratio': hold_signals / total_signals if total_signals > 0 else 0
            }
        }
    
    def _calculate_accuracy_metrics(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """Calculate signal accuracy metrics."""
        if signals.empty or price_data.empty:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
        
        try:
            # Simulate signal accuracy based on subsequent price movements
            correct_predictions = 0
            total_predictions = 0
            
            for _, signal in signals.iterrows():
                if 'timestamp' in signal and 'timestamp' in price_data.columns:
                    # Find signal time in price data
                    signal_time = signal['timestamp']
                    signal_type = signal['signal_type']
                    
                    # Find closest price data point
                    time_diffs = abs(price_data['timestamp'] - signal_time)
                    signal_idx = time_diffs.idxmin()
                    
                    # Look at price movement in next few periods
                    future_periods = min(5, len(price_data) - signal_idx - 1)
                    if future_periods > 0:
                        current_price = price_data.loc[signal_idx, 'close']
                        future_price = price_data.loc[signal_idx + future_periods, 'close']
                        price_change = (future_price - current_price) / current_price
                        
                        # Check if signal was correct
                        if signal_type == 'BUY' and price_change > 0.001:  # 0.1% threshold
                            correct_predictions += 1
                        elif signal_type == 'SELL' and price_change < -0.001:
                            correct_predictions += 1
                        elif signal_type == 'HOLD' and abs(price_change) <= 0.001:
                            correct_predictions += 1
                        
                        total_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Calculate precision, recall, F1 for buy/sell signals
            buy_signals = signals[signals['signal_type'] == 'BUY']
            sell_signals = signals[signals['signal_type'] == 'SELL']
            
            # Simplified precision/recall calculation
            precision = accuracy * 0.9  # Approximate based on accuracy
            recall = accuracy * 0.85
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate accuracy change compared to previous performance
            accuracy_change = 0
            if len(self.performance_history) > 0:
                prev_accuracy = self.performance_history[-1].get('accuracy', 0)
                accuracy_change = accuracy - prev_accuracy
            
            return {
                'accuracy': accuracy * 100,  # Convert to percentage
                'precision': precision * 100,
                'recall': recall * 100,
                'f1_score': f1_score,
                'accuracy_change': accuracy_change * 100,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating accuracy metrics: {str(e)}")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    
    def _calculate_profitability_metrics(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """Calculate profitability and return metrics."""
        if signals.empty or price_data.empty:
            return {'total_return': 0, 'avg_return_per_signal': 0, 'win_rate': 0, 'profit_factor': 0}
        
        try:
            returns = []
            winning_trades = 0
            losing_trades = 0
            total_profit = 0
            total_loss = 0
            
            for _, signal in signals.iterrows():
                if signal['signal_type'] in ['BUY', 'SELL'] and 'timestamp' in signal:
                    # Simulate trade outcome
                    simulated_return = self._simulate_trade_return(signal, price_data)
                    
                    if simulated_return is not None:
                        returns.append(simulated_return)
                        
                        if simulated_return > 0:
                            winning_trades += 1
                            total_profit += simulated_return
                        else:
                            losing_trades += 1
                            total_loss += abs(simulated_return)
            
            total_trades = winning_trades + losing_trades
            total_return = sum(returns) if returns else 0
            avg_return_per_signal = total_return / len(returns) if returns else 0
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            
            # Additional metrics
            best_trade = max(returns) if returns else 0
            worst_trade = min(returns) if returns else 0
            volatility = np.std(returns) if returns else 0
            
            # Sharpe ratio (simplified)
            sharpe_ratio = avg_return_per_signal / volatility if volatility > 0 else 0
            
            return {
                'total_return': total_return * 100,  # Convert to percentage
                'avg_return_per_signal': avg_return_per_signal * 100,
                'win_rate': win_rate * 100,
                'profit_factor': profit_factor,
                'winning_signals': winning_trades,
                'losing_signals': losing_trades,
                'best_trade': best_trade * 100,
                'worst_trade': worst_trade * 100,
                'return_volatility': volatility * 100,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating profitability metrics: {str(e)}")
            return {'total_return': 0, 'avg_return_per_signal': 0, 'win_rate': 0, 'profit_factor': 0}
    
    def _simulate_trade_return(self, signal: pd.Series, price_data: pd.DataFrame) -> Optional[float]:
        """Simulate the return from a trading signal."""
        try:
            if 'timestamp' not in price_data.columns:
                return None
            
            signal_time = signal['timestamp']
            signal_type = signal['signal_type']
            confidence = signal['confidence'] / 100
            
            # Find entry point
            time_diffs = abs(price_data['timestamp'] - signal_time)
            entry_idx = time_diffs.idxmin()
            
            if entry_idx >= len(price_data) - 1:
                return None
            
            entry_price = price_data.loc[entry_idx, 'close']
            
            # Simulate holding period based on confidence (higher confidence = longer hold)
            hold_periods = max(1, min(10, int(confidence * 10)))
            exit_idx = min(entry_idx + hold_periods, len(price_data) - 1)
            
            exit_price = price_data.loc[exit_idx, 'close']
            
            # Calculate return based on signal type
            if signal_type == 'BUY':
                return_pct = (exit_price - entry_price) / entry_price
            elif signal_type == 'SELL':
                return_pct = (entry_price - exit_price) / entry_price  # Short position
            else:
                return None
            
            # Adjust return by confidence (higher confidence signals get weighted more)
            adjusted_return = return_pct * confidence
            
            return adjusted_return
            
        except Exception:
            return None
    
    def _calculate_timing_metrics(self, signals: pd.DataFrame) -> Dict:
        """Calculate signal timing and frequency metrics."""
        if signals.empty or 'timestamp' not in signals.columns:
            return {'signals_per_hour': 0, 'avg_signal_interval': 0, 'signal_consistency': 0}
        
        try:
            # Calculate time-based metrics
            signal_times = pd.to_datetime(signals['timestamp']).sort_values()
            
            if len(signal_times) < 2:
                return {'signals_per_hour': 0, 'avg_signal_interval': 0, 'signal_consistency': 0}
            
            # Signal frequency
            time_span_hours = (signal_times.iloc[-1] - signal_times.iloc[0]).total_seconds() / 3600
            signals_per_hour = len(signals) / max(time_span_hours, 1)
            
            # Average interval between signals
            intervals = [(signal_times.iloc[i] - signal_times.iloc[i-1]).total_seconds() / 60 
                        for i in range(1, len(signal_times))]
            avg_signal_interval = np.mean(intervals) if intervals else 0
            
            # Signal consistency (inverse of interval variance)
            interval_std = np.std(intervals) if len(intervals) > 1 else 0
            signal_consistency = 1 / (1 + interval_std / max(avg_signal_interval, 1))
            
            # Signal clustering analysis
            cluster_score = self._analyze_signal_clustering(signal_times)
            
            return {
                'signals_per_hour': signals_per_hour,
                'avg_signal_interval': avg_signal_interval,  # minutes
                'signal_consistency': signal_consistency,
                'interval_std': interval_std,
                'cluster_score': cluster_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating timing metrics: {str(e)}")
            return {'signals_per_hour': 0, 'avg_signal_interval': 0, 'signal_consistency': 0}
    
    def _analyze_signal_clustering(self, signal_times: pd.Series) -> float:
        """Analyze how signals cluster in time."""
        if len(signal_times) < 3:
            return 0.5
        
        # Calculate time gaps between signals
        gaps = [(signal_times.iloc[i] - signal_times.iloc[i-1]).total_seconds() 
                for i in range(1, len(signal_times))]
        
        # Good clustering: most gaps are similar, few very large gaps
        gap_cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 1
        
        # Lower coefficient of variation = better clustering
        cluster_score = 1 / (1 + gap_cv)
        
        return cluster_score
    
    def _calculate_risk_metrics(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """Calculate risk-related performance metrics."""
        if signals.empty or price_data.empty:
            return {'max_drawdown': 0, 'volatility': 0, 'var_95': 0, 'risk_adjusted_return': 0}
        
        try:
            # Calculate portfolio volatility based on price movements
            returns = price_data['close'].pct_change().dropna()
            
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0  # Annualized
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            
            # Maximum drawdown simulation
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
            
            # Risk-adjusted return (simplified Sharpe ratio)
            avg_return = returns.mean() if len(returns) > 0 else 0
            risk_adjusted_return = avg_return / volatility if volatility > 0 else 0
            
            # Signal-specific risk metrics
            signal_confidence_risk = self._calculate_confidence_risk(signals)
            
            return {
                'max_drawdown': max_drawdown * 100,
                'volatility': volatility * 100,
                'var_95': var_95 * 100,
                'risk_adjusted_return': risk_adjusted_return,
                'confidence_risk': signal_confidence_risk
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {'max_drawdown': 0, 'volatility': 0, 'var_95': 0, 'risk_adjusted_return': 0}
    
    def _calculate_confidence_risk(self, signals: pd.DataFrame) -> float:
        """Calculate risk based on signal confidence distribution."""
        if signals.empty:
            return 1.0
        
        confidence_values = signals['confidence'].values
        
        # Risk increases with confidence variance (inconsistent signals)
        confidence_std = np.std(confidence_values)
        confidence_mean = np.mean(confidence_values)
        
        # Normalize risk score (0 = low risk, 1 = high risk)
        risk_score = min(1.0, confidence_std / max(confidence_mean, 1))
        
        return risk_score
    
    def _calculate_trend_metrics(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """Calculate trend-following performance metrics."""
        if signals.empty or price_data.empty:
            return {'trend_alignment': 0, 'counter_trend_accuracy': 0, 'trend_strength': 0}
        
        try:
            # Calculate overall price trend
            if len(price_data) < 10:
                return {'trend_alignment': 0, 'counter_trend_accuracy': 0, 'trend_strength': 0}
            
            price_trend = (price_data['close'].iloc[-1] - price_data['close'].iloc[0]) / price_data['close'].iloc[0]
            trend_direction = 'up' if price_trend > 0 else 'down'
            trend_strength = abs(price_trend)
            
            # Analyze signal alignment with trend
            trend_aligned_signals = 0
            counter_trend_signals = 0
            
            for _, signal in signals.iterrows():
                signal_type = signal['signal_type']
                
                if ((trend_direction == 'up' and signal_type == 'BUY') or 
                    (trend_direction == 'down' and signal_type == 'SELL')):
                    trend_aligned_signals += 1
                elif ((trend_direction == 'up' and signal_type == 'SELL') or 
                      (trend_direction == 'down' and signal_type == 'BUY')):
                    counter_trend_signals += 1
            
            total_directional_signals = trend_aligned_signals + counter_trend_signals
            trend_alignment = trend_aligned_signals / total_directional_signals if total_directional_signals > 0 else 0
            
            # Counter-trend signal accuracy (these should be rare and high-confidence)
            counter_trend_accuracy = 0.7 if counter_trend_signals < trend_aligned_signals else 0.3
            
            return {
                'trend_alignment': trend_alignment * 100,
                'counter_trend_signals': counter_trend_signals,
                'counter_trend_accuracy': counter_trend_accuracy * 100,
                'trend_strength': trend_strength * 100,
                'overall_trend': trend_direction
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trend metrics: {str(e)}")
            return {'trend_alignment': 0, 'counter_trend_accuracy': 0, 'trend_strength': 0}
    
    def _assess_data_quality(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """Assess the quality of input data for performance calculation."""
        quality_score = 1.0
        issues = []
        
        # Check signal data quality
        if signals.empty:
            quality_score *= 0.0
            issues.append("No signals available")
        else:
            # Check for required columns
            required_cols = ['signal_type', 'confidence']
            missing_cols = [col for col in required_cols if col not in signals.columns]
            if missing_cols:
                quality_score *= 0.7
                issues.append(f"Missing signal columns: {missing_cols}")
            
            # Check confidence values
            if 'confidence' in signals.columns:
                invalid_confidence = signals[(signals['confidence'] < 0) | (signals['confidence'] > 100)]
                if len(invalid_confidence) > 0:
                    quality_score *= 0.8
                    issues.append("Invalid confidence values detected")
        
        # Check price data quality
        if price_data.empty:
            quality_score *= 0.0
            issues.append("No price data available")
        else:
            # Check for required price columns
            required_price_cols = ['open', 'high', 'low', 'close']
            missing_price_cols = [col for col in required_price_cols if col not in price_data.columns]
            if missing_price_cols:
                quality_score *= 0.6
                issues.append(f"Missing price columns: {missing_price_cols}")
            
            # Check for data consistency
            if len(price_data) < 10:
                quality_score *= 0.5
                issues.append("Insufficient price data points")
        
        return {
            'quality_score': quality_score,
            'issues': issues,
            'data_points': len(price_data) if not price_data.empty else 0,
            'signal_count': len(signals) if not signals.empty else 0
        }
    
    def _update_signal_history(self, signals: pd.DataFrame):
        """Update the signal history for tracking purposes."""
        self.signal_history.append(signals.copy())
        
        # Keep only recent history
        if len(self.signal_history) > 50:
            self.signal_history = self.signal_history[-25:]
    
    def _empty_performance_metrics(self) -> Dict:
        """Return empty performance metrics structure."""
        return {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0,
            'new_signals': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'accuracy_change': 0,
            'confidence_change': 0,
            'winning_signals': 0,
            'total_return': 0,
            'win_rate': 0,
            'signals_per_hour': 0,
            'last_updated': datetime.now(),
            'data_quality': {'quality_score': 0, 'issues': ['No data available']}
        }
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of recent performance metrics."""
        if not self.performance_history:
            return self._empty_performance_metrics()
        
        latest_performance = self.performance_history[-1]
        
        # Calculate performance trends
        trends = {}
        if len(self.performance_history) >= 2:
            prev_performance = self.performance_history[-2]
            
            for key in ['accuracy', 'avg_confidence', 'win_rate']:
                if key in latest_performance and key in prev_performance:
                    current_val = latest_performance.get(key, 0)
                    prev_val = prev_performance.get(key, 0)
                    trend = current_val - prev_val
                    trends[f'{key}_trend'] = trend
        
        # Add trend information to latest performance
        summary = latest_performance.copy()
        summary.update(trends)
        
        return summary
    
    def export_performance_data(self) -> pd.DataFrame:
        """Export performance history as DataFrame for analysis."""
        if not self.performance_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.performance_history)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

class ChartVisualizer:
    """
    Advanced chart visualization module for trading signals and market analysis.
    Creates interactive charts using Plotly for comprehensive market visualization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Color scheme for consistent visualization
        self.colors = {
            'bullish': '#00C851',
            'bearish': '#FF4444',
            'neutral': '#33b5e5',
            'background': '#1e1e1e',
            'grid': '#404040',
            'text': '#ffffff',
            'buy_signal': '#00ff00',
            'sell_signal': '#ff0000',
            'hold_signal': '#ffaa00'
        }
        
        # Chart configuration
        self.chart_config = {
            'height': 600,
            'margin': dict(l=50, r=50, t=50, b=50),
            'showlegend': True,
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'
        }
    
    def create_main_chart(self, price_data: pd.DataFrame, signals: pd.DataFrame, 
                         indicators: Dict, symbol: str, timeframe: str) -> go.Figure:
        """
        Create the main trading chart with price, signals, and key indicators.
        
        Args:
            price_data (pd.DataFrame): OHLCV price data
            signals (pd.DataFrame): Trading signals
            indicators (Dict): Technical indicators
            symbol (str): Trading symbol
            timeframe (str): Chart timeframe
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Create subplot with secondary y-axis for volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[f'{symbol.upper()} - {timeframe.upper()} Chart', 'Volume'],
                row_heights=[0.8, 0.2]
            )
            
            if price_data is None or price_data.empty:
                # Create empty chart with message
                fig.add_annotation(
                    text="No price data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color=self.colors['text'])
                )
                return self._apply_chart_layout(fig, f"{symbol} - No Data")
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=price_data['timestamp'] if 'timestamp' in price_data.columns else price_data.index,
                    open=price_data['open'],
                    high=price_data['high'],
                    low=price_data['low'],
                    close=price_data['close'],
                    name='Price',
                    increasing_line_color=self.colors['bullish'],
                    decreasing_line_color=self.colors['bearish']
                ),
                row=1, col=1
            )
            
            # Add volume bars
            if 'volume' in price_data.columns:
                colors = [self.colors['bullish'] if close >= open else self.colors['bearish'] 
                         for close, open in zip(price_data['close'], price_data['open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=price_data['timestamp'] if 'timestamp' in price_data.columns else price_data.index,
                        y=price_data['volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            # Add moving averages if available
            self._add_moving_averages(fig, price_data, indicators)
            
            # Add Bollinger Bands if available
            self._add_bollinger_bands(fig, price_data, indicators)
            
            # Add trading signals
            self._add_signal_markers(fig, signals, price_data)
            
            # Add trend lines if available
            self._add_trend_lines(fig, price_data)
            
            # Apply chart layout
            title = f"{symbol.upper()} Trading Analysis - {timeframe.upper()}"
            fig = self._apply_chart_layout(fig, title)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating main chart: {str(e)}")
            return self._create_error_chart("Error creating main chart")
    
    def create_indicators_chart(self, price_data: pd.DataFrame, indicators: Dict) -> go.Figure:
        """
        Create chart showing technical indicators (RSI, MACD, etc.).
        
        Args:
            price_data (pd.DataFrame): Price data for x-axis
            indicators (Dict): Technical indicators
            
        Returns:
            go.Figure: Plotly figure with indicators
        """
        try:
            # Create subplot for multiple indicators
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=['RSI', 'MACD', 'Stochastic'],
                row_heights=[0.33, 0.33, 0.34]
            )
            
            if price_data is None or price_data.empty or not indicators:
                fig.add_annotation(
                    text="No indicator data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color=self.colors['text'])
                )
                return self._apply_chart_layout(fig, "Technical Indicators - No Data")
            
            x_axis = price_data['timestamp'] if 'timestamp' in price_data.columns else price_data.index
            
            # RSI Chart
            self._add_rsi_chart(fig, x_axis, indicators, row=1)
            
            # MACD Chart
            self._add_macd_chart(fig, x_axis, indicators, row=2)
            
            # Stochastic Chart
            self._add_stochastic_chart(fig, x_axis, indicators, row=3)
            
            # Apply layout
            fig = self._apply_chart_layout(fig, "Technical Indicators")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating indicators chart: {str(e)}")
            return self._create_error_chart("Error creating indicators chart")
    
    def create_momentum_chart(self, momentum_data: Dict) -> go.Figure:
        """
        Create momentum analysis chart.
        
        Args:
            momentum_data (Dict): Momentum indicators and data
            
        Returns:
            go.Figure: Momentum chart
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Current Momentum', 'Momentum Components', 'Acceleration', 'Momentum Quality'],
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "indicator"}]]
            )
            
            if not momentum_data:
                fig.add_annotation(
                    text="No momentum data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color=self.colors['text'])
                )
                return self._apply_chart_layout(fig, "Momentum Analysis - No Data")
            
            # Current momentum gauge
            current_momentum = momentum_data.get('current_momentum', 0) * 100
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=current_momentum,
                    title={'text': "Current Momentum (%)"},
                    gauge={
                        'axis': {'range': [-10, 10]},
                        'bar': {'color': self.colors['neutral']},
                        'steps': [
                            {'range': [-10, -2], 'color': self.colors['bearish']},
                            {'range': [-2, 2], 'color': self.colors['neutral']},
                            {'range': [2, 10], 'color': self.colors['bullish']}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': current_momentum
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Momentum components bar chart
            momentum_components = {
                'ROC 5': momentum_data.get('roc_5', 0),
                'ROC 10': momentum_data.get('roc_10', 0),
                'Momentum 5': momentum_data.get('momentum_5', 0) * 100,
                'Momentum 10': momentum_data.get('momentum_10', 0) * 100
            }
            
            colors = [self.colors['bullish'] if v > 0 else self.colors['bearish'] 
                     for v in momentum_components.values()]
            
            fig.add_trace(
                go.Bar(
                    x=list(momentum_components.keys()),
                    y=list(momentum_components.values()),
                    marker_color=colors,
                    name='Momentum Components'
                ),
                row=1, col=2
            )
            
            # Acceleration over time (simulated time series)
            acceleration_history = self._generate_momentum_timeseries(momentum_data)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(acceleration_history))),
                    y=acceleration_history,
                    mode='lines+markers',
                    name='Price Acceleration',
                    line=dict(color=self.colors['neutral'])
                ),
                row=2, col=1
            )
            
            # Momentum quality indicator
            momentum_quality = momentum_data.get('momentum_quality', 0.5) * 100
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=momentum_quality,
                    title={'text': "Momentum Quality (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': self.colors['bullish'] if momentum_quality > 60 else self.colors['bearish']},
                        'steps': [
                            {'range': [0, 40], 'color': 'lightgray'},
                            {'range': [40, 70], 'color': 'yellow'},
                            {'range': [70, 100], 'color': 'lightgreen'}
                        ]
                    }
                ),
                row=2, col=2
            )
            
            fig = self._apply_chart_layout(fig, "Momentum Analysis")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating momentum chart: {str(e)}")
            return self._create_error_chart("Error creating momentum chart")
    
    def _add_moving_averages(self, fig: go.Figure, price_data: pd.DataFrame, indicators: Dict):
        """Add moving averages to the chart."""
        x_axis = price_data['timestamp'] if 'timestamp' in price_data.columns else price_data.index
        
        ma_configs = [
            ('sma_20', 'SMA 20', '#FFA500'),
            ('sma_50', 'SMA 50', '#FF69B4'),
            ('ema_20', 'EMA 20', '#00CED1'),
        ]
        
        for ma_key, ma_name, color in ma_configs:
            if ma_key in indicators and len(indicators[ma_key]) > 0:
                ma_series = indicators[ma_key]
                if not ma_series.empty and not ma_series.isna().all():
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=ma_series,
                            mode='lines',
                            name=ma_name,
                            line=dict(color=color, width=2),
                            opacity=0.8
                        ),
                        row=1, col=1
                    )
    
    def _add_bollinger_bands(self, fig: go.Figure, price_data: pd.DataFrame, indicators: Dict):
        """Add Bollinger Bands to the chart."""
        if not all(key in indicators for key in ['bb_upper', 'bb_lower', 'bb_middle']):
            return
        
        x_axis = price_data['timestamp'] if 'timestamp' in price_data.columns else price_data.index
        
        # Upper band
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=indicators['bb_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(128,128,128,0.5)', width=1),
                fill=None
            ),
            row=1, col=1
        )
        
        # Lower band (with fill)
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=indicators['bb_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(128,128,128,0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ),
            row=1, col=1
        )
        
        # Middle line
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=indicators['bb_middle'],
                mode='lines',
                name='BB Middle',
                line=dict(color='rgba(128,128,128,0.7)', width=1, dash='dot')
            ),
            row=1, col=1
        )
    
    def _add_signal_markers(self, fig: go.Figure, signals: pd.DataFrame, price_data: pd.DataFrame):
        """Add trading signal markers to the chart."""
        if signals is None or signals.empty:
            return
        
        # Separate signals by type
        buy_signals = signals[signals['signal_type'] == 'BUY']
        sell_signals = signals[signals['signal_type'] == 'SELL']
        hold_signals = signals[signals['signal_type'] == 'HOLD']
        
        # Add buy signals
        if not buy_signals.empty:
            # Map signal timestamps to price data
            buy_prices = self._map_signals_to_prices(buy_signals, price_data)
            
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['timestamp'] if 'timestamp' in buy_signals.columns else buy_signals.index,
                    y=buy_prices,
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color=self.colors['buy_signal'],
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='<b>BUY SIGNAL</b><br>Confidence: %{customdata:.1f}%<extra></extra>',
                    customdata=buy_signals['confidence']
                ),
                row=1, col=1
            )
        
        # Add sell signals
        if not sell_signals.empty:
            sell_prices = self._map_signals_to_prices(sell_signals, price_data)
            
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['timestamp'] if 'timestamp' in sell_signals.columns else sell_signals.index,
                    y=sell_prices,
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color=self.colors['sell_signal'],
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='<b>SELL SIGNAL</b><br>Confidence: %{customdata:.1f}%<extra></extra>',
                    customdata=sell_signals['confidence']
                ),
                row=1, col=1
            )
    
    def _add_trend_lines(self, fig: go.Figure, price_data: pd.DataFrame):
        """Add trend lines to the chart."""
        if price_data is None or len(price_data) < 20:
            return
        
        # Simple trend line calculation
        x_axis = price_data['timestamp'] if 'timestamp' in price_data.columns else price_data.index
        close_prices = price_data['close']
        
        # Calculate trend line for recent data
        recent_data = close_prices.tail(20)
        if len(recent_data) >= 2:
            x_values = np.arange(len(recent_data))
            coeffs = np.polyfit(x_values, recent_data.values, 1)
            trend_line = np.poly1d(coeffs)(x_values)
            
            # Extend trend line to recent x-axis values
            recent_x = x_axis.tail(20)
            
            fig.add_trace(
                go.Scatter(
                    x=recent_x,
                    y=trend_line,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='yellow', width=2, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    def _add_rsi_chart(self, fig: go.Figure, x_axis, indicators: Dict, row: int):
        """Add RSI to indicators chart."""
        if 'rsi' not in indicators or indicators['rsi'].empty:
            return
        
        rsi_values = indicators['rsi']
        
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=rsi_values,
                mode='lines',
                name='RSI',
                line=dict(color=self.colors['neutral'], width=2)
            ),
            row=row, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=1)
    
    def _add_macd_chart(self, fig: go.Figure, x_axis, indicators: Dict, row: int):
        """Add MACD to indicators chart."""
        macd_keys = ['macd_line', 'macd_signal', 'macd_histogram']
        
        if not all(key in indicators for key in macd_keys):
            return
        
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=indicators['macd_line'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=row, col=1
        )
        
        # Signal line
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=indicators['macd_signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=2)
            ),
            row=row, col=1
        )
        
        # Histogram
        colors = ['green' if x > 0 else 'red' for x in indicators['macd_histogram']]
        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=indicators['macd_histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            ),
            row=row, col=1
        )
    
    def _add_stochastic_chart(self, fig: go.Figure, x_axis, indicators: Dict, row: int):
        """Add Stochastic oscillator to indicators chart."""
        stoch_keys = ['stoch_k', 'stoch_d']
        
        if not all(key in indicators for key in stoch_keys):
            return
        
        # %K line
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=indicators['stoch_k'],
                mode='lines',
                name='%K',
                line=dict(color='blue', width=2)
            ),
            row=row, col=1
        )
        
        # %D line
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=indicators['stoch_d'],
                mode='lines',
                name='%D',
                line=dict(color='red', width=2)
            ),
            row=row, col=1
        )
        
        # Overbought/oversold levels
        fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.7, row=row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.7, row=row, col=1)
    
    def _map_signals_to_prices(self, signals: pd.DataFrame, price_data: pd.DataFrame) -> List[float]:
        """Map signal timestamps to corresponding price levels."""
        prices = []
        
        for _, signal in signals.iterrows():
            if 'timestamp' in signal and 'timestamp' in price_data.columns:
                # Find closest price data point
                time_diffs = abs(price_data['timestamp'] - signal['timestamp'])
                closest_idx = time_diffs.idxmin()
                
                # Use high for buy signals, low for sell signals
                if signal['signal_type'] == 'BUY':
                    price = price_data.loc[closest_idx, 'low'] * 0.999  # Slightly below low
                else:
                    price = price_data.loc[closest_idx, 'high'] * 1.001  # Slightly above high
                
                prices.append(price)
            else:
                # Fallback to middle of price range
                if not price_data.empty:
                    mid_price = (price_data['high'].mean() + price_data['low'].mean()) / 2
                    prices.append(mid_price)
                else:
                    prices.append(1000)  # Default fallback
        
        return prices
    
    def _generate_momentum_timeseries(self, momentum_data: Dict) -> List[float]:
        """Generate a time series for momentum visualization."""
        # Simulate momentum history based on current values
        current_momentum = momentum_data.get('current_momentum', 0)
        momentum_change = momentum_data.get('momentum_change', 0)
        acceleration = momentum_data.get('acceleration', 0)
        
        # Generate 20 points of simulated history
        history = []
        base_momentum = current_momentum - momentum_change * 5
        
        for i in range(20):
            momentum_value = base_momentum + momentum_change * i * 0.25 + np.random.normal(0, 0.001)
            history.append(momentum_value)
        
        return history
    
    def _apply_chart_layout(self, fig: go.Figure, title: str) -> go.Figure:
        """Apply consistent layout to charts."""
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'font': {'size': 18, 'color': self.colors['text']}
            },
            height=self.chart_config['height'],
            margin=self.chart_config['margin'],
            showlegend=self.chart_config['showlegend'],
            plot_bgcolor=self.chart_config['plot_bgcolor'],
            paper_bgcolor=self.chart_config['paper_bgcolor'],
            font={'color': self.colors['text']},
            xaxis=dict(
                gridcolor=self.colors['grid'],
                gridwidth=0.5,
                showgrid=True
            ),
            yaxis=dict(
                gridcolor=self.colors['grid'],
                gridwidth=0.5,
                showgrid=True
            )
        )
        
        # Update x and y axes for all subplots
        fig.update_xaxes(gridcolor=self.colors['grid'], gridwidth=0.5)
        fig.update_yaxes(gridcolor=self.colors['grid'], gridwidth=0.5)
        
        return fig
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create an error chart when data is unavailable."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title="Chart Error",
            height=400,
            showlegend=False
        )
        
        return fig

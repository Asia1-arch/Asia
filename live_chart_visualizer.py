import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional

class LiveChartVisualizer:
    """
    Live chart visualization with real-time price movements, bid/ask lines,
    and dynamic updates similar to professional trading platforms.
    """
    
    def __init__(self):
        self.chart_height = 600
        self.colors = {
            'background': '#0E1117',
            'grid': '#262730',
            'text': '#FAFAFA',
            'bid_line': '#00D4AA',  # Green for bid
            'ask_line': '#FF6B6B',  # Red for ask
            'mid_price': '#FFD700', # Gold for mid price
            'volume': '#4CAF50',
            'buy_signal': '#00FF00',
            'sell_signal': '#FF0000'
        }
    
    def create_live_price_chart(self, price_data: pd.DataFrame, symbol: str, 
                               bid_price: float = None, ask_price: float = None,
                               live_price: float = None) -> go.Figure:
        """Create live price chart with bid/ask lines and moving price."""
        
        fig = sp.make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(f'{symbol.upper()} Live Price Chart', 'Volume'),
            row_heights=[0.8, 0.2]
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=price_data['timestamp'],
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='Price',
                increasing_line_color='#00D4AA',
                decreasing_line_color='#FF6B6B',
                increasing_fillcolor='rgba(0, 212, 170, 0.3)',
                decreasing_fillcolor='rgba(255, 107, 107, 0.3)'
            ),
            row=1, col=1
        )
        
        # Current price line (mid price)
        if live_price:
            fig.add_hline(
                y=live_price,
                line=dict(color=self.colors['mid_price'], width=2, dash="solid"),
                annotation_text=f"Live: {live_price:.3f}",
                annotation_position="right",
                row=1, col=1
            )
        
        # Bid line
        if bid_price:
            fig.add_hline(
                y=bid_price,
                line=dict(color=self.colors['bid_line'], width=1, dash="dash"),
                annotation_text=f"Bid: {bid_price:.3f}",
                annotation_position="left",
                row=1, col=1
            )
        
        # Ask line
        if ask_price:
            fig.add_hline(
                y=ask_price,
                line=dict(color=self.colors['ask_line'], width=1, dash="dash"),
                annotation_text=f"Ask: {ask_price:.3f}",
                annotation_position="left",
                row=1, col=1
            )
        
        # Volume bars
        fig.add_trace(
            go.Bar(
                x=price_data['timestamp'],
                y=price_data['volume'],
                name='Volume',
                marker_color=self.colors['volume'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Chart styling
        fig.update_layout(
            title=f"{symbol.upper()} - Live Trading Chart",
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            height=self.chart_height,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                type='date',
                rangeslider=dict(visible=False)
            )
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=self.colors['grid'],
            showgrid=True,
            title_text="Time",
            row=2, col=1
        )
        
        fig.update_yaxes(
            gridcolor=self.colors['grid'],
            showgrid=True,
            title_text="Price",
            row=1, col=1
        )
        
        fig.update_yaxes(
            gridcolor=self.colors['grid'],
            showgrid=True,
            title_text="Volume",
            row=2, col=1
        )
        
        return fig
    
    def create_tick_chart(self, tick_history: List[Dict], symbol: str,
                         current_bid: float = None, current_ask: float = None) -> go.Figure:
        """Create real-time tick chart showing price movements."""
        
        if not tick_history:
            return go.Figure().add_annotation(
                text="No tick data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Convert tick history to DataFrame
        df = pd.DataFrame(tick_history)
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['price'],
                mode='lines',
                name='Live Price',
                line=dict(color=self.colors['mid_price'], width=2),
                hovertemplate='<b>Price:</b> %{y:.5f}<br><b>Time:</b> %{x}<extra></extra>'
            )
        )
        
        # Add bid/ask area if available
        if current_bid and current_ask:
            fig.add_hline(
                y=current_bid,
                line=dict(color=self.colors['bid_line'], width=1, dash="dash"),
                annotation_text=f"Bid: {current_bid:.5f}",
                annotation_position="right"
            )
            
            fig.add_hline(
                y=current_ask,
                line=dict(color=self.colors['ask_line'], width=1, dash="dash"),
                annotation_text=f"Ask: {current_ask:.5f}",
                annotation_position="right"
            )
            
            # Spread area
            fig.add_hrect(
                y0=current_bid, y1=current_ask,
                fillcolor="rgba(255, 255, 255, 0.1)",
                line_width=0,
                annotation_text="Spread",
                annotation_position="top left"
            )
        
        # Styling
        fig.update_layout(
            title=f"{symbol.upper()} - Live Tick Chart",
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            height=400,
            xaxis=dict(
                title="Time",
                type='date',
                gridcolor=self.colors['grid']
            ),
            yaxis=dict(
                title="Price",
                gridcolor=self.colors['grid']
            ),
            showlegend=True
        )
        
        return fig
    
    def create_price_depth_chart(self, symbol: str, bid_price: float, 
                                ask_price: float, spread: float) -> go.Figure:
        """Create a price depth visualization showing bid/ask levels."""
        
        fig = go.Figure()
        
        # Create depth levels around current price
        levels = 10
        price_range = spread * 5
        
        bid_levels = np.linspace(bid_price - price_range, bid_price, levels)
        ask_levels = np.linspace(ask_price, ask_price + price_range, levels)
        
        # Simulate depth data
        bid_volumes = np.random.exponential(1000, levels)
        ask_volumes = np.random.exponential(1000, levels)
        
        # Bid side (buy orders)
        fig.add_trace(
            go.Bar(
                x=-bid_volumes,
                y=bid_levels,
                orientation='h',
                name='Bids',
                marker_color=self.colors['bid_line'],
                opacity=0.7
            )
        )
        
        # Ask side (sell orders)
        fig.add_trace(
            go.Bar(
                x=ask_volumes,
                y=ask_levels,
                orientation='h',
                name='Asks',
                marker_color=self.colors['ask_line'],
                opacity=0.7
            )
        )
        
        # Current price line
        mid_price = (bid_price + ask_price) / 2
        fig.add_hline(
            y=mid_price,
            line=dict(color=self.colors['mid_price'], width=3),
            annotation_text=f"Current: {mid_price:.5f}"
        )
        
        fig.update_layout(
            title=f"{symbol.upper()} - Market Depth",
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            height=400,
            xaxis=dict(title="Volume", gridcolor=self.colors['grid']),
            yaxis=dict(title="Price", gridcolor=self.colors['grid']),
            barmode='overlay'
        )
        
        return fig
    
    def create_price_dashboard(self, symbol: str, price_data: Dict) -> go.Figure:
        """Create a comprehensive price dashboard with multiple views."""
        
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{symbol.upper()} Price Chart',
                'Market Depth',
                'Price Distribution',
                'Live Metrics'
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "table"}]
            ]
        )
        
        # Main price chart (top left)
        if 'historical' in price_data:
            df = price_data['historical']
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )
        
        # Market depth (top right)
        if 'bid' in price_data and 'ask' in price_data:
            bid_price = price_data['bid']
            ask_price = price_data['ask']
            
            # Simulate depth data
            depth_levels = 5
            bid_levels = np.linspace(bid_price * 0.999, bid_price, depth_levels)
            ask_levels = np.linspace(ask_price, ask_price * 1.001, depth_levels)
            volumes = np.random.exponential(1000, depth_levels)
            
            fig.add_trace(
                go.Bar(x=-volumes, y=bid_levels, orientation='h', 
                      name='Bids', marker_color=self.colors['bid_line']),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=volumes, y=ask_levels, orientation='h',
                      name='Asks', marker_color=self.colors['ask_line']),
                row=1, col=2
            )
        
        # Price distribution (bottom left)
        if 'tick_history' in price_data:
            prices = [tick['price'] for tick in price_data['tick_history']]
            fig.add_trace(
                go.Histogram(x=prices, name='Price Distribution',
                           marker_color=self.colors['volume']),
                row=2, col=1
            )
        
        # Live metrics table (bottom right)
        metrics_data = [
            ['Current Price', f"{price_data.get('current', 0):.5f}"],
            ['Bid Price', f"{price_data.get('bid', 0):.5f}"],
            ['Ask Price', f"{price_data.get('ask', 0):.5f}"],
            ['Spread', f"{price_data.get('spread', 0):.5f}"],
            ['Daily Change', f"{price_data.get('change', 0):.2f}%"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*metrics_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            height=800,
            showlegend=False
        )
        
        return fig
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import os
import threading
from signal_engine import SignalEngine
from visualization import ChartVisualizer
from performance_tracker import PerformanceTracker
from smooth_price_engine import SmoothPriceEngine
from realtime_candlestick_engine import RealtimeCandlestickEngine
from live_chart_visualizer import LiveChartVisualizer

# Configure page
st.set_page_config(
    page_title="Advanced Trading Signals - Synthetic Indices",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set Deriv API token from environment
os.environ['DERIV_API_TOKEN'] = 'V4FOmNuhORnxbHT'

# Initialize session state
if 'signal_engine' not in st.session_state:
    st.session_state.signal_engine = SignalEngine()
if 'performance_tracker' not in st.session_state:
    st.session_state.performance_tracker = PerformanceTracker()
if 'chart_visualizer' not in st.session_state:
    st.session_state.chart_visualizer = ChartVisualizer()
if 'live_chart_visualizer' not in st.session_state:
    st.session_state.live_chart_visualizer = LiveChartVisualizer()
if 'smooth_price_engine' not in st.session_state:
    st.session_state.smooth_price_engine = None
if 'candlestick_engine' not in st.session_state:
    st.session_state.candlestick_engine = None
if 'live_prices' not in st.session_state:
    st.session_state.live_prices = {}
if 'streaming_active' not in st.session_state:
    st.session_state.streaming_active = False

def main():
    st.title("🚀 Advanced Trading Signals Dashboard")
    st.subheader("Synthetic Indices Analysis with Mathematical Precision")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Symbol selection with API connectivity status
        symbol_mapping = {
            "Crash 300 ✓": "crash300",
            "Crash 1000": "crash1000",
            "Boom 300": "boom300", 
            "Boom 1000": "boom1000"
        }
        
        selected_display = st.selectbox(
            "Select Synthetic Index",
            list(symbol_mapping.keys()),
            index=0,
            help="✓ indicates confirmed API connectivity"
        )
        
        symbol = symbol_mapping[selected_display]
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Analysis Timeframe",
            ["1m", "5m", "15m", "1h"],
            index=0
        )
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        confidence_threshold = st.slider(
            "Minimum Signal Confidence %",
            min_value=60,
            max_value=95,
            value=75,
            step=5
        )
        
        lookback_periods = st.number_input(
            "Lookback Periods",
            min_value=50,
            max_value=500,
            value=200,
            step=50
        )
        
        # Live streaming controls
        st.subheader("Live Price Streaming")
        
        # Show connection status
        if st.session_state.smooth_price_engine:
            status = st.session_state.smooth_price_engine.get_connection_status()
            if status['connected']:
                st.success("🟢 Connected to live data feed")
                st.write(f"Active symbols: {len(status['active_symbols'])}")
                if status['smoothing_enabled']:
                    st.info("🔄 Smooth price transitions enabled")
            else:
                st.warning("🟡 Connecting to data feed...")
        
        live_streaming = st.checkbox("Enable Live Price Stream", value=False)
        auto_refresh = st.checkbox("Auto Refresh Charts", value=False)
        
        # Chart type and settings (initialize outside conditional)
        chart_type = "Smooth Price Line"
        timeframe_seconds = 60
        
        if live_streaming:
            st.subheader("Chart Type")
            chart_type = st.radio(
                "Select chart display",
                ["Real-time Candlesticks with Bid/Ask", "Smooth Price Line"],
                index=0,
                help="Choose between candlestick formation or smooth price movements"
            )
            
            if chart_type == "Smooth Price Line":
                st.subheader("Smoothing Settings")
                smoothing_enabled = st.checkbox("Enable Price Smoothing", value=True)
                if smoothing_enabled:
                    update_frequency = st.slider("Update Frequency (ms)", 100, 1000, 200, 50)
                    interpolation_steps = st.slider("Smoothing Steps", 3, 10, 5)
                    
                    if st.session_state.smooth_price_engine:
                        st.session_state.smooth_price_engine.set_smoothing(
                            smoothing_enabled, interpolation_steps, update_frequency
                        )
            else:
                st.subheader("Candlestick Settings")
                timeframe_seconds = st.selectbox(
                    "Candle Timeframe",
                    [30, 60, 120, 300],
                    index=1,
                    format_func=lambda x: f"{x//60}m" if x >= 60 else f"{x}s"
                )
        
        # Streaming controls
        if live_streaming:
            if not st.session_state.streaming_active:
                if st.button("Start Live Stream"):
                    success = False
                    
                    if chart_type == "Real-time Candlesticks with Bid/Ask":
                        # Initialize candlestick engine
                        if st.session_state.candlestick_engine is None:
                            st.session_state.candlestick_engine = RealtimeCandlestickEngine(
                                timeframe_seconds=timeframe_seconds
                            )
                        
                        # Start candlestick engine
                        if st.session_state.candlestick_engine.start():
                            def candle_callback(symbol, candle):
                                st.session_state.live_prices[symbol] = {
                                    'price': candle.close,
                                    'open': candle.open,
                                    'high': candle.high,
                                    'low': candle.low,
                                    'timestamp': candle.timestamp,
                                    'is_forming': candle.is_forming,
                                    'chart_type': 'candlestick'
                                }
                            
                            def bid_ask_callback(symbol, bid_ask):
                                if symbol in st.session_state.live_prices:
                                    st.session_state.live_prices[symbol].update({
                                        'bid': bid_ask.bid,
                                        'ask': bid_ask.ask,
                                        'spread': bid_ask.spread
                                    })
                            
                            # Subscribe to candlestick and bid/ask data
                            if st.session_state.candlestick_engine.subscribe_symbol(
                                symbol, candle_callback, bid_ask_callback
                            ):
                                success = True
                                st.success(f"Started real-time candlesticks for {symbol}")
                        
                    else:
                        # Initialize smooth price engine
                        if st.session_state.smooth_price_engine is None:
                            st.session_state.smooth_price_engine = SmoothPriceEngine()
                        
                        # Start smooth price engine
                        if st.session_state.smooth_price_engine.start():
                            def price_callback(symbol, market_state):
                                st.session_state.live_prices[symbol] = {
                                    'price': market_state.current_price,
                                    'bid': market_state.bid,
                                    'ask': market_state.ask,
                                    'spread': market_state.spread,
                                    'timestamp': market_state.last_update,
                                    'trend': market_state.trend,
                                    'volatility': market_state.volatility,
                                    'chart_type': 'smooth'
                                }
                            
                            # Subscribe to smooth price data
                            if st.session_state.smooth_price_engine.subscribe_symbol(symbol, price_callback):
                                success = True
                                st.success(f"Started smooth live streaming for {symbol}")
                    
                    if success:
                        st.session_state.streaming_active = True
                        st.rerun()
                    else:
                        st.error("Failed to start live streaming")
            else:
                if st.button("Stop Live Stream"):
                    # Stop both engines
                    if st.session_state.smooth_price_engine:
                        st.session_state.smooth_price_engine.unsubscribe_symbol(symbol)
                        st.session_state.smooth_price_engine.stop()
                    if st.session_state.candlestick_engine:
                        st.session_state.candlestick_engine.unsubscribe_symbol(symbol)
                        st.session_state.candlestick_engine.stop()
                    
                    st.session_state.streaming_active = False
                    st.info("Live streaming stopped")
                    st.rerun()
        
        # Connection status indicator
        if st.button("Test API Connection"):
            with st.spinner("Testing Deriv API connection..."):
                from test_deriv_connection import test_deriv_connection
                connection_success = test_deriv_connection()
                if connection_success:
                    st.success("Connected to Deriv API")
                else:
                    st.error("Connection failed")
        
        if st.button("🔄 Manual Refresh"):
            st.rerun()
    
    # Main dashboard layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"📊 {symbol} - {timeframe} Analysis")
        
        # Generate signals
        with st.spinner("Analyzing market data and generating signals..."):
            try:
                signal_data = st.session_state.signal_engine.generate_signals(
                    symbol=symbol.lower().replace(" ", ""),
                    timeframe=timeframe,
                    lookback_periods=lookback_periods,
                    confidence_threshold=confidence_threshold
                )
                
                if signal_data and 'price_data' in signal_data and not signal_data['price_data'].empty:
                    # Display live price info if streaming is active
                    if st.session_state.streaming_active and symbol in st.session_state.live_prices:
                        live_data = st.session_state.live_prices[symbol]
                        chart_type = live_data.get('chart_type', 'smooth')
                        
                        # Live price ticker
                        if chart_type == 'candlestick':
                            # Candlestick metrics
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                candle_status = "🔴 Forming" if live_data.get('is_forming', False) else "🟢 Completed"
                                st.metric("Current Price", f"{live_data['price']:.5f}", 
                                        delta=candle_status, delta_color="off")
                            with col2:
                                st.metric("Open", f"{live_data.get('open', 0):.5f}")
                            with col3:
                                st.metric("High", f"{live_data.get('high', 0):.5f}")
                            with col4:
                                st.metric("Low", f"{live_data.get('low', 0):.5f}")
                            with col5:
                                if 'spread' in live_data:
                                    st.metric("Spread", f"{live_data['spread']:.5f}")
                            
                            # Bid/Ask lines
                            if 'bid' in live_data and 'ask' in live_data:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Bid", f"{live_data['bid']:.5f}", 
                                            delta=f"-{live_data.get('spread', 0):.5f}")
                                with col2:
                                    st.metric("Ask", f"{live_data['ask']:.5f}", 
                                            delta=f"+{live_data.get('spread', 0):.5f}")
                            
                            # Real-time candlestick chart with bid/ask lines
                            if st.session_state.candlestick_engine:
                                candlestick_chart = st.session_state.candlestick_engine.create_live_candlestick_chart(symbol)
                                st.plotly_chart(candlestick_chart, use_container_width=True)
                                
                                # Connection status for candlesticks
                                status = st.session_state.candlestick_engine.get_connection_status()
                                if status['connected']:
                                    st.success("🟢 Real-time candlestick streaming active")
                                    st.info(f"Timeframe: {status['timeframe_seconds']}s | Active symbols: {len(status['active_symbols'])}")
                        else:
                            # Smooth price metrics
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                trend_indicator = {"up": "↗", "down": "↘", "neutral": "→"}.get(live_data.get('trend', 'neutral'), "→")
                                st.metric("Live Price", f"{live_data['price']:.5f}", 
                                        delta=trend_indicator, delta_color="off")
                            with col2:
                                st.metric("Bid", f"{live_data['bid']:.5f}", 
                                        delta=f"-{live_data['spread']:.5f}")
                            with col3:
                                st.metric("Ask", f"{live_data['ask']:.5f}", 
                                        delta=f"+{live_data['spread']:.5f}")
                            with col4:
                                st.metric("Spread", f"{live_data['spread']:.5f}")
                            with col5:
                                volatility = live_data.get('volatility', 0)
                                st.metric("Volatility", f"{volatility:.6f}")
                            
                            # Smooth price chart
                            live_chart = st.session_state.live_chart_visualizer.create_live_price_chart(
                                signal_data['price_data'],
                                symbol,
                                bid_price=live_data['bid'],
                                ask_price=live_data['ask'],
                                live_price=live_data['price']
                            )
                            st.plotly_chart(live_chart, use_container_width=True)
                            
                            # Connection status for smooth streaming
                            if st.session_state.smooth_price_engine:
                                status = st.session_state.smooth_price_engine.get_connection_status()
                                if status['pending_updates'] == 0:
                                    st.success("🟢 Smooth streaming - No interruptions")
                                else:
                                    st.info(f"🔄 Smoothing {status['pending_updates']} transitions")
                    else:
                        # Standard chart when streaming is not active
                        chart = st.session_state.chart_visualizer.create_main_chart(
                            signal_data['price_data'],
                            signal_data['signals'],
                            signal_data['indicators'],
                            symbol,
                            timeframe
                        )
                        st.plotly_chart(chart, use_container_width=True)
                    
                    # Technical indicators subplot
                    indicators_chart = st.session_state.chart_visualizer.create_indicators_chart(
                        signal_data['price_data'],
                        signal_data['indicators']
                    )
                    st.plotly_chart(indicators_chart, use_container_width=True)
                    
                else:
                    if symbol in ['crash300', 'boom300']:
                        st.error("Connection issue with Deriv API for real market data.")
                        st.info("The API connection is working for this symbol but data retrieval failed. This may be a temporary network issue.")
                    else:
                        st.warning("Limited API access for this synthetic index.")
                        st.info("This symbol may require additional API permissions or a different access token. Crash 300 and Boom 300 are confirmed to work with your current token.")
                    
                    # Display connection status
                    with st.expander("Connection Details"):
                        if symbol in ['crash300', 'boom300']:
                            st.write("**Status:** API Connected (temporary data issue)")
                            st.write("**Symbol Access:** Confirmed working")
                            st.write("**Action:** Retry connection or check network")
                        else:
                            st.write("**Status:** Limited symbol access")
                            st.write("**Available Symbols:** Crash 300, Boom 300") 
                            st.write("**Note:** This symbol may require additional API permissions")
                    
                    signal_data = None
                    
            except Exception as e:
                st.error(f"❌ Error generating signals: {str(e)}")
                st.info("Please check your configuration and try again.")
                signal_data = None
    
    with col2:
        st.subheader("🎯 Live Signals")
        
        if signal_data and 'signals' in signal_data and not signal_data['signals'].empty:
            current_signals = signal_data['signals'].tail(5)
            
            # Display symbol information at the top
            st.markdown(f"**Symbol:** {symbol}")
            st.markdown(f"**Timeframe:** {timeframe}")
            st.markdown("---")
            
            for _, signal in current_signals.iterrows():
                signal_type = signal.get('signal_type', 'HOLD')
                confidence = signal.get('confidence', 0)
                timestamp = signal.get('timestamp', datetime.datetime.now())
                signal_source = signal.get('signal_source', 'combined')
                
                # Signal styling based on type
                if signal_type == 'BUY':
                    st.success(f"🟢 **BUY SIGNAL** - {symbol}")
                elif signal_type == 'SELL':
                    st.error(f"🔴 **SELL SIGNAL** - {symbol}")
                else:
                    st.info(f"🟡 **HOLD** - {symbol}")
                
                st.write(f"**Confidence:** {confidence:.1f}%")
                st.write(f"**Source:** {signal_source.title()}")
                if isinstance(timestamp, datetime.datetime):
                    st.write(f"**Time:** {timestamp.strftime('%H:%M:%S')}")
                else:
                    st.write(f"**Time:** {str(timestamp)}")
                st.write("---")
                
        else:
            st.info("📊 Analyzing market conditions...")
            st.write(f"Symbol: {symbol}")
            st.write(f"Timeframe: {timeframe}")
            st.write("No signals available yet. Market analysis in progress.")
    
    with col3:
        st.subheader("📈 Performance Metrics")
        
        if signal_data:
            # Performance summary
            performance_data = st.session_state.performance_tracker.calculate_performance(signal_data)
            
            if performance_data:
                st.metric(
                    "Signal Accuracy",
                    f"{performance_data.get('accuracy', 0):.1f}%",
                    delta=f"{performance_data.get('accuracy_change', 0):.1f}%"
                )
                
                st.metric(
                    "Total Signals",
                    performance_data.get('total_signals', 0),
                    delta=performance_data.get('new_signals', 0)
                )
                
                st.metric(
                    "Avg Confidence",
                    f"{performance_data.get('avg_confidence', 0):.1f}%",
                    delta=f"{performance_data.get('confidence_change', 0):.1f}%"
                )
                
                # Win/Loss ratio
                if performance_data.get('total_signals', 0) > 0:
                    win_rate = performance_data.get('winning_signals', 0) / performance_data.get('total_signals', 1) * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
            else:
                st.info("📊 Performance data will be available after sufficient signal history.")
        else:
            st.info("⏳ Waiting for signal data...")
    
    # Advanced Analytics Section
    st.header("🔬 Advanced Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Pattern Detection", "Trend Analysis", "Statistical Validation", "Momentum Analysis"])
    
    with tab1:
        st.subheader("Pattern Recognition")
        if signal_data and 'patterns' in signal_data:
            patterns_df = signal_data['patterns']
            if not patterns_df.empty:
                st.dataframe(patterns_df, use_container_width=True)
            else:
                st.info("No significant patterns detected in current timeframe.")
        else:
            st.info("Pattern analysis requires market data. Please wait for data loading.")
    
    with tab2:
        st.subheader("Trend Line Analysis")
        if signal_data and 'trend_lines' in signal_data:
            trend_data = signal_data['trend_lines']
            if trend_data:
                for trend in trend_data:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**{trend['type']} Trend**")
                        st.write(f"Strength: {trend['strength']:.2f}")
                    with col_b:
                        st.write(f"Slope: {trend['slope']:.4f}")
                        st.write(f"R²: {trend['r_squared']:.3f}")
            else:
                st.info("No significant trend lines identified.")
        else:
            st.info("Trend analysis requires sufficient price history.")
    
    with tab3:
        st.subheader("Statistical Validation")
        if signal_data and 'statistics' in signal_data:
            stats = signal_data['statistics']
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("P-Value", f"{stats.get('p_value', 0):.4f}")
                st.metric("Volatility", f"{stats.get('volatility', 0):.4f}")
                
            with col_stat2:
                st.metric("Sharpe Ratio", f"{stats.get('sharpe_ratio', 0):.3f}")
                st.metric("Signal Strength", f"{stats.get('signal_strength', 0):.3f}")
        else:
            st.info("Statistical validation requires historical signal data.")
    
    with tab4:
        st.subheader("Momentum Indicators")
        if signal_data and 'momentum' in signal_data:
            momentum_data = signal_data['momentum']
            
            # Create momentum chart
            momentum_chart = st.session_state.chart_visualizer.create_momentum_chart(momentum_data)
            st.plotly_chart(momentum_chart, use_container_width=True)
            
            # Momentum metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Current Momentum", f"{momentum_data.get('current_momentum', 0):.3f}")
            with col_m2:
                st.metric("Momentum Change", f"{momentum_data.get('momentum_change', 0):.3f}")
            with col_m3:
                st.metric("Acceleration", f"{momentum_data.get('acceleration', 0):.4f}")
        else:
            st.info("Momentum analysis requires price movement data.")
    
    # Auto-refresh mechanism for live charts
    if auto_refresh or st.session_state.streaming_active:
        # Faster refresh for live streaming, slower for regular auto-refresh
        refresh_interval = 3 if st.session_state.streaming_active else 10
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()

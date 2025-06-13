"""
Comprehensive list of all chart patterns and technical indicators
available in the Advanced Trading Signals system.
"""

def get_all_indicators_and_patterns():
    """Return comprehensive list of all available indicators and patterns."""
    
    indicators_and_patterns = {
        "Technical Indicators": {
            "Trend Indicators": [
                "Simple Moving Average (SMA) - 20, 50, 200 periods",
                "Exponential Moving Average (EMA) - 12, 26 periods", 
                "Moving Average Convergence Divergence (MACD) - 12, 26, 9",
                "Average Directional Index (ADX) - 14 periods",
                "Parabolic SAR - acceleration factor 0.02",
                "Ichimoku Cloud - 9, 26, 52 periods",
                "Trend Line Analysis - automatic detection",
                "Linear Regression Trend - multiple periods"
            ],
            
            "Momentum Oscillators": [
                "Relative Strength Index (RSI) - 14 periods",
                "Stochastic Oscillator - %K 14, %D 3",
                "Williams %R - 14 periods", 
                "Commodity Channel Index (CCI) - 20 periods",
                "Rate of Change (ROC) - 12 periods",
                "Momentum Index - 14 periods",
                "Chande Momentum Oscillator (CMO) - 14 periods",
                "Price Momentum Oscillator (PMO) - 35, 20 periods"
            ],
            
            "Volatility Indicators": [
                "Bollinger Bands - 20 periods, 2 standard deviations",
                "Average True Range (ATR) - 14 periods",
                "Bollinger Band Width - volatility measure",
                "Bollinger Band %B - position within bands",
                "Volatility Index - custom calculation",
                "Price Acceleration - velocity and acceleration metrics"
            ],
            
            "Volume Indicators": [
                "Volume Moving Average - 20 periods",
                "Volume Rate of Change - price-volume relationship",
                "Volume Momentum - volume-based momentum",
                "On-Balance Volume (OBV) - cumulative volume",
                "Volume Weighted Average Price (VWAP) - intraday"
            ]
        },
        
        "Chart Patterns": {
            "Reversal Patterns": [
                "Head and Shoulders - bearish reversal",
                "Inverse Head and Shoulders - bullish reversal", 
                "Double Top - bearish reversal at resistance",
                "Double Bottom - bullish reversal at support",
                "Triple Top - strong bearish reversal",
                "Triple Bottom - strong bullish reversal",
                "Rising Wedge - bearish reversal pattern",
                "Falling Wedge - bullish reversal pattern"
            ],
            
            "Continuation Patterns": [
                "Ascending Triangle - bullish continuation",
                "Descending Triangle - bearish continuation", 
                "Symmetrical Triangle - neutral continuation",
                "Flag Pattern - trend continuation after sharp move",
                "Pennant Pattern - small symmetrical triangle",
                "Rectangle/Channel - horizontal price movement",
                "Cup and Handle - bullish continuation"
            ],
            
            "Candlestick Patterns": [
                "Doji - indecision pattern",
                "Hammer - bullish reversal at support",
                "Hanging Man - bearish reversal at resistance",
                "Bullish Engulfing - strong bullish signal",
                "Bearish Engulfing - strong bearish signal", 
                "Morning Star - three-candle bullish reversal",
                "Evening Star - three-candle bearish reversal",
                "Shooting Star - bearish reversal at top",
                "Inverted Hammer - potential bullish reversal"
            ]
        },
        
        "Support and Resistance": {
            "Level Detection": [
                "Horizontal Support Levels - price floors",
                "Horizontal Resistance Levels - price ceilings",
                "Dynamic Support/Resistance - moving averages",
                "Pivot Points - daily, weekly, monthly",
                "Fibonacci Retracement Levels - 23.6%, 38.2%, 50%, 61.8%",
                "Fibonacci Extension Levels - projection targets",
                "Psychological Levels - round numbers",
                "Volume Profile Levels - high volume areas"
            ]
        },
        
        "Advanced Analysis": {
            "Mathematical Models": [
                "Linear Regression Analysis - trend direction",
                "Polynomial Regression - curve fitting",
                "Statistical Validation - confidence intervals",
                "Correlation Analysis - price relationships",
                "Standard Deviation Channels - volatility bands",
                "Z-Score Analysis - statistical deviation",
                "Monte Carlo Simulation - probability analysis"
            ],
            
            "Pattern Recognition": [
                "Automatic Peak and Trough Detection",
                "Trend Line Break Detection", 
                "Support/Resistance Break Confirmation",
                "Pattern Completion Signals",
                "Divergence Detection - price vs indicators",
                "Confluence Analysis - multiple signal alignment",
                "Signal Strength Scoring - confidence levels"
            ]
        },
        
        "Risk Management": {
            "Risk Metrics": [
                "Position Size Calculator - based on risk percentage",
                "Stop Loss Recommendations - ATR-based",
                "Take Profit Targets - risk-reward ratios",
                "Maximum Drawdown Analysis",
                "Sharpe Ratio Calculation",
                "Win Rate Statistics",
                "Average Risk-Reward Ratio"
            ]
        },
        
        "Performance Tracking": {
            "Signal Analysis": [
                "Signal Accuracy Percentage",
                "Average Hold Time",
                "Profitability Metrics",
                "Signal Frequency Analysis",
                "Confidence Score Distribution",
                "Success Rate by Pattern Type",
                "Market Condition Performance"
            ]
        }
    }
    
    return indicators_and_patterns

def print_comprehensive_list():
    """Print formatted list of all indicators and patterns."""
    data = get_all_indicators_and_patterns()
    
    print("=" * 80)
    print("COMPREHENSIVE TRADING INDICATORS & PATTERNS LIST")
    print("Advanced Trading Signals System - Synthetic Indices")
    print("=" * 80)
    
    for category, subcategories in data.items():
        print(f"\n🔸 {category.upper()}")
        print("-" * 60)
        
        if isinstance(subcategories, dict):
            for subcat, items in subcategories.items():
                print(f"\n  📊 {subcat}")
                for item in items:
                    print(f"    • {item}")
        else:
            for item in subcategories:
                print(f"    • {item}")
    
    print("\n" + "=" * 80)
    print("SIGNAL GENERATION FEATURES:")
    print("=" * 80)
    print("• Multi-timeframe Analysis (1m, 5m, 15m, 1h)")
    print("• Real-time Price Streaming with Bid/Ask spreads") 
    print("• Confidence Scoring (60-95% threshold)")
    print("• Signal Combination and Weighting")
    print("• Live Performance Tracking")
    print("• Risk-adjusted Position Sizing")
    print("• Automated Pattern Recognition")
    print("• Statistical Validation of Signals")
    print("• Divergence Detection and Analysis")
    print("• Market Condition Assessment")
    
    print("\n" + "=" * 80)
    print("SUPPORTED SYNTHETIC INDICES:")
    print("=" * 80)
    print("✓ Crash 300 (CRASH300N) - High volatility crash index")
    print("✓ Boom 300 (BOOM300N) - High volatility boom index") 
    print("✓ Crash 1000 (CRASH1000) - Moderate volatility crash index")
    print("✓ Boom 1000 (BOOM1000) - Moderate volatility boom index")
    
    return data

if __name__ == "__main__":
    print_comprehensive_list()
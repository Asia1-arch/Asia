import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
import os
from deriv_api import DerivAPI

class DataProcessor:
    """
    Data processor for handling market data for synthetic indices.
    In production, this would connect to real data feeds.
    For this implementation, it generates realistic synthetic data for demonstration.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize Deriv API client
        self.deriv_api = DerivAPI()
        self.api_connected = False
        
        # Symbol configurations for metadata
        self.symbol_configs = {
            'crash300': {
                'deriv_symbol': 'R_10',
                'description': 'Crash 300 Index',
                'tick_size': 0.01
            },
            'boom300': {
                'deriv_symbol': 'R_25',
                'description': 'Boom 300 Index', 
                'tick_size': 0.01
            },
            'crash1000': {
                'deriv_symbol': 'R_100',
                'description': 'Crash 1000 Index',
                'tick_size': 0.01
            },
            'boom1000': {
                'deriv_symbol': 'R_75',
                'description': 'Boom 1000 Index',
                'tick_size': 0.01
            }
        }
        
        # Timeframe configurations
        self.timeframe_configs = {
            '1m': {'minutes': 1, 'periods_per_hour': 60},
            '5m': {'minutes': 5, 'periods_per_hour': 12},
            '15m': {'minutes': 15, 'periods_per_hour': 4},
            '1h': {'minutes': 60, 'periods_per_hour': 1}
        }
        
        # Try to connect to Deriv API
        self._initialize_api_connection()
    
    def _initialize_api_connection(self):
        """Initialize connection to Deriv API."""
        try:
            self.api_connected = self.deriv_api.connect()
            if self.api_connected:
                self.logger.info("Successfully connected to Deriv API")
            else:
                self.logger.warning("Failed to connect to Deriv API, will use fallback data")
        except Exception as e:
            self.logger.error(f"Error connecting to Deriv API: {str(e)}")
            self.api_connected = False
    
    def get_market_data(self, symbol: str, timeframe: str, periods: int = 200) -> Optional[pd.DataFrame]:
        """
        Get market data for the specified symbol and timeframe.
        
        Args:
            symbol (str): Trading symbol (e.g., 'crash300', 'boom1000')
            timeframe (str): Time interval ('1m', '5m', '15m', '1h')
            periods (int): Number of periods to retrieve
            
        Returns:
            pd.DataFrame: OHLCV data with timestamp
        """
        try:
            # First try to get real data from Deriv API
            if self.api_connected and self.deriv_api.is_connected:
                real_data = self._get_real_market_data(symbol, timeframe, periods)
                if real_data is not None and not real_data.empty:
                    self.logger.info(f"Retrieved real market data for {symbol}")
                    return real_data
            
            # If API is not available, check if we have cached data
            cached_data = self.deriv_api.get_cached_data(symbol)
            if cached_data is not None and not cached_data.empty:
                self.logger.info(f"Using cached data for {symbol}")
                return cached_data.tail(periods)
            
            # If no real data available, return None to signal data unavailability
            self.logger.warning(f"No real market data available for {symbol} - may require additional API permissions")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None
    
    def _get_real_market_data(self, symbol: str, timeframe: str, periods: int) -> Optional[pd.DataFrame]:
        """Get real market data from Deriv API with connection retry."""
        import time
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Force fresh connection for reliability
                if hasattr(self.deriv_api, 'disconnect'):
                    self.deriv_api.disconnect()
                    time.sleep(1)
                
                # Get historical data from Deriv
                data = self.deriv_api.get_historical_data(symbol, timeframe, periods)
                
                if data is not None and not data.empty:
                    # Validate and clean the data
                    if self.validate_data(data):
                        self.logger.info(f"Retrieved real market data for {symbol}")
                        return self.clean_data(data)
                    else:
                        self.logger.warning(f"Invalid data received for {symbol}")
                        
                if attempt < max_attempts - 1:
                    self.logger.warning(f"Retrying data fetch for {symbol} (attempt {attempt + 2})")
                    time.sleep(2)
                    
            except Exception as e:
                self.logger.error(f"Error fetching real market data for {symbol} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(2)
        
        self.logger.error(f"Failed to retrieve real market data for {symbol} after {max_attempts} attempts")
        return None
    
    def _generate_synthetic_data(self, symbol: str, timeframe: str, periods: int) -> pd.DataFrame:
        """Generate realistic synthetic market data for testing and demonstration."""
        
        if symbol not in self.symbol_configs:
            self.logger.warning(f"Unknown symbol: {symbol}")
            # Return empty DataFrame to trigger proper error handling in the UI
            return pd.DataFrame()
        
        if timeframe not in self.timeframe_configs:
            self.logger.warning(f"Unknown timeframe: {timeframe}")
            return pd.DataFrame()
        
        config = self.symbol_configs[symbol]
        tf_config = self.timeframe_configs[timeframe]
        
        # Generate timestamps
        now = datetime.now()
        interval_minutes = tf_config['minutes']
        timestamps = []
        
        for i in range(periods):
            timestamp = now - timedelta(minutes=interval_minutes * (periods - i - 1))
            timestamps.append(timestamp)
        
        # Generate price data with realistic characteristics
        np.random.seed(42)  # For reproducible results
        
        base_price = config['base_price']
        volatility = config['volatility']
        
        # Generate returns using geometric Brownian motion
        dt = interval_minutes / (24 * 60)  # Convert to days
        drift = 0.0001  # Small positive drift
        
        returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), periods)
        
        # Add crash/boom events for synthetic indices
        if 'crash' in symbol:
            crash_prob = config['crash_probability']
            crash_magnitude = config['crash_magnitude']
            
            for i in range(len(returns)):
                if np.random.random() < crash_prob:
                    returns[i] = crash_magnitude
                    
        elif 'boom' in symbol:
            boom_prob = config['boom_probability']
            boom_magnitude = config['boom_magnitude']
            
            for i in range(len(returns)):
                if np.random.random() < boom_prob:
                    returns[i] = boom_magnitude
        
        # Calculate prices
        prices = [base_price]
        for i in range(1, periods):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        # Generate OHLCV data
        data = []
        
        for i in range(periods):
            # Generate intraperiod price movements
            open_price = prices[i]
            close_price = prices[i]
            
            # Add some intraperiod volatility
            intraperiod_volatility = volatility * 0.3
            high_multiplier = 1 + abs(np.random.normal(0, intraperiod_volatility))
            low_multiplier = 1 - abs(np.random.normal(0, intraperiod_volatility))
            
            high_price = max(open_price, close_price) * high_multiplier
            low_price = min(open_price, close_price) * low_multiplier
            
            # Generate realistic volume
            base_volume = 1000
            volume_multiplier = np.random.lognormal(0, 0.5)
            volume = int(base_volume * volume_multiplier)
            
            data.append({
                'timestamp': timestamps[i],
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate market data for completeness and consistency.
        
        Args:
            data (pd.DataFrame): Market data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if data is None or data.empty:
            return False
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check if all required columns are present
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for null values
        if data[required_columns].isnull().any().any():
            self.logger.error("Data contains null values")
            return False
        
        # Check price consistency (high >= low, etc.)
        price_checks = (
            (data['high'] >= data['low']) &
            (data['high'] >= data['open']) &
            (data['high'] >= data['close']) &
            (data['low'] <= data['open']) &
            (data['low'] <= data['close']) &
            (data['volume'] >= 0)
        )
        
        if not price_checks.all():
            self.logger.error("Data contains inconsistent price values")
            return False
        
        # Check timestamp ordering
        if not data['timestamp'].is_monotonic_increasing:
            self.logger.error("Timestamps are not in ascending order")
            return False
        
        return True
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess market data.
        
        Args:
            data (pd.DataFrame): Raw market data
            
        Returns:
            pd.DataFrame: Cleaned market data
        """
        if data is None or data.empty:
            return data
        
        # Remove duplicates
        data = data.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Forward fill missing values (if any)
        price_columns = ['open', 'high', 'low', 'close']
        data[price_columns] = data[price_columns].fillna(method='ffill')
        
        # Fill volume with 0 if missing
        data['volume'] = data['volume'].fillna(0)
        
        # Remove any remaining rows with null values
        data = data.dropna().reset_index(drop=True)
        
        return data
    
    def resample_data(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            data (pd.DataFrame): Original market data
            target_timeframe (str): Target timeframe ('1m', '5m', '15m', '1h')
            
        Returns:
            pd.DataFrame: Resampled data
        """
        if data is None or data.empty:
            return data
        
        # Map timeframe to pandas frequency
        timeframe_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '1h': '1H'
        }
        
        if target_timeframe not in timeframe_map:
            self.logger.error(f"Unknown target timeframe: {target_timeframe}")
            return data
        
        frequency = timeframe_map[target_timeframe]
        
        # Set timestamp as index for resampling
        data_copy = data.copy()
        data_copy.set_index('timestamp', inplace=True)
        
        # Resample OHLCV data
        resampled = data_copy.resample(frequency).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Reset index to get timestamp back as column
        resampled.reset_index(inplace=True)
        
        return resampled
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get information about a trading symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Dict: Symbol information
        """
        if symbol in self.symbol_configs:
            config = self.symbol_configs[symbol].copy()
            
            # Add additional metadata
            config.update({
                'symbol': symbol,
                'description': self._get_symbol_description(symbol),
                'min_tick': 0.01,
                'contract_size': 1,
                'currency': 'USD'
            })
            
            return config
        
        return {}
    
    def _get_symbol_description(self, symbol: str) -> str:
        """Get human-readable description for a symbol."""
        descriptions = {
            'crash300': 'Crash 300 Index - High volatility synthetic index with crash events',
            'boom300': 'Boom 300 Index - High volatility synthetic index with boom events',
            'crash1000': 'Crash 1000 Index - Very high volatility synthetic index with crash events',
            'boom1000': 'Boom 1000 Index - Very high volatility synthetic index with boom events'
        }
        
        return descriptions.get(symbol, f'{symbol.upper()} - Synthetic Index')
    
    def calculate_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate basic statistics for market data.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            Dict: Statistical measures
        """
        if data is None or data.empty:
            return {}
        
        try:
            close_prices = data['close']
            returns = close_prices.pct_change().dropna()
            
            statistics = {
                'current_price': float(close_prices.iloc[-1]),
                'price_change': float(close_prices.iloc[-1] - close_prices.iloc[0]),
                'price_change_pct': float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100),
                'volatility': float(returns.std() * np.sqrt(252)),  # Annualized
                'max_price': float(close_prices.max()),
                'min_price': float(close_prices.min()),
                'avg_price': float(close_prices.mean()),
                'avg_volume': float(data['volume'].mean()) if 'volume' in data.columns else 0,
                'total_volume': float(data['volume'].sum()) if 'volume' in data.columns else 0
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            return {}

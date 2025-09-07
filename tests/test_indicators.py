# backend/tests/test_indicators.py
import pytest
import numpy as np
from app.core.indicators.atr import calculate_atr, ATRIndicator
from app.core.indicators.rsi import calculate_rsi, RSIIndicator
from app.core.indicators.adx import calculate_adx, ADXIndicator
from app.core.indicators.vwap import calculate_vwap, VWAPIndicator
from app.core.indicators.volatility import calculate_realized_volatility, VolatilityIndicator

class TestATRIndicator:
    """Test ATR calculation"""
    
    def test_calculate_atr_basic(self):
        """Test basic ATR calculation"""
        highs = [10, 12, 11, 13, 12]
        lows = [8, 9, 8, 10, 9]
        closes = [9, 11, 10, 12, 11]
        
        atr_values = calculate_atr(highs, lows, closes, period=3)
        
        assert len(atr_values) > 0
        assert all(atr > 0 for atr in atr_values)
    
    def test_atr_indicator_stateful(self):
        """Test stateful ATR indicator"""
        atr = ATRIndicator(period=3)
        
        # Test data
        data = [
            (10, 8, 9),   # high, low, close
            (12, 9, 11),
            (11, 8, 10),
            (13, 10, 12),
            (12, 9, 11)
        ]
        
        results = []
        for high, low, close in data:
            result = atr.update(high, low, close)
            if result is not None:
                results.append(result)
        
        assert len(results) > 0
        assert all(isinstance(r, float) and r > 0 for r in results)
    
    def test_atr_indicator_insufficient_data(self):
        """Test ATR indicator with insufficient data"""
        atr = ATRIndicator(period=14)
        
        # Only one data point
        result = atr.update(100, 95, 98)
        assert result is None  # Should return None for insufficient data

class TestRSIIndicator:
    """Test RSI calculation"""
    
    def test_calculate_rsi_basic(self):
        """Test basic RSI calculation"""
        # Generate price series with trend
        prices = [100]
        for i in range(20):
            prices.append(prices[-1] + np.random.uniform(-2, 3))  # Slight upward trend
        
        rsi_values = calculate_rsi(prices, period=14)
        
        assert len(rsi_values) > 0
        assert all(0 <= rsi <= 100 for rsi in rsi_values)
    
    def test_rsi_indicator_stateful(self):
        """Test stateful RSI indicator"""
        rsi = RSIIndicator(period=5)  # Shorter period for testing
        
        # Generate test prices
        prices = [100, 102, 101, 103, 104, 102, 105, 106, 104, 107]
        
        results = []
        for price in prices:
            result = rsi.update(price)
            if result is not None:
                results.append(result)
        
        assert len(results) > 0
        assert all(0 <= r <= 100 for r in results)
    
    def test_rsi_extreme_values(self):
        """Test RSI with extreme price movements"""
        rsi = RSIIndicator(period=5)
        
        # All increasing prices
        increasing_prices = [100, 105, 110, 115, 120, 125]
        for price in increasing_prices:
            result = rsi.update(price)
        
        # RSI should be high (overbought)
        assert result > 70

class TestADXIndicator:
    """Test ADX calculation"""
    
    def test_calculate_adx_basic(self):
        """Test basic ADX calculation"""
        # Generate trending data
        highs = []
        lows = []
        closes = []
        
        base_price = 100
        for i in range(30):
            trend = i * 0.5  # Upward trend
            high = base_price + trend + 2
            low = base_price + trend - 2
            close = base_price + trend
            
            highs.append(high)
            lows.append(low)
            closes.append(close)
        
        adx_values, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14)
        
        assert len(adx_values) > 0
        assert len(plus_di) > 0
        assert len(minus_di) > 0
        assert all(0 <= adx <= 100 for adx in adx_values)
    
    def test_adx_indicator_stateful(self):
        """Test stateful ADX indicator"""
        adx = ADXIndicator(period=5)  # Shorter period for testing
        
        # Generate trending data
        base_price = 100
        for i in range(20):
            trend = i * 0.3
            high = base_price + trend + 1
            low = base_price + trend - 1
            close = base_price + trend
            
            result = adx.update(high, low, close)
        
        assert result is not None
        adx_val, plus_di_val, minus_di_val = result
        assert 0 <= adx_val <= 100
        assert plus_di_val >= 0
        assert minus_di_val >= 0

class TestVWAPIndicator:
    """Test VWAP calculation"""
    
    def test_calculate_vwap_basic(self):
        """Test basic VWAP calculation"""
        prices = [100, 101, 102, 101, 103]
        volumes = [1000, 1500, 800, 1200, 900]
        
        vwap_values = calculate_vwap(prices, volumes)
        
        assert len(vwap_values) == len(prices)
        assert all(isinstance(v, float) for v in vwap_values)
    
    def test_vwap_indicator_stateful(self):
        """Test stateful VWAP indicator"""
        vwap = VWAPIndicator()
        
        # Test data
        data = [
            (100, 1000),  # price, volume
            (101, 1500),
            (102, 800),
            (101, 1200),
            (103, 900)
        ]
        
        results = []
        for price, volume in data:
            result = vwap.update(price, volume)
            results.append(result)
        
        assert len(results) == len(data)
        assert all(isinstance(r, float) for r in results)

class TestVolatilityIndicator:
    """Test volatility calculation"""
    
    def test_calculate_volatility_basic(self):
        """Test basic volatility calculation"""
        # Generate price series
        prices = [100]
        for i in range(50):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
        
        vol_values = calculate_realized_volatility(prices, period=30, annualize=True)
        
        assert len(vol_values) > 0
        assert all(vol >= 0 for vol in vol_values)
    
    def test_volatility_indicator_stateful(self):
        """Test stateful volatility indicator"""
        vol = VolatilityIndicator(period=10, frequency="1m", annualize=True)
        
        # Generate price series
        prices = [100]
        for i in range(20):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))
        
        results = []
        for price in prices:
            result = vol.update(price)
            if result is not None:
                results.append(result)
        
        assert len(results) > 0
        assert all(vol_val >= 0 for vol_val in results)


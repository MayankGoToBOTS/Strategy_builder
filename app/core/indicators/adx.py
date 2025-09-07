# backend/app/core/indicators/adx.py
import numpy as np
from typing import List, Optional, Tuple

def calculate_adx(highs: List[float], lows: List[float], closes: List[float], 
                 period: int = 14) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate Average Directional Index (ADX), +DI, and -DI
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: ADX period (default 14)
        
    Returns:
        Tuple of (ADX values, +DI values, -DI values)
    """
    if len(highs) < period + 1 or len(highs) != len(lows) or len(highs) != len(closes):
        return [], [], []
    
    # Calculate True Range and Directional Movement
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []
    
    for i in range(1, len(highs)):
        # True Range
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        tr = max(tr1, tr2, tr3)
        tr_list.append(tr)
        
        # Directional Movement
        plus_dm = max(highs[i] - highs[i-1], 0) if highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0
        minus_dm = max(lows[i-1] - lows[i], 0) if lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0
        
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
    
    if len(tr_list) < period:
        return [], [], []
    
    # Calculate smoothed TR and DM
    atr_values = []
    plus_di_values = []
    minus_di_values = []
    adx_values = []
    
    # Initial smoothed values
    smoothed_tr = sum(tr_list[:period])
    smoothed_plus_dm = sum(plus_dm_list[:period])
    smoothed_minus_dm = sum(minus_dm_list[:period])
    
    # Calculate DI values
    plus_di = (smoothed_plus_dm / smoothed_tr) * 100 if smoothed_tr > 0 else 0
    minus_di = (smoothed_minus_dm / smoothed_tr) * 100 if smoothed_tr > 0 else 0
    
    plus_di_values.append(plus_di)
    minus_di_values.append(minus_di)
    
    # Continue smoothing for remaining periods
    dx_values = []
    for i in range(period, len(tr_list)):
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr_list[i]
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm_list[i]
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm_list[i]
        
        plus_di = (smoothed_plus_dm / smoothed_tr) * 100 if smoothed_tr > 0 else 0
        minus_di = (smoothed_minus_dm / smoothed_tr) * 100 if smoothed_tr > 0 else 0
        
        plus_di_values.append(plus_di)
        minus_di_values.append(minus_di)
        
        # Calculate DX
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = (di_diff / di_sum) * 100 if di_sum > 0 else 0
        dx_values.append(dx)
    
    # Calculate ADX from DX values
    if len(dx_values) >= period:
        # First ADX
        adx = sum(dx_values[:period]) / period
        adx_values.append(adx)
        
        # Smoothed ADX
        for i in range(period, len(dx_values)):
            adx = (adx * (period - 1) + dx_values[i]) / period
            adx_values.append(adx)
    
    return adx_values, plus_di_values, minus_di_values

class ADXIndicator:
    """Stateful ADX indicator for real-time calculation"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []
        self.smoothed_tr: Optional[float] = None
        self.smoothed_plus_dm: Optional[float] = None
        self.smoothed_minus_dm: Optional[float] = None
        self.adx_value: Optional[float] = None
        self.plus_di: Optional[float] = None
        self.minus_di: Optional[float] = None
        self.dx_values: List[float] = []
    
    def update(self, high: float, low: float, close: float) -> Optional[Tuple[float, float, float]]:
        """Update ADX with new price data"""
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        if len(self.highs) < 2:
            return None
        
        # Calculate TR and DM for current bar
        prev_close = self.closes[-2]
        prev_high = self.highs[-2]
        prev_low = self.lows[-2]
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = max(tr1, tr2, tr3)
        
        plus_dm = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
        minus_dm = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
        
        # Initialize or update smoothed values
        if len(self.highs) == self.period + 1:
            # First calculation
            tr_values = []
            plus_dm_values = []
            minus_dm_values = []
            
            for i in range(1, len(self.highs)):
                # Recalculate all TR and DM values
                h, l, c = self.highs[i], self.lows[i], self.closes[i]
                prev_c = self.closes[i-1]
                prev_h = self.highs[i-1] if i > 1 else h
                prev_l = self.lows[i-1] if i > 1 else l
                
                tr_val = max(h - l, abs(h - prev_c), abs(l - prev_c))
                plus_dm_val = max(h - prev_h, 0) if h - prev_h > prev_l - l else 0
                minus_dm_val = max(prev_l - l, 0) if prev_l - l > h - prev_h else 0
                
                tr_values.append(tr_val)
                plus_dm_values.append(plus_dm_val)
                minus_dm_values.append(minus_dm_val)
            
            self.smoothed_tr = sum(tr_values)
            self.smoothed_plus_dm = sum(plus_dm_values)
            self.smoothed_minus_dm = sum(minus_dm_values)
            
        elif len(self.highs) > self.period + 1:
            # Update smoothed values
            self.smoothed_tr = self.smoothed_tr - (self.smoothed_tr / self.period) + tr
            self.smoothed_plus_dm = self.smoothed_plus_dm - (self.smoothed_plus_dm / self.period) + plus_dm
            self.smoothed_minus_dm = self.smoothed_minus_dm - (self.smoothed_minus_dm / self.period) + minus_dm
        
        # Calculate DI values
        if self.smoothed_tr and self.smoothed_tr > 0:
            self.plus_di = (self.smoothed_plus_dm / self.smoothed_tr) * 100
            self.minus_di = (self.smoothed_minus_dm / self.smoothed_tr) * 100
            
            # Calculate DX
            di_diff = abs(self.plus_di - self.minus_di)
            di_sum = self.plus_di + self.minus_di
            dx = (di_diff / di_sum) * 100 if di_sum > 0 else 0
            self.dx_values.append(dx)
            
            # Calculate ADX
            if len(self.dx_values) >= self.period:
                if self.adx_value is None:
                    self.adx_value = sum(self.dx_values[-self.period:]) / self.period
                else:
                    self.adx_value = (self.adx_value * (self.period - 1) + dx) / self.period
        
        if self.adx_value is not None:
            return self.adx_value, self.plus_di, self.minus_di
        
        return None
    
    def get_values(self) -> Optional[Tuple[float, float, float]]:
        """Get current ADX, +DI, -DI values"""
        if self.adx_value is not None:
            return self.adx_value, self.plus_di, self.minus_di
        return None
    
    def reset(self):
        """Reset indicator state"""
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.smoothed_tr = None
        self.smoothed_plus_dm = None
        self.smoothed_minus_dm = None
        self.adx_value = None
        self.plus_di = None
        self.minus_di = None
        self.dx_values.clear()
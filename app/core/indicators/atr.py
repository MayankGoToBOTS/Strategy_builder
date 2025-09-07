# backend/app/core/indicators/atr.py
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class OHLCData:
    """Standard OHLC data structure"""
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

def calculate_atr(highs: List[float], lows: List[float], closes: List[float], 
                 period: int = 14) -> List[float]:
    """
    Calculate Average True Range (ATR)
    
    Args:
        highs: List of high prices
        lows: List of low prices  
        closes: List of close prices
        period: ATR period (default 14)
        
    Returns:
        List of ATR values
    """
    if len(highs) < 2 or len(highs) != len(lows) or len(highs) != len(closes):
        return []
    
    true_ranges = []
    
    # Calculate True Range for each period
    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]  # High - Low
        tr2 = abs(highs[i] - closes[i-1])  # High - Previous Close
        tr3 = abs(lows[i] - closes[i-1])   # Low - Previous Close
        
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    if len(true_ranges) < period:
        return []
    
    # Calculate ATR using simple moving average
    atr_values = []
    
    # First ATR value is simple average
    first_atr = sum(true_ranges[:period]) / period
    atr_values.append(first_atr)
    
    # Subsequent ATR values use smoothing
    for i in range(period, len(true_ranges)):
        current_atr = (atr_values[-1] * (period - 1) + true_ranges[i]) / period
        atr_values.append(current_atr)
    
    return atr_values

class ATRIndicator:
    """Stateful ATR indicator for real-time calculation"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.closes: List[float] = []
        self.true_ranges: List[float] = []
        self.atr_value: Optional[float] = None
    
    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """Update ATR with new price data"""
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        # Calculate true range if we have previous close
        if len(self.closes) > 1:
            prev_close = self.closes[-2]
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = max(tr1, tr2, tr3)
            self.true_ranges.append(true_range)
        
        # Calculate ATR if we have enough data
        if len(self.true_ranges) >= self.period:
            if self.atr_value is None:
                # First ATR calculation
                self.atr_value = sum(self.true_ranges[-self.period:]) / self.period
            else:
                # Smoothed ATR calculation
                current_tr = self.true_ranges[-1]
                self.atr_value = (self.atr_value * (self.period - 1) + current_tr) / self.period
        
        return self.atr_value
    
    def get_value(self) -> Optional[float]:
        """Get current ATR value"""
        return self.atr_value
    
    def reset(self):
        """Reset indicator state"""
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.true_ranges.clear()
        self.atr_value = None


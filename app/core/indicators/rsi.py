# backend/app/core/indicators/rsi.py
import numpy as np
from typing import List, Optional

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: List of prices (typically close prices)
        period: RSI period (default 14)
        
    Returns:
        List of RSI values
    """
    if len(prices) < period + 1:
        return []
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi_values = []
    
    # Calculate first RSI using simple average
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    
    # Calculate subsequent RSI values using smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
    
    return rsi_values

class RSIIndicator:
    """Stateful RSI indicator for real-time calculation"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.prices: List[float] = []
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.rsi_value: Optional[float] = None
    
    def update(self, price: float) -> Optional[float]:
        """Update RSI with new price"""
        self.prices.append(price)
        
        if len(self.prices) < 2:
            return None
        
        # Calculate price change
        change = self.prices[-1] - self.prices[-2]
        gain = max(change, 0)
        loss = max(-change, 0)
        
        if len(self.prices) <= self.period + 1:
            # Still building initial period
            if len(self.prices) == self.period + 1:
                # Calculate first RSI
                deltas = np.diff(self.prices)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                self.avg_gain = np.mean(gains)
                self.avg_loss = np.mean(losses)
                
                if self.avg_loss == 0:
                    self.rsi_value = 100.0
                else:
                    rs = self.avg_gain / self.avg_loss
                    self.rsi_value = 100 - (100 / (1 + rs))
        else:
            # Update using smoothing
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period
            
            if self.avg_loss == 0:
                self.rsi_value = 100.0
            else:
                rs = self.avg_gain / self.avg_loss
                self.rsi_value = 100 - (100 / (1 + rs))
        
        return self.rsi_value
    
    def get_value(self) -> Optional[float]:
        """Get current RSI value"""
        return self.rsi_value
    
    def reset(self):
        """Reset indicator state"""
        self.prices.clear()
        self.avg_gain = None
        self.avg_loss = None
        self.rsi_value = None


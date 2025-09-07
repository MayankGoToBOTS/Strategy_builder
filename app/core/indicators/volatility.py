# backend/app/core/indicators/volatility.py
import numpy as np
from typing import List, Optional

def calculate_realized_volatility(prices: List[float], period: int = 30, 
                                annualize: bool = True, 
                                frequency: str = "1m") -> List[float]:
    """
    Calculate realized volatility using price returns
    
    Args:
        prices: List of prices
        period: Lookback period for volatility calculation
        annualize: Whether to annualize the volatility
        frequency: Data frequency ("1m", "5m", "1h", "1d")
        
    Returns:
        List of realized volatility values
    """
    if len(prices) < period + 1:
        return []
    
    # Calculate log returns
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    
    # Frequency multipliers for annualization
    freq_multipliers = {
        "1m": 525600,  # minutes in a year
        "5m": 105120,  # 5-minute periods in a year
        "15m": 35040,  # 15-minute periods in a year
        "1h": 8760,    # hours in a year
        "4h": 2190,    # 4-hour periods in a year
        "1d": 365      # days in a year
    }
    
    multiplier = freq_multipliers.get(frequency, 1) if annualize else 1
    
    volatility_values = []
    
    for i in range(period - 1, len(returns)):
        period_returns = returns[i - period + 1:i + 1]
        vol = np.std(period_returns) * np.sqrt(multiplier)
        volatility_values.append(vol)
    
    return volatility_values

class VolatilityIndicator:
    """Stateful volatility indicator for real-time calculation"""
    
    def __init__(self, period: int = 30, frequency: str = "1m", annualize: bool = True):
        self.period = period
        self.frequency = frequency
        self.annualize = annualize
        self.prices: List[float] = []
        self.returns: List[float] = []
        self.volatility_value: Optional[float] = None
        
        # Frequency multipliers for annualization
        self.freq_multipliers = {
            "1m": 525600,
            "5m": 105120,
            "15m": 35040,
            "1h": 8760,
            "4h": 2190,
            "1d": 365
        }
    
    def update(self, price: float) -> Optional[float]:
        """Update volatility with new price"""
        self.prices.append(price)
        
        # Calculate return if we have previous price
        if len(self.prices) > 1:
            return_val = np.log(self.prices[-1] / self.prices[-2])
            self.returns.append(return_val)
        
        # Calculate volatility if we have enough returns
        if len(self.returns) >= self.period:
            recent_returns = self.returns[-self.period:]
            std_dev = np.std(recent_returns)
            
            if self.annualize:
                multiplier = self.freq_multipliers.get(self.frequency, 1)
                self.volatility_value = std_dev * np.sqrt(multiplier)
            else:
                self.volatility_value = std_dev
        
        return self.volatility_value
    
    def get_value(self) -> Optional[float]:
        """Get current volatility value"""
        return self.volatility_value
    
    def reset(self):
        """Reset indicator state"""
        self.prices.clear()
        self.returns.clear()
        self.volatility_value = None


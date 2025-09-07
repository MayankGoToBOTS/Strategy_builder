# backend/app/core/indicators/vwap.py
import numpy as np
from typing import List, Optional
from datetime import datetime, time

def calculate_vwap(prices: List[float], volumes: List[float], 
                  timestamps: Optional[List[datetime]] = None, 
                  session_start: time = time(0, 0)) -> List[float]:
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Args:
        prices: List of prices (typically close or typical price)
        volumes: List of volumes
        timestamps: Optional list of timestamps for session resets
        session_start: Time when VWAP resets (default midnight)
        
    Returns:
        List of VWAP values
    """
    if len(prices) != len(volumes) or len(prices) == 0:
        return []
    
    vwap_values = []
    cumulative_pv = 0.0
    cumulative_volume = 0.0
    last_session_date = None
    
    for i, (price, volume) in enumerate(zip(prices, volumes)):
        # Check for session reset if timestamps provided
        if timestamps and i < len(timestamps):
            current_date = timestamps[i].date()
            current_time = timestamps[i].time()
            
            if (last_session_date is not None and 
                current_date != last_session_date and 
                current_time >= session_start):
                # Reset for new session
                cumulative_pv = 0.0
                cumulative_volume = 0.0
            
            last_session_date = current_date
        
        # Update cumulative values
        cumulative_pv += price * volume
        cumulative_volume += volume
        
        # Calculate VWAP
        vwap = cumulative_pv / cumulative_volume if cumulative_volume > 0 else price
        vwap_values.append(vwap)
    
    return vwap_values

class VWAPIndicator:
    """Stateful VWAP indicator for real-time calculation"""
    
    def __init__(self, session_start: time = time(0, 0)):
        self.session_start = session_start
        self.cumulative_pv = 0.0
        self.cumulative_volume = 0.0
        self.vwap_value: Optional[float] = None
        self.last_session_date: Optional[datetime] = None
    
    def update(self, price: float, volume: float, timestamp: datetime = None) -> float:
        """Update VWAP with new price and volume"""
        # Check for session reset
        if timestamp:
            current_date = timestamp.date()
            current_time = timestamp.time()
            
            if (self.last_session_date is not None and 
                current_date != self.last_session_date and 
                current_time >= self.session_start):
                # Reset for new session
                self.cumulative_pv = 0.0
                self.cumulative_volume = 0.0
            
            self.last_session_date = current_date
        
        # Update cumulative values
        self.cumulative_pv += price * volume
        self.cumulative_volume += volume
        
        # Calculate VWAP
        self.vwap_value = (self.cumulative_pv / self.cumulative_volume 
                          if self.cumulative_volume > 0 else price)
        
        return self.vwap_value
    
    def get_value(self) -> Optional[float]:
        """Get current VWAP value"""
        return self.vwap_value
    
    def reset(self):
        """Reset indicator state"""
        self.cumulative_pv = 0.0
        self.cumulative_volume = 0.0
        self.vwap_value = None
        self.last_session_date = None


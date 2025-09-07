# backend/app/services/backtester/slippage.py
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class SlippageCalculator:
    """Calculate slippage based on market conditions and order characteristics"""
    
    def __init__(self):
        self.base_slippage_ppm = 150  # Parts per million (0.015%)
        self.latency_ms = 80
        self.market_impact_factor = 1.0
        
        # Slippage parameters
        self.volume_impact_threshold = 100000  # USD notional threshold
        self.volatility_multiplier = 2.0
    
    def configure(self, latency_ms: int = 80, base_slippage_ppm: int = 150):
        """Configure slippage calculator"""
        self.latency_ms = latency_ms
        self.base_slippage_ppm = base_slippage_ppm
        logger.info(f"Slippage calculator configured: {latency_ms}ms latency, {base_slippage_ppm}ppm base slippage")
    
    def apply_slippage(self, price: float, side: str, order_type: str, 
                      volume: Optional[float] = None, volatility: Optional[float] = None) -> float:
        """Apply slippage to order price"""
        try:
            # Base slippage
            base_slippage = self.base_slippage_ppm / 1_000_000
            
            # Latency impact
            latency_impact = (self.latency_ms / 1000) * 0.001  # 0.1% per second
            
            # Order type impact
            order_impact = self._get_order_type_impact(order_type)
            
            # Volume impact
            volume_impact = self._calculate_volume_impact(volume) if volume else 0
            
            # Volatility impact
            vol_impact = self._calculate_volatility_impact(volatility) if volatility else 0
            
            # Total slippage
            total_slippage = base_slippage + latency_impact + order_impact + volume_impact + vol_impact
            
            # Apply slippage direction
            if side.lower() == "buy":
                slipped_price = price * (1 + total_slippage)
            else:  # sell
                slipped_price = price * (1 - total_slippage)
            
            logger.debug(f"Applied {total_slippage:.4%} slippage to {side} order")
            return slipped_price
            
        except Exception as e:
            logger.error(f"Slippage calculation failed: {e}")
            # Return original price as fallback
            return price
    
    def _get_order_type_impact(self, order_type: str) -> float:
        """Get slippage impact based on order type"""
        impact_map = {
            "market": 0.0005,     # 0.05% additional for market orders
            "limit": 0.0,         # No additional slippage for limit orders
            "limit_post_only": 0.0,  # No additional slippage for post-only
            "stop": 0.001,        # 0.1% additional for stop orders
            "stop_limit": 0.0005  # 0.05% additional for stop-limit
        }
        
        return impact_map.get(order_type, 0.0005)
    
    def _calculate_volume_impact(self, volume: float) -> float:
        """Calculate market impact based on volume"""
        if volume > self.volume_impact_threshold:
            # Logarithmic impact model
            excess_volume = volume - self.volume_impact_threshold
            impact = 0.0001 * np.log(1 + excess_volume / self.volume_impact_threshold)
            return min(impact, 0.002)  # Cap at 0.2%
        
        return 0.0
    
    def _calculate_volatility_impact(self, volatility: float) -> float:
        """Calculate slippage impact based on market volatility"""
        # Higher volatility increases slippage
        if volatility > 0.5:  # 50% annual volatility threshold
            vol_impact = (volatility - 0.5) * self.volatility_multiplier * 0.0001
            return min(vol_impact, 0.001)  # Cap at 0.1%
        
        return 0.0


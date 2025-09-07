# backend/app/services/backtester/fees.py
from typing import Dict, Optional
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class FeeCalculator:
    """Calculate trading fees based on exchange and order type"""
    
    def __init__(self):
        self.fee_schedules = {
            "binance_spot_default": {
                "maker": 0.001,  # 0.1%
                "taker": 0.001,  # 0.1%
                "minimum_fee": 0.0  # No minimum fee
            },
            "binance_futures_default": {
                "maker": 0.0002,  # 0.02%
                "taker": 0.0004,  # 0.04%
                "minimum_fee": 0.0
            },
            "bybit_spot_default": {
                "maker": 0.001,  # 0.1%
                "taker": 0.001,  # 0.1%
                "minimum_fee": 0.0
            },
            "bybit_futures_default": {
                "maker": 0.0002,  # 0.02%
                "taker": 0.0004,  # 0.04%
                "minimum_fee": 0.0
            }
        }
        
        self.current_schedule = "binance_futures_default"
    
    def configure(self, fees_schema: str):
        """Configure fee calculator with specific fee schedule"""
        if fees_schema in self.fee_schedules:
            self.current_schedule = fees_schema
            logger.info(f"Fee calculator configured with {fees_schema}")
        else:
            logger.warning(f"Unknown fee schedule {fees_schema}, using default")
    
    def calculate_fee(self, notional: float, order_type: str = "market") -> float:
        """Calculate fee for a trade"""
        try:
            schedule = self.fee_schedules[self.current_schedule]
            
            # Determine if maker or taker
            if order_type in ["limit", "limit_post_only"]:
                fee_rate = schedule["maker"]
            else:  # market orders
                fee_rate = schedule["taker"]
            
            fee = notional * fee_rate
            
            # Apply minimum fee if specified
            min_fee = schedule.get("minimum_fee", 0.0)
            fee = max(fee, min_fee)
            
            return fee
            
        except Exception as e:
            logger.error(f"Fee calculation failed: {e}")
            # Fallback to default commission
            return notional * settings.DEFAULT_COMMISSION


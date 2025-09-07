# backend/app/services/backtester/strategies/dca_periodic.py
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from app.services.backtester.engine import BaseStrategy, BacktestContext
from app.core.indicators.atr import ATRIndicator
from app.core.indicators.rsi import RSIIndicator
from app.core.indicators.adx import ADXIndicator
import logging

logger = logging.getLogger(__name__)

class DCAStrategy(BaseStrategy):
    """
    Dollar Cost Averaging (DCA) strategy with periodic entries
    
    Entry Logic:
    - Time-based periodic purchases
    - ATR-aware position sizing
    - Optional momentum filter to pause DCA in strong downtrends
    
    Risk Management:
    - Gradual position building
    - ATR-based stop loss
    - Momentum veto to pause buying
    """
    
    def __init__(self, bot_config):
        super().__init__(bot_config)
        
        # Strategy parameters
        self.purchase_interval_bars = self.params.get("purchase_interval_bars", 1440)  # Daily
        self.base_purchase_amount = self.params.get("base_purchase_amount", 0.05)  # 5% of capital
        self.atr_size_adjustment = self.params.get("atr_size_adjustment", True)
        self.momentum_veto = self.params.get("momentum_veto", True)
        self.rsi_oversold_threshold = self.params.get("rsi_oversold_threshold", 30)
        self.rsi_overbought_threshold = self.params.get("rsi_overbought_threshold", 70)
        self.adx_trend_threshold = self.params.get("adx_trend_threshold", 25)
        self.stop_loss_atr = self.params.get("stop_loss_atr", 4.0)
        
        # Indicators
        self.atr = ATRIndicator(period=14)
        self.rsi = RSIIndicator(period=14)
        self.adx = ADXIndicator(period=14)
        
        # State tracking
        self.last_purchase_bar = -999999
        self.total_shares_bought = 0
        self.average_purchase_price = 0
        self.dca_active = True
        
    def get_warmup_bars(self) -> int:
        return 50
    
    def on_start(self, context: BacktestContext):
        """Initialize strategy"""
        self.is_initialized = True
        self.last_purchase_bar = context.bars_processed
        logger.info(f"DCA strategy initialized for {self.symbols}")
    
    def on_bar(self, context: BacktestContext) -> List[Dict]:
        """Process new bar and generate DCA orders"""
        try:
            bar = context.current_bar
            features = context.features
            
            # Skip if not our symbol
            if bar.symbol not in self.symbols:
                return []
            
            # Update indicators
            atr_value = self.atr.update(bar.high, bar.low, bar.close)
            rsi_value = self.rsi.update(bar.close)
            adx_values = self.adx.update(bar.high, bar.low, bar.close)
            
            # Need minimum data
            if not all([atr_value, rsi_value]):
                return []
            
            # Check if it's time for next DCA purchase
            bars_since_last = context.bars_processed - self.last_purchase_bar
            if bars_since_last < self.purchase_interval_bars:
                return []
            
            # Check momentum veto
            if self.momentum_veto and not self._check_momentum_conditions(rsi_value, adx_values):
                logger.debug("DCA purchase vetoed due to momentum conditions")
                return []
            
            # Create DCA purchase order
            order = self._create_dca_order(bar, atr_value, context)
            if order:
                self.last_purchase_bar = context.bars_processed
                return [order]
            
            return []
            
        except Exception as e:
            logger.error(f"DCA strategy error: {e}")
            return []
    
    def on_stop(self, context: BacktestContext):
        """Finalize strategy"""
        if self.total_shares_bought > 0:
            final_price = context.current_bar.close
            total_return = (final_price - self.average_purchase_price) / self.average_purchase_price * 100
            logger.info(f"DCA strategy completed. Average price: ${self.average_purchase_price:.4f}, "
                       f"Final price: ${final_price:.4f}, Return: {total_return:.2f}%")
    
    def _check_momentum_conditions(self, rsi_value: float, 
                                  adx_values: Optional[tuple]) -> bool:
        """Check if momentum conditions allow DCA purchase"""
        try:
            # Always allow purchases in oversold conditions
            if rsi_value <= self.rsi_oversold_threshold:
                return True
            
            # Avoid purchases when extremely overbought
            if rsi_value >= self.rsi_overbought_threshold:
                return False
            
            # Check for strong downtrend using ADX
            if adx_values:
                adx, plus_di, minus_di = adx_values
                
                # Strong downtrend: High ADX + Minus DI > Plus DI
                if adx > self.adx_trend_threshold and minus_di > plus_di:
                    # Pause DCA in strong downtrends unless very oversold
                    return rsi_value <= 35  # Allow only if very oversold
            
            # Default: allow purchase
            return True
            
        except Exception as e:
            logger.error(f"Momentum condition check failed: {e}")
            return True  # Default to allowing purchase
    
    def _create_dca_order(self, bar, atr_value, context: BacktestContext) -> Optional[Dict]:
        """Create DCA purchase order"""
        try:
            # Base purchase amount
            purchase_amount = context.portfolio_value * self.base_purchase_amount
            
            # Adjust size based on ATR (higher volatility = smaller size)
            if self.atr_size_adjustment:
                atr_pct = atr_value / bar.close
                # Reduce size if ATR is high (>3%), increase if low (<1%)
                if atr_pct > 0.03:
                    size_multiplier = 0.7  # Reduce size in high volatility
                elif atr_pct < 0.01:
                    size_multiplier = 1.3  # Increase size in low volatility
                else:
                    size_multiplier = 1.0
                
                purchase_amount *= size_multiplier
            
            # Calculate quantity
            quantity = purchase_amount / bar.close
            
            # Minimum quantity check
            if quantity < 0.001:
                return None
            
            # Update tracking
            new_total_cost = (self.total_shares_bought * self.average_purchase_price) + purchase_amount
            new_total_shares = self.total_shares_bought + quantity
            self.average_purchase_price = new_total_cost / new_total_shares
            self.total_shares_bought = new_total_shares
            
            # Calculate stop loss
            stop_loss = bar.close - (atr_value * self.stop_loss_atr)
            
            return {
                "symbol": bar.symbol,
                "side": "buy",
                "quantity": quantity,
                "type": "market",
                "stop_loss": stop_loss,
                "strategy": self.name,
                "dca_purchase": True,
                "purchase_amount": purchase_amount
            }
            
        except Exception as e:
            logger.error(f"DCA order creation failed: {e}")
            return None


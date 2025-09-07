# backend/app/services/backtester/strategies/momentum_trend.py
import numpy as np
from typing import List, Dict, Optional, Tuple
from app.services.backtester.engine import BaseStrategy, BacktestContext
from app.core.indicators.atr import ATRIndicator
from app.core.indicators.adx import ADXIndicator
from app.core.indicators.rsi import RSIIndicator
import logging

logger = logging.getLogger(__name__)

class MomentumStrategy(BaseStrategy):
    """
    Momentum/Trend following strategy
    
    Entry Logic:
    - ADX > threshold indicates trending market
    - +DI > -DI for long, -DI > +DI for short
    - RSI confirmation for momentum direction
    - Moving average filter
    
    Risk Management:
    - ATR-based trailing stop
    - Position sizing based on volatility
    - Trend strength filtering
    """
    
    def __init__(self, bot_config):
        super().__init__(bot_config)
        
        # Strategy parameters
        self.adx_threshold = self.params.get("adx_threshold", 25)
        self.rsi_entry_long = self.params.get("rsi_entry_long", 55)
        self.rsi_entry_short = self.params.get("rsi_entry_short", 45)
        self.ma_period = self.params.get("ma_period", 20)
        self.atr_stop_mult = self.params.get("atr_stop_mult", 2.0)
        self.atr_trail_mult = self.params.get("atr_trail_mult", 1.5)
        self.position_size_pct = self.params.get("position_size_pct", 0.1)  # 10% of capital
        self.min_trend_strength = self.params.get("min_trend_strength", 30)
        
        # Indicators
        self.atr = ATRIndicator(period=14)
        self.adx = ADXIndicator(period=14)
        self.rsi = RSIIndicator(period=14)
        
        # State tracking
        self.ma_prices = []
        self.current_position = None  # {"side": "long/short", "entry": price, "stop": price, "size": qty}
        self.highest_profit = 0
        self.lowest_loss = 0
        
    def get_warmup_bars(self) -> int:
        return 50
    
    def on_start(self, context: BacktestContext):
        """Initialize strategy"""
        self.is_initialized = True
        logger.info(f"Momentum strategy initialized for {self.symbols}")
    
    def on_bar(self, context: BacktestContext) -> List[Dict]:
        """Process new bar and generate momentum signals"""
        try:
            bar = context.current_bar
            features = context.features
            
            # Skip if not our symbol
            if bar.symbol not in self.symbols:
                return []
            
            # Update indicators
            atr_value = self.atr.update(bar.high, bar.low, bar.close)
            adx_values = self.adx.update(bar.high, bar.low, bar.close)
            rsi_value = self.rsi.update(bar.close)
            
            # Update moving average
            self._update_moving_average(bar.close)
            
            # Need minimum data
            if not all([atr_value, adx_values, rsi_value]) or len(self.ma_prices) < self.ma_period:
                return []
            
            adx, plus_di, minus_di = adx_values
            ma_value = np.mean(self.ma_prices[-self.ma_period:])
            
            orders = []
            
            # Update existing position
            if self.current_position:
                exit_order = self._update_position(bar, atr_value, adx, plus_di, minus_di)
                if exit_order:
                    orders.append(exit_order)
                    self.current_position = None
            else:
                # Look for new entry
                entry_order = self._check_entry_signals(bar, atr_value, adx, plus_di, minus_di, 
                                                      rsi_value, ma_value, context)
                if entry_order:
                    orders.append(entry_order)
                    self._track_new_position(entry_order, bar.close, atr_value)
            
            return orders
            
        except Exception as e:
            logger.error(f"Momentum strategy error: {e}")
            return []
    
    def on_stop(self, context: BacktestContext):
        """Finalize strategy"""
        if self.current_position:
            logger.info(f"Momentum strategy stopped with open {self.current_position['side']} position")
        else:
            logger.info("Momentum strategy stopped with no open positions")
    
    def _update_moving_average(self, price: float):
        """Update moving average prices"""
        self.ma_prices.append(price)
        if len(self.ma_prices) > self.ma_period + 5:  # Keep some extra for calculation
            self.ma_prices = self.ma_prices[-(self.ma_period + 5):]
    
    def _check_entry_signals(self, bar, atr_value: float, adx: float, plus_di: float, 
                           minus_di: float, rsi_value: float, ma_value: float, 
                           context: BacktestContext) -> Optional[Dict]:
        """Check for momentum entry signals"""
        try:
            # Must have strong trend
            if adx < self.adx_threshold:
                return None
            
            # Long signal
            if (plus_di > minus_di and  # Positive momentum
                bar.close > ma_value and  # Above moving average
                rsi_value > self.rsi_entry_long and  # RSI confirmation
                adx > self.min_trend_strength):  # Strong trend
                
                return self._create_long_order(bar, atr_value, context)
            
            # Short signal
            elif (minus_di > plus_di and  # Negative momentum
                  bar.close < ma_value and  # Below moving average
                  rsi_value < self.rsi_entry_short and  # RSI confirmation
                  adx > self.min_trend_strength):  # Strong trend
                
                return self._create_short_order(bar, atr_value, context)
            
            return None
            
        except Exception as e:
            logger.error(f"Entry signal check failed: {e}")
            return None
    
    def _create_long_order(self, bar, atr_value: float, context: BacktestContext) -> Optional[Dict]:
        """Create long momentum order"""
        try:
            entry_price = bar.close
            initial_stop = entry_price - (atr_value * self.atr_stop_mult)
            
            # Position sizing based on volatility
            atr_pct = atr_value / entry_price
            base_size = context.portfolio_value * self.position_size_pct
            
            # Adjust size based on volatility (lower size for higher volatility)
            if atr_pct > 0.05:  # >5% ATR
                size_mult = 0.6
            elif atr_pct > 0.03:  # >3% ATR
                size_mult = 0.8
            else:
                size_mult = 1.0
            
            position_value = base_size * size_mult
            quantity = position_value / entry_price
            
            # Minimum quantity check
            if quantity < 0.001:
                return None
            
            return {
                "symbol": bar.symbol,
                "side": "buy",
                "quantity": quantity,
                "type": "market",
                "stop_loss": initial_stop,
                "strategy": self.name,
                "momentum_long": True
            }
            
        except Exception as e:
            logger.error(f"Long momentum order creation failed: {e}")
            return None
    
    def _create_short_order(self, bar, atr_value: float, context: BacktestContext) -> Optional[Dict]:
        """Create short momentum order"""
        try:
            entry_price = bar.close
            initial_stop = entry_price + (atr_value * self.atr_stop_mult)
            
            # Position sizing based on volatility
            atr_pct = atr_value / entry_price
            base_size = context.portfolio_value * self.position_size_pct
            
            # Adjust size based on volatility
            if atr_pct > 0.05:
                size_mult = 0.6
            elif atr_pct > 0.03:
                size_mult = 0.8
            else:
                size_mult = 1.0
            
            position_value = base_size * size_mult
            quantity = position_value / entry_price
            
            # Minimum quantity check
            if quantity < 0.001:
                return None
            
            return {
                "symbol": bar.symbol,
                "side": "sell",
                "quantity": quantity,
                "type": "market",
                "stop_loss": initial_stop,
                "strategy": self.name,
                "momentum_short": True
            }
            
        except Exception as e:
            logger.error(f"Short momentum order creation failed: {e}")
            return None
    
    def _track_new_position(self, order: Dict, entry_price: float, atr_value: float):
        """Track new position for trailing stop management"""
        side = "long" if order["side"] == "buy" else "short"
        
        self.current_position = {
            "side": side,
            "entry": entry_price,
            "stop": order["stop_loss"],
            "size": order["quantity"],
            "highest_price": entry_price if side == "long" else entry_price,
            "lowest_price": entry_price if side == "short" else entry_price
        }
        
        self.highest_profit = 0
        self.lowest_loss = 0
    
    def _update_position(self, bar, atr_value: float, adx: float, 
                        plus_di: float, minus_di: float) -> Optional[Dict]:
        """Update existing position with trailing stop"""
        try:
            if not self.current_position:
                return None
            
            current_price = bar.close
            position = self.current_position
            
            # Update price extremes
            if position["side"] == "long":
                position["highest_price"] = max(position["highest_price"], current_price)
                
                # Calculate trailing stop
                trail_stop = position["highest_price"] - (atr_value * self.atr_trail_mult)
                position["stop"] = max(position["stop"], trail_stop)
                
                # Check exit conditions
                if (current_price <= position["stop"] or  # Stop loss hit
                    adx < self.adx_threshold or  # Trend weakening
                    minus_di > plus_di):  # Momentum reversal
                    
                    return {
                        "symbol": bar.symbol,
                        "side": "sell",
                        "quantity": position["size"],
                        "type": "market",
                        "strategy": self.name,
                        "exit_long": True
                    }
            
            else:  # short position
                position["lowest_price"] = min(position["lowest_price"], current_price)
                
                # Calculate trailing stop
                trail_stop = position["lowest_price"] + (atr_value * self.atr_trail_mult)
                position["stop"] = min(position["stop"], trail_stop)
                
                # Check exit conditions
                if (current_price >= position["stop"] or  # Stop loss hit
                    adx < self.adx_threshold or  # Trend weakening
                    plus_di > minus_di):  # Momentum reversal
                    
                    return {
                        "symbol": bar.symbol,
                        "side": "buy",
                        "quantity": position["size"],
                        "type": "market",
                        "strategy": self.name,
                        "exit_short": True
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Position update failed: {e}")
            return None
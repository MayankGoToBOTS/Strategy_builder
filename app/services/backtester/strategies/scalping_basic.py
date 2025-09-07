# backend/app/services/backtester/strategies/scalping_basic.py
import numpy as np
from typing import List, Dict, Optional
from app.services.backtester.engine import BaseStrategy, BacktestContext
from app.core.indicators.vwap import VWAPIndicator
from app.core.indicators.atr import ATRIndicator
from app.core.indicators.rsi import RSIIndicator
import logging

logger = logging.getLogger(__name__)

class ScalpingStrategy(BaseStrategy):
    """
    Basic scalping strategy using VWAP bias and micro-range breakouts
    
    Entry Logic:
    - Price above VWAP + RSI > 50 = Long bias
    - Price below VWAP + RSI < 50 = Short bias
    - Micro breakout: price breaks recent high/low with volume confirmation
    
    Risk Management:
    - Tight ATR-based stops (1.2x ATR)
    - Quick profit targets (0.8x ATR)
    - Cooldown period between trades
    """
    
    def __init__(self, bot_config):
        super().__init__(bot_config)
        
        # Strategy parameters
        self.atr_mult_sl = self.params.get("atr_mult_sl", 1.2)
        self.atr_mult_tp = self.params.get("atr_mult_tp", 0.8)
        self.cooldown_bars = self.params.get("cooldown_bars", 5)
        self.volume_threshold = self.params.get("volume_threshold", 1.2)
        self.breakout_period = self.params.get("breakout_period", 10)
        
        # Indicators
        self.vwap = VWAPIndicator()
        self.atr = ATRIndicator(period=14)
        self.rsi = RSIIndicator(period=14)
        
        # State tracking
        self.last_trade_bar = -999
        self.recent_highs = []
        self.recent_lows = []
        self.recent_volumes = []
        
    def get_warmup_bars(self) -> int:
        return 30
    
    def on_start(self, context: BacktestContext):
        """Initialize strategy"""
        self.is_initialized = True
        logger.info(f"Scalping strategy initialized for {self.symbols}")
    
    def on_bar(self, context: BacktestContext) -> List[Dict]:
        """Process new bar and generate signals"""
        try:
            bar = context.current_bar
            features = context.features
            
            # Skip if not our symbol
            if bar.symbol not in self.symbols:
                return []
            
            # Update indicators
            vwap_value = self.vwap.update(bar.close, bar.volume, bar.timestamp)
            atr_value = self.atr.update(bar.high, bar.low, bar.close)
            rsi_value = self.rsi.update(bar.close)
            
            # Need minimum data
            if not all([vwap_value, atr_value, rsi_value]):
                return []
            
            # Update recent data
            self._update_recent_data(bar)
            
            # Check cooldown
            if context.bars_processed - self.last_trade_bar < self.cooldown_bars:
                return []
            
            # Check for entry signals
            orders = []
            
            # Long signal
            long_signal = self._check_long_signal(bar, vwap_value, rsi_value, atr_value)
            if long_signal:
                order = self._create_long_order(bar, atr_value, context)
                if order:
                    orders.append(order)
                    self.last_trade_bar = context.bars_processed
            
            # Short signal
            short_signal = self._check_short_signal(bar, vwap_value, rsi_value, atr_value)
            if short_signal:
                order = self._create_short_order(bar, atr_value, context)
                if order:
                    orders.append(order)
                    self.last_trade_bar = context.bars_processed
            
            return orders
            
        except Exception as e:
            logger.error(f"Scalping strategy error: {e}")
            return []
    
    def on_stop(self, context: BacktestContext):
        """Finalize strategy"""
        logger.info("Scalping strategy stopped")
    
    def _update_recent_data(self, bar):
        """Update recent price and volume data"""
        self.recent_highs.append(bar.high)
        self.recent_lows.append(bar.low)
        self.recent_volumes.append(bar.volume)
        
        # Keep only recent data
        max_length = self.breakout_period + 5
        if len(self.recent_highs) > max_length:
            self.recent_highs = self.recent_highs[-max_length:]
            self.recent_lows = self.recent_lows[-max_length:]
            self.recent_volumes = self.recent_volumes[-max_length:]
    
    def _check_long_signal(self, bar, vwap_value, rsi_value, atr_value) -> bool:
        """Check for long entry signal"""
        try:
            # VWAP bias: price above VWAP
            vwap_bias = bar.close > vwap_value
            
            # RSI bias: RSI > 50 (bullish momentum)
            rsi_bias = rsi_value > 50
            
            # Micro breakout: current high breaks recent highs
            if len(self.recent_highs) >= self.breakout_period:
                recent_high = max(self.recent_highs[-self.breakout_period:-1])  # Exclude current bar
                breakout = bar.high > recent_high
            else:
                breakout = False
            
            # Volume confirmation
            if len(self.recent_volumes) >= 5:
                avg_volume = np.mean(self.recent_volumes[-5:-1])
                volume_confirm = bar.volume > avg_volume * self.volume_threshold
            else:
                volume_confirm = True  # Default to true if insufficient data
            
            return vwap_bias and rsi_bias and breakout and volume_confirm
            
        except Exception as e:
            logger.error(f"Long signal check failed: {e}")
            return False
    
    def _check_short_signal(self, bar, vwap_value, rsi_value, atr_value) -> bool:
        """Check for short entry signal"""
        try:
            # VWAP bias: price below VWAP
            vwap_bias = bar.close < vwap_value
            
            # RSI bias: RSI < 50 (bearish momentum)
            rsi_bias = rsi_value < 50
            
            # Micro breakdown: current low breaks recent lows
            if len(self.recent_lows) >= self.breakout_period:
                recent_low = min(self.recent_lows[-self.breakout_period:-1])  # Exclude current bar
                breakdown = bar.low < recent_low
            else:
                breakdown = False
            
            # Volume confirmation
            if len(self.recent_volumes) >= 5:
                avg_volume = np.mean(self.recent_volumes[-5:-1])
                volume_confirm = bar.volume > avg_volume * self.volume_threshold
            else:
                volume_confirm = True
            
            return vwap_bias and rsi_bias and breakdown and volume_confirm
            
        except Exception as e:
            logger.error(f"Short signal check failed: {e}")
            return False
    
    def _create_long_order(self, bar, atr_value, context) -> Optional[Dict]:
        """Create long order with risk management"""
        try:
            entry_price = bar.close
            stop_loss = entry_price - (atr_value * self.atr_mult_sl)
            take_profit = entry_price + (atr_value * self.atr_mult_tp)
            
            # Calculate position size based on risk
            risk_per_share = entry_price - stop_loss
            if risk_per_share <= 0:
                return None
            
            # Use 0.5% risk per trade for scalping
            risk_amount = context.portfolio_value * 0.005
            quantity = risk_amount / risk_per_share
            
            # Minimum quantity check
            if quantity < 0.001:  # Minimum trade size
                return None
            
            return {
                "symbol": bar.symbol,
                "side": "buy",
                "quantity": quantity,
                "type": "market",
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy": self.name
            }
            
        except Exception as e:
            logger.error(f"Long order creation failed: {e}")
            return None
    
    def _create_short_order(self, bar, atr_value, context) -> Optional[Dict]:
        """Create short order with risk management"""
        try:
            entry_price = bar.close
            stop_loss = entry_price + (atr_value * self.atr_mult_sl)
            take_profit = entry_price - (atr_value * self.atr_mult_tp)
            
            # Calculate position size based on risk
            risk_per_share = stop_loss - entry_price
            if risk_per_share <= 0:
                return None
            
            # Use 0.5% risk per trade for scalping
            risk_amount = context.portfolio_value * 0.005
            quantity = risk_amount / risk_per_share
            
            # Minimum quantity check
            if quantity < 0.001:
                return None
            
            return {
                "symbol": bar.symbol,
                "side": "sell",
                "quantity": quantity,
                "type": "market",
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy": self.name
            }
            
        except Exception as e:
            logger.error(f"Short order creation failed: {e}")
            return None


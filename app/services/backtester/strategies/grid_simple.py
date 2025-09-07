# backend/app/services/backtester/strategies/grid_simple.py
import numpy as np
from typing import List, Dict, Optional, Tuple
from app.services.backtester.engine import BaseStrategy, BacktestContext
from app.core.indicators.atr import ATRIndicator
from app.core.indicators.regime import RegimeDetector
import logging

logger = logging.getLogger(__name__)

class GridStrategy(BaseStrategy):
    """
    Simple grid trading strategy
    
    Entry Logic:
    - Only active in range-bound markets (regime detection)
    - Creates buy/sell grid levels based on ATR
    - Dynamic grid spacing based on volatility
    
    Risk Management:
    - Maximum number of grid levels
    - ATR-based grid spacing
    - Stop loss if trend breaks out
    """
    
    def __init__(self, bot_config):
        super().__init__(bot_config)
        
        # Strategy parameters
        self.grid_levels = self.params.get("grid_levels", 5)
        self.grid_spacing_atr = self.params.get("grid_spacing_atr", 0.5)
        self.max_position_per_level = self.params.get("max_position_per_level", 0.02)  # 2% of capital
        self.stop_loss_atr = self.params.get("stop_loss_atr", 3.0)
        self.min_atr_threshold = self.params.get("min_atr_threshold", 0.01)  # 1% minimum ATR
        self.max_atr_threshold = self.params.get("max_atr_threshold", 0.05)  # 5% maximum ATR
        
        # Indicators
        self.atr = ATRIndicator(period=14)
        self.regime_detector = RegimeDetector()
        
        # Grid state
        self.grid_center = None
        self.grid_levels_active = False
        self.buy_levels = []
        self.sell_levels = []
        self.grid_positions = {}  # level -> position info
        
    def get_warmup_bars(self) -> int:
        return 50
    
    def on_start(self, context: BacktestContext):
        """Initialize strategy"""
        self.is_initialized = True
        logger.info(f"Grid strategy initialized for {self.symbols}")
    
    def on_bar(self, context: BacktestContext) -> List[Dict]:
        """Process new bar and generate grid orders"""
        try:
            bar = context.current_bar
            features = context.features
            
            # Skip if not our symbol
            if bar.symbol not in self.symbols:
                return []
            
            # Update indicators
            atr_value = self.atr.update(bar.high, bar.low, bar.close)
            
            # Update regime detection
            regime_signal = self.regime_detector.update(features)
            
            # Need minimum data
            if not atr_value or not regime_signal:
                return []
            
            # Check if conditions are suitable for grid trading
            if not self._is_grid_suitable(regime_signal, atr_value, bar.close):
                # Deactivate grid and close positions if needed
                if self.grid_levels_active:
                    self.grid_levels_active = False
                    return self._close_all_grid_positions(bar)
                return []
            
            # Initialize or update grid
            if not self.grid_levels_active:
                self._initialize_grid(bar.close, atr_value)
                self.grid_levels_active = True
            
            # Generate grid orders
            orders = self._generate_grid_orders(bar, context)
            
            return orders
            
        except Exception as e:
            logger.error(f"Grid strategy error: {e}")
            return []
    
    def on_stop(self, context: BacktestContext):
        """Finalize strategy"""
        logger.info("Grid strategy stopped")
    
    def _is_grid_suitable(self, regime_signal, atr_value, current_price) -> bool:
        """Check if market conditions are suitable for grid trading"""
        try:
            # Only trade in range-bound markets
            if regime_signal.regime.value not in ["range", "low_liquidity"]:
                return False
            
            # Check ATR bounds
            atr_pct = atr_value / current_price
            if atr_pct < self.min_atr_threshold or atr_pct > self.max_atr_threshold:
                return False
            
            # Check regime confidence
            if regime_signal.confidence < 0.5:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Grid suitability check failed: {e}")
            return False
    
    def _initialize_grid(self, center_price: float, atr_value: float):
        """Initialize grid levels around current price"""
        try:
            self.grid_center = center_price
            grid_spacing = atr_value * self.grid_spacing_atr
            
            # Create buy levels (below center)
            self.buy_levels = []
            for i in range(1, self.grid_levels + 1):
                level_price = center_price - (grid_spacing * i)
                self.buy_levels.append(level_price)
            
            # Create sell levels (above center)
            self.sell_levels = []
            for i in range(1, self.grid_levels + 1):
                level_price = center_price + (grid_spacing * i)
                self.sell_levels.append(level_price)
            
            # Reset position tracking
            self.grid_positions = {}
            
            logger.info(f"Grid initialized: center=${center_price:.4f}, spacing=${grid_spacing:.4f}")
            
        except Exception as e:
            logger.error(f"Grid initialization failed: {e}")
    
    def _generate_grid_orders(self, bar, context: BacktestContext) -> List[Dict]:
        """Generate orders for grid levels"""
        orders = []
        
        try:
            current_price = bar.close
            
            # Check buy levels
            for level_price in self.buy_levels:
                if current_price <= level_price and level_price not in self.grid_positions:
                    # Price hit buy level, create buy order
                    order = self._create_grid_buy_order(bar, level_price, context)
                    if order:
                        orders.append(order)
                        # Mark level as filled
                        self.grid_positions[level_price] = {
                            "side": "buy",
                            "entry_price": level_price,
                            "quantity": order["quantity"]
                        }
            
            # Check sell levels
            for level_price in self.sell_levels:
                if current_price >= level_price and level_price not in self.grid_positions:
                    # Price hit sell level, create sell order
                    order = self._create_grid_sell_order(bar, level_price, context)
                    if order:
                        orders.append(order)
                        # Mark level as filled
                        self.grid_positions[level_price] = {
                            "side": "sell",
                            "entry_price": level_price,
                            "quantity": order["quantity"]
                        }
            
            # Check for profit taking (close positions when price moves to opposite side)
            profit_orders = self._check_profit_taking(bar, context)
            orders.extend(profit_orders)
            
            return orders
            
        except Exception as e:
            logger.error(f"Grid order generation failed: {e}")
            return []
    
    def _create_grid_buy_order(self, bar, level_price: float, context: BacktestContext) -> Optional[Dict]:
        """Create buy order for grid level"""
        try:
            # Calculate position size
            position_value = context.portfolio_value * self.max_position_per_level
            quantity = position_value / level_price
            
            # Minimum quantity check
            if quantity < 0.001:
                return None
            
            return {
                "symbol": bar.symbol,
                "side": "buy",
                "quantity": quantity,
                "type": "limit",
                "price": level_price,
                "strategy": self.name,
                "grid_level": level_price
            }
            
        except Exception as e:
            logger.error(f"Grid buy order creation failed: {e}")
            return None
    
    def _create_grid_sell_order(self, bar, level_price: float, context: BacktestContext) -> Optional[Dict]:
        """Create sell order for grid level"""
        try:
            # Calculate position size
            position_value = context.portfolio_value * self.max_position_per_level
            quantity = position_value / level_price
            
            # Minimum quantity check
            if quantity < 0.001:
                return None
            
            return {
                "symbol": bar.symbol,
                "side": "sell",
                "quantity": quantity,
                "type": "limit",
                "price": level_price,
                "strategy": self.name,
                "grid_level": level_price
            }
            
        except Exception as e:
            logger.error(f"Grid sell order creation failed: {e}")
            return None
    
    def _check_profit_taking(self, bar, context: BacktestContext) -> List[Dict]:
        """Check for profit taking opportunities"""
        orders = []
        
        try:
            current_price = bar.close
            positions_to_close = []
            
            for level_price, position in self.grid_positions.items():
                if position["side"] == "buy":
                    # Close buy position if price moved up significantly
                    profit_threshold = level_price * 1.01  # 1% profit
                    if current_price >= profit_threshold:
                        # Create sell order to close position
                        order = {
                            "symbol": bar.symbol,
                            "side": "sell",
                            "quantity": position["quantity"],
                            "type": "market",
                            "strategy": self.name,
                            "close_grid_level": level_price
                        }
                        orders.append(order)
                        positions_to_close.append(level_price)
                
                elif position["side"] == "sell":
                    # Close sell position if price moved down significantly
                    profit_threshold = level_price * 0.99  # 1% profit
                    if current_price <= profit_threshold:
                        # Create buy order to close position
                        order = {
                            "symbol": bar.symbol,
                            "side": "buy",
                            "quantity": position["quantity"],
                            "type": "market",
                            "strategy": self.name,
                            "close_grid_level": level_price
                        }
                        orders.append(order)
                        positions_to_close.append(level_price)
            
            # Remove closed positions
            for level_price in positions_to_close:
                del self.grid_positions[level_price]
            
            return orders
            
        except Exception as e:
            logger.error(f"Profit taking check failed: {e}")
            return []
    
    def _close_all_grid_positions(self, bar) -> List[Dict]:
        """Close all active grid positions"""
        orders = []
        
        try:
            for level_price, position in self.grid_positions.items():
                # Create opposite order to close position
                opposite_side = "sell" if position["side"] == "buy" else "buy"
                
                order = {
                    "symbol": bar.symbol,
                    "side": opposite_side,
                    "quantity": position["quantity"],
                    "type": "market",
                    "strategy": self.name,
                    "close_all_grid": True
                }
                orders.append(order)
            
            # Clear all positions
            self.grid_positions = {}
            
            logger.info("Closing all grid positions due to unsuitable conditions")
            return orders
            
        except Exception as e:
            logger.error(f"Grid position closing failed: {e}")
            return []
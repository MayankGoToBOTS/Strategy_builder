# backend/app/services/backtester/strategies/mw_pattern_detector.py
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from app.services.backtester.engine import BaseStrategy, BacktestContext
from app.core.indicators.atr import ATRIndicator
from app.core.indicators.rsi import RSIIndicator
import logging

logger = logging.getLogger(__name__)

@dataclass
class PricePoint:
    """Price point for pattern analysis"""
    price: float
    timestamp: int  # Bar index
    bar_high: float
    bar_low: float
    volume: float

@dataclass
class PatternSignal:
    """Pattern detection signal"""
    pattern_type: str
    entry_price: float
    stop_loss: float
    target_price: float
    confidence: float
    neckline: Optional[float] = None

class PatternStrategy(BaseStrategy):
    """
    Market structure pattern detection strategy
    
    Pattern Types:
    - Head and Shoulders / Inverse Head and Shoulders
    - Double Top / Double Bottom
    - Triangle breakouts
    - Support/Resistance breakouts
    
    Entry Logic:
    - Detect local extrema (swing highs/lows)
    - Identify pattern formations
    - Wait for neckline break with volume confirmation
    - Enter on retest of breakout level
    
    Risk Management:
    - Pattern-based stop losses
    - ATR-adjusted position sizing
    - Volume confirmation requirements
    """
    
    def __init__(self, bot_config):
        super().__init__(bot_config)
        
        # Strategy parameters
        self.min_pattern_bars = self.params.get("min_pattern_bars", 20)
        self.max_pattern_bars = self.params.get("max_pattern_bars", 100)
        self.volume_confirmation = self.params.get("volume_confirmation", True)
        self.volume_threshold = self.params.get("volume_threshold", 1.5)  # 1.5x average volume
        self.retest_tolerance = self.params.get("retest_tolerance", 0.005)  # 0.5% tolerance
        self.min_pattern_height = self.params.get("min_pattern_height", 0.02)  # 2% minimum height
        self.position_size_pct = self.params.get("position_size_pct", 0.08)  # 8% of capital
        
        # Indicators
        self.atr = ATRIndicator(period=14)
        self.rsi = RSIIndicator(period=14)
        
        # Pattern tracking
        self.price_history = []
        self.swing_highs = []
        self.swing_lows = []
        self.volume_history = []
        self.detected_patterns = []
        self.pending_retests = []
        
        # Current state
        self.last_pattern_check = 0
        self.pattern_check_interval = 5  # Check every 5 bars
        
    def get_warmup_bars(self) -> int:
        return 60
    
    def on_start(self, context: BacktestContext):
        """Initialize strategy"""
        self.is_initialized = True
        logger.info(f"Pattern detection strategy initialized for {self.symbols}")
    
    def on_bar(self, context: BacktestContext) -> List[Dict]:
        """Process new bar and detect patterns"""
        try:
            bar = context.current_bar
            features = context.features
            
            # Skip if not our symbol
            if bar.symbol not in self.symbols:
                return []
            
            # Update indicators
            atr_value = self.atr.update(bar.high, bar.low, bar.close)
            rsi_value = self.rsi.update(bar.close)
            
            # Update price and volume history
            self._update_history(bar, context.bars_processed)
            
            # Need minimum data
            if len(self.price_history) < self.min_pattern_bars or not atr_value:
                return []
            
            orders = []
            
            # Check for pattern breakouts and retests
            breakout_orders = self._check_pattern_breakouts(bar, atr_value, context)
            orders.extend(breakout_orders)
            
            # Detect new patterns periodically
            if context.bars_processed - self.last_pattern_check >= self.pattern_check_interval:
                self._detect_patterns()
                self.last_pattern_check = context.bars_processed
            
            return orders
            
        except Exception as e:
            logger.error(f"Pattern strategy error: {e}")
            return []
    
    def on_stop(self, context: BacktestContext):
        """Finalize strategy"""
        logger.info(f"Pattern strategy stopped. Detected {len(self.detected_patterns)} patterns")
    
    def _update_history(self, bar, bar_index: int):
        """Update price and volume history"""
        # Add current bar to history
        price_point = PricePoint(
            price=bar.close,
            timestamp=bar_index,
            bar_high=bar.high,
            bar_low=bar.low,
            volume=bar.volume
        )
        
        self.price_history.append(price_point)
        self.volume_history.append(bar.volume)
        
        # Keep only recent history
        max_history = self.max_pattern_bars + 20
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
        
        # Update swing points
        self._update_swing_points()
    
    def _update_swing_points(self):
        """Identify swing highs and lows"""
        if len(self.price_history) < 5:
            return
        
        # Look for swing highs and lows using a simple pivot detection
        lookback = 3
        
        for i in range(lookback, len(self.price_history) - lookback):
            point = self.price_history[i]
            
            # Check for swing high
            is_swing_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and self.price_history[j].bar_high >= point.bar_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                # Check if this swing high is not already recorded
                if not any(sh.timestamp == point.timestamp for sh in self.swing_highs):
                    self.swing_highs.append(point)
            
            # Check for swing low
            is_swing_low = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and self.price_history[j].bar_low <= point.bar_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                # Check if this swing low is not already recorded
                if not any(sl.timestamp == point.timestamp for sl in self.swing_lows):
                    self.swing_lows.append(point)
        
        # Keep only recent swing points
        max_swings = 20
        if len(self.swing_highs) > max_swings:
            self.swing_highs = self.swing_highs[-max_swings:]
        if len(self.swing_lows) > max_swings:
            self.swing_lows = self.swing_lows[-max_swings:]
    
    def _detect_patterns(self):
        """Detect chart patterns from swing points"""
        try:
            # Detect various patterns
            self._detect_head_and_shoulders()
            self._detect_double_tops_bottoms()
            self._detect_triangles()
            
            # Remove old patterns
            current_time = self.price_history[-1].timestamp if self.price_history else 0
            self.detected_patterns = [
                p for p in self.detected_patterns 
                if current_time - p.get('detected_at', 0) < self.max_pattern_bars
            ]
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
    
    def _detect_head_and_shoulders(self):
        """Detect Head and Shoulders patterns"""
        try:
            if len(self.swing_highs) < 3:
                return
            
            # Look for 3 consecutive swing highs forming H&S
            for i in range(len(self.swing_highs) - 2):
                left_shoulder = self.swing_highs[i]
                head = self.swing_highs[i + 1]
                right_shoulder = self.swing_highs[i + 2]
                
                # Basic H&S conditions
                if (head.bar_high > left_shoulder.bar_high and 
                    head.bar_high > right_shoulder.bar_high and
                    abs(left_shoulder.bar_high - right_shoulder.bar_high) / left_shoulder.bar_high < 0.05):  # Shoulders similar height
                    
                    # Find neckline (low between shoulders)
                    neckline_candidates = [
                        p for p in self.swing_lows 
                        if left_shoulder.timestamp < p.timestamp < right_shoulder.timestamp
                    ]
                    
                    if neckline_candidates:
                        neckline = min(neckline_candidates, key=lambda x: x.bar_low)
                        
                        # Calculate pattern height and target
                        pattern_height = head.bar_high - neckline.bar_low
                        if pattern_height / neckline.bar_low > self.min_pattern_height:
                            
                            target_price = neckline.bar_low - pattern_height
                            
                            pattern = {
                                "type": "head_and_shoulders",
                                "neckline": neckline.bar_low,
                                "target": target_price,
                                "stop_loss": head.bar_high,
                                "confidence": 0.7,
                                "detected_at": self.price_history[-1].timestamp,
                                "breakout_confirmed": False
                            }
                            
                            self.detected_patterns.append(pattern)
                            logger.info(f"Head and Shoulders pattern detected, neckline: {neckline.bar_low:.4f}")
            
            # Also check for Inverse Head and Shoulders
            if len(self.swing_lows) >= 3:
                for i in range(len(self.swing_lows) - 2):
                    left_shoulder = self.swing_lows[i]
                    head = self.swing_lows[i + 1]
                    right_shoulder = self.swing_lows[i + 2]
                    
                    if (head.bar_low < left_shoulder.bar_low and 
                        head.bar_low < right_shoulder.bar_low and
                        abs(left_shoulder.bar_low - right_shoulder.bar_low) / left_shoulder.bar_low < 0.05):
                        
                        # Find neckline (high between shoulders)
                        neckline_candidates = [
                            p for p in self.swing_highs 
                            if left_shoulder.timestamp < p.timestamp < right_shoulder.timestamp
                        ]
                        
                        if neckline_candidates:
                            neckline = max(neckline_candidates, key=lambda x: x.bar_high)
                            
                            pattern_height = neckline.bar_high - head.bar_low
                            if pattern_height / head.bar_low > self.min_pattern_height:
                                
                                target_price = neckline.bar_high + pattern_height
                                
                                pattern = {
                                    "type": "inverse_head_and_shoulders",
                                    "neckline": neckline.bar_high,
                                    "target": target_price,
                                    "stop_loss": head.bar_low,
                                    "confidence": 0.7,
                                    "detected_at": self.price_history[-1].timestamp,
                                    "breakout_confirmed": False
                                }
                                
                                self.detected_patterns.append(pattern)
                                logger.info(f"Inverse Head and Shoulders pattern detected, neckline: {neckline.bar_high:.4f}")
            
        except Exception as e:
            logger.error(f"Head and shoulders detection failed: {e}")
    
    def _detect_double_tops_bottoms(self):
        """Detect Double Top/Bottom patterns"""
        try:
            # Double tops
            if len(self.swing_highs) >= 2:
                for i in range(len(self.swing_highs) - 1):
                    first_peak = self.swing_highs[i]
                    second_peak = self.swing_highs[i + 1]
                    
                    # Check if peaks are similar height
                    height_diff = abs(first_peak.bar_high - second_peak.bar_high) / first_peak.bar_high
                    if height_diff < 0.03:  # Within 3%
                        
                        # Find valley between peaks
                        valley_candidates = [
                            p for p in self.swing_lows 
                            if first_peak.timestamp < p.timestamp < second_peak.timestamp
                        ]
                        
                        if valley_candidates:
                            valley = min(valley_candidates, key=lambda x: x.bar_low)
                            
                            pattern_height = first_peak.bar_high - valley.bar_low
                            if pattern_height / valley.bar_low > self.min_pattern_height:
                                
                                target_price = valley.bar_low - pattern_height
                                
                                pattern = {
                                    "type": "double_top",
                                    "neckline": valley.bar_low,
                                    "target": target_price,
                                    "stop_loss": max(first_peak.bar_high, second_peak.bar_high),
                                    "confidence": 0.6,
                                    "detected_at": self.price_history[-1].timestamp,
                                    "breakout_confirmed": False
                                }
                                
                                self.detected_patterns.append(pattern)
                                logger.info(f"Double Top pattern detected, neckline: {valley.bar_low:.4f}")
            
            # Double bottoms
            if len(self.swing_lows) >= 2:
                for i in range(len(self.swing_lows) - 1):
                    first_trough = self.swing_lows[i]
                    second_trough = self.swing_lows[i + 1]
                    
                    height_diff = abs(first_trough.bar_low - second_trough.bar_low) / first_trough.bar_low
                    if height_diff < 0.03:
                        
                        # Find peak between troughs
                        peak_candidates = [
                            p for p in self.swing_highs 
                            if first_trough.timestamp < p.timestamp < second_trough.timestamp
                        ]
                        
                        if peak_candidates:
                            peak = max(peak_candidates, key=lambda x: x.bar_high)
                            
                            pattern_height = peak.bar_high - first_trough.bar_low
                            if pattern_height / first_trough.bar_low > self.min_pattern_height:
                                
                                target_price = peak.bar_high + pattern_height
                                
                                pattern = {
                                    "type": "double_bottom",
                                    "neckline": peak.bar_high,
                                    "target": target_price,
                                    "stop_loss": min(first_trough.bar_low, second_trough.bar_low),
                                    "confidence": 0.6,
                                    "detected_at": self.price_history[-1].timestamp,
                                    "breakout_confirmed": False
                                }
                                
                                self.detected_patterns.append(pattern)
                                logger.info(f"Double Bottom pattern detected, neckline: {peak.bar_high:.4f}")
            
        except Exception as e:
            logger.error(f"Double top/bottom detection failed: {e}")
    
    def _detect_triangles(self):
        """Detect triangle patterns (simplified)"""
        try:
            if len(self.swing_highs) < 3 or len(self.swing_lows) < 3:
                return
            
            # Look for converging trend lines
            recent_highs = self.swing_highs[-3:]
            recent_lows = self.swing_lows[-3:]
            
            # Check if highs are descending and lows are ascending (symmetrical triangle)
            highs_descending = all(recent_highs[i].bar_high > recent_highs[i+1].bar_high 
                                 for i in range(len(recent_highs)-1))
            lows_ascending = all(recent_lows[i].bar_low < recent_lows[i+1].bar_low 
                               for i in range(len(recent_lows)-1))
            
            if highs_descending and lows_ascending:
                # Estimate breakout level (midpoint of triangle)
                latest_high = recent_highs[-1].bar_high
                latest_low = recent_lows[-1].bar_low
                breakout_level = (latest_high + latest_low) / 2
                
                pattern = {
                    "type": "symmetrical_triangle",
                    "neckline": breakout_level,
                    "target_up": latest_high + (latest_high - latest_low) * 0.5,
                    "target_down": latest_low - (latest_high - latest_low) * 0.5,
                    "stop_loss_up": latest_low,
                    "stop_loss_down": latest_high,
                    "confidence": 0.5,
                    "detected_at": self.price_history[-1].timestamp,
                    "breakout_confirmed": False,
                    "breakout_direction": None
                }
                
                self.detected_patterns.append(pattern)
                logger.info(f"Symmetrical Triangle pattern detected, breakout level: {breakout_level:.4f}")
            
        except Exception as e:
            logger.error(f"Triangle detection failed: {e}")
    
    def _check_pattern_breakouts(self, bar, atr_value: float, context: BacktestContext) -> List[Dict]:
        """Check for pattern breakouts and generate orders"""
        orders = []
        
        try:
            current_price = bar.close
            
            for pattern in self.detected_patterns:
                if pattern.get("breakout_confirmed"):
                    continue
                
                # Check for breakout
                breakout_order = None
                
                if pattern["type"] in ["head_and_shoulders", "double_top"]:
                    # Bearish patterns - look for breakdown
                    if current_price < pattern["neckline"]:
                        # Confirm with volume if required
                        if self._confirm_volume_breakout(bar):
                            breakout_order = self._create_breakout_order(
                                bar, "sell", pattern["target"], pattern["stop_loss"], 
                                atr_value, context
                            )
                            pattern["breakout_confirmed"] = True
                
                elif pattern["type"] in ["inverse_head_and_shoulders", "double_bottom"]:
                    # Bullish patterns - look for breakup
                    if current_price > pattern["neckline"]:
                        if self._confirm_volume_breakout(bar):
                            breakout_order = self._create_breakout_order(
                                bar, "buy", pattern["target"], pattern["stop_loss"], 
                                atr_value, context
                            )
                            pattern["breakout_confirmed"] = True
                
                elif pattern["type"] == "symmetrical_triangle":
                    # Bidirectional breakout
                    if current_price > pattern["neckline"] * 1.01:  # 1% above midpoint
                        if self._confirm_volume_breakout(bar):
                            breakout_order = self._create_breakout_order(
                                bar, "buy", pattern["target_up"], pattern["stop_loss_up"], 
                                atr_value, context
                            )
                            pattern["breakout_confirmed"] = True
                            pattern["breakout_direction"] = "up"
                    
                    elif current_price < pattern["neckline"] * 0.99:  # 1% below midpoint
                        if self._confirm_volume_breakout(bar):
                            breakout_order = self._create_breakout_order(
                                bar, "sell", pattern["target_down"], pattern["stop_loss_down"], 
                                atr_value, context
                            )
                            pattern["breakout_confirmed"] = True
                            pattern["breakout_direction"] = "down"
                
                if breakout_order:
                    orders.append(breakout_order)
                    logger.info(f"Pattern breakout confirmed: {pattern['type']}")
            
            return orders
            
        except Exception as e:
            logger.error(f"Pattern breakout check failed: {e}")
            return []
    
    def _confirm_volume_breakout(self, bar) -> bool:
        """Confirm breakout with volume analysis"""
        if not self.volume_confirmation or len(self.volume_history) < 10:
            return True  # Skip volume confirmation if not required or insufficient data
        
        try:
            avg_volume = np.mean(self.volume_history[-10:])
            return bar.volume > avg_volume * self.volume_threshold
        except:
            return True
    
    def _create_breakout_order(self, bar, side: str, target: float, stop_loss: float, 
                              atr_value: float, context: BacktestContext) -> Optional[Dict]:
        """Create order for pattern breakout"""
        try:
            # Position sizing based on pattern confidence and ATR
            base_size = context.portfolio_value * self.position_size_pct
            
            # Adjust size based on ATR volatility
            atr_pct = atr_value / bar.close
            if atr_pct > 0.04:  # High volatility
                size_mult = 0.7
            elif atr_pct < 0.02:  # Low volatility
                size_mult = 1.2
            else:
                size_mult = 1.0
            
            position_value = base_size * size_mult
            quantity = position_value / bar.close
            
            # Minimum quantity check
            if quantity < 0.001:
                return None
            
            return {
                "symbol": bar.symbol,
                "side": side,
                "quantity": quantity,
                "type": "market",
                "stop_loss": stop_loss,
                "take_profit": target,
                "strategy": self.name,
                "pattern_breakout": True
            }
            
        except Exception as e:
            logger.error(f"Breakout order creation failed: {e}")
            return None
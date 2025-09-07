# backend/app/core/indicators/regime.py
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class RegimeType(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    HIGH_VOL = "high_vol"
    LOW_LIQUIDITY = "low_liquidity"
    BREAKOUT = "breakout"

@dataclass
class RegimeSignal:
    regime: RegimeType
    confidence: float
    features: Dict[str, float]
    timestamp: Optional[float] = None

class RegimeDetector:
    """
    Multi-factor regime detection using technical indicators
    """
    
    def __init__(self, atr_period: int = 14, rsi_period: int = 14, 
                 adx_period: int = 14, vol_period: int = 30):
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.vol_period = vol_period
        
        # Thresholds for regime classification
        self.thresholds = {
            'adx_trend': 25.0,      # ADX > 25 indicates trend
            'adx_strong': 40.0,     # ADX > 40 indicates strong trend
            'rsi_oversold': 30.0,   # RSI < 30 oversold
            'rsi_overbought': 70.0, # RSI > 70 overbought
            'vol_high': 0.5,        # High volatility threshold
            'vol_low': 0.2,         # Low volatility threshold
            'atr_expansion': 1.5,   # ATR expansion ratio
        }
        
        # Historical data for calculations
        self.history: List[Dict[str, float]] = []
        self.current_regime: Optional[RegimeSignal] = None
    
    def update(self, features: Dict[str, float]) -> Optional[RegimeSignal]:
        """
        Update regime detection with new feature set
        
        Expected features:
        - atr_14: Average True Range
        - rsi_14: Relative Strength Index
        - adx_14: Average Directional Index
        - plus_di_14: Positive Directional Indicator
        - minus_di_14: Negative Directional Indicator
        - realized_vol_30: Realized volatility
        - price: Current price
        - volume: Current volume
        """
        self.history.append(features.copy())
        
        # Keep only recent history
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        # Need minimum data for regime detection
        if len(self.history) < max(self.atr_period, self.rsi_period, self.adx_period):
            return None
        
        # Detect regime
        regime_signal = self._detect_regime(features)
        self.current_regime = regime_signal
        
        return regime_signal
    
    def _detect_regime(self, features: Dict[str, float]) -> RegimeSignal:
        """Detect current market regime based on features"""
        
        # Extract key features
        adx = features.get('adx_14', 0)
        plus_di = features.get('plus_di_14', 0)
        minus_di = features.get('minus_di_14', 0)
        rsi = features.get('rsi_14', 50)
        atr = features.get('atr_14', 0)
        vol = features.get('realized_vol_30', 0)
        price = features.get('price', 0)
        volume = features.get('volume', 0)
        
        # Calculate additional metrics
        price_trend = self._calculate_price_trend()
        vol_regime = self._classify_volatility(vol)
        liquidity_score = self._assess_liquidity(volume, atr)
        
        # Regime detection logic
        confidence = 0.0
        regime_type = RegimeType.RANGE  # Default
        
        # 1. Strong Trend Detection
        if adx > self.thresholds['adx_strong']:
            if plus_di > minus_di:
                regime_type = RegimeType.TREND_UP
                confidence = min(0.9, adx / 50.0)
            else:
                regime_type = RegimeType.TREND_DOWN
                confidence = min(0.9, adx / 50.0)
        
        # 2. Moderate Trend Detection
        elif adx > self.thresholds['adx_trend']:
            if plus_di > minus_di and rsi > 50:
                regime_type = RegimeType.TREND_UP
                confidence = min(0.7, adx / 40.0)
            elif minus_di > plus_di and rsi < 50:
                regime_type = RegimeType.TREND_DOWN
                confidence = min(0.7, adx / 40.0)
            else:
                regime_type = RegimeType.RANGE
                confidence = 0.6
        
        # 3. High Volatility Regime
        elif vol > self.thresholds['vol_high']:
            regime_type = RegimeType.HIGH_VOL
            confidence = min(0.8, vol / 1.0)
        
        # 4. Low Liquidity Regime
        elif liquidity_score < 0.3:
            regime_type = RegimeType.LOW_LIQUIDITY
            confidence = 0.7
        
        # 5. Breakout Detection
        elif self._detect_breakout(atr, vol):
            regime_type = RegimeType.BREAKOUT
            confidence = 0.75
        
        # 6. Default Range Regime
        else:
            regime_type = RegimeType.RANGE
            confidence = 0.5
        
        return RegimeSignal(
            regime=regime_type,
            confidence=confidence,
            features=features.copy()
        )
    
    def _calculate_price_trend(self) -> float:
        """Calculate price trend strength over recent history"""
        if len(self.history) < 10:
            return 0.0
        
        recent_prices = [h.get('price', 0) for h in self.history[-10:]]
        if not all(recent_prices):
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(recent_prices))
        y = np.array(recent_prices)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope / recent_prices[-1]  # Normalize by current price
        
        return 0.0
    
    def _classify_volatility(self, vol: float) -> str:
        """Classify volatility regime"""
        if vol > self.thresholds['vol_high']:
            return "high"
        elif vol < self.thresholds['vol_low']:
            return "low"
        else:
            return "normal"
    
    def _assess_liquidity(self, volume: float, atr: float) -> float:
        """Assess market liquidity based on volume and ATR"""
        if len(self.history) < 20:
            return 0.5
        
        # Get recent volume and ATR history
        recent_volumes = [h.get('volume', 0) for h in self.history[-20:]]
        recent_atrs = [h.get('atr_14', 0) for h in self.history[-20:]]
        
        if not all(recent_volumes) or not all(recent_atrs):
            return 0.5
        
        # Compare current metrics to recent averages
        avg_volume = np.mean(recent_volumes)
        avg_atr = np.mean(recent_atrs)
        
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        atr_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        
        # Higher volume and lower ATR indicate better liquidity
        liquidity_score = (volume_ratio) / (1 + atr_ratio)
        
        return min(1.0, max(0.0, liquidity_score))
    
    def _detect_breakout(self, atr: float, vol: float) -> bool:
        """Detect potential breakout conditions"""
        if len(self.history) < self.atr_period:
            return False
        
        # Get recent ATR history
        recent_atrs = [h.get('atr_14', 0) for h in self.history[-self.atr_period:]]
        
        if not all(recent_atrs):
            return False
        
        avg_atr = np.mean(recent_atrs[:-1])  # Exclude current ATR
        
        # Breakout if current ATR significantly higher than recent average
        atr_expansion = atr / avg_atr if avg_atr > 0 else 1.0
        
        return (atr_expansion > self.thresholds['atr_expansion'] and 
                vol > self.thresholds['vol_low'])
    
    def get_current_regime(self) -> Optional[RegimeSignal]:
        """Get current regime signal"""
        return self.current_regime
    
    def get_regime_rules(self) -> str:
        """Get decision rules for current regime"""
        if not self.current_regime:
            return "Insufficient data for regime detection"
        
        regime = self.current_regime.regime
        
        rules_map = {
            RegimeType.TREND_UP: "Strong uptrend detected. Use momentum/DCA strategies. Avoid mean reversion.",
            RegimeType.TREND_DOWN: "Strong downtrend detected. Use short momentum or defensive strategies.",
            RegimeType.RANGE: "Range-bound market. Use mean reversion, grid, or scalping strategies.",
            RegimeType.HIGH_VOL: "High volatility regime. Reduce position sizes, use wider stops.",
            RegimeType.LOW_LIQUIDITY: "Low liquidity conditions. Use smaller sizes, limit orders.",
            RegimeType.BREAKOUT: "Breakout conditions detected. Monitor for momentum continuation."
        }
        
        return rules_map.get(regime, "Unknown regime")
    
    def reset(self):
        """Reset regime detector state"""
        self.history.clear()
        self.current_regime = None
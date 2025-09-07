# backend/app/core/risk.py
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for position sizing and monitoring"""
    position_size_usd: float
    max_loss_usd: float
    leverage_used: float
    margin_required: float
    risk_percentage: float
    confidence_level: float

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size_pct: float = 10.0  # % of capital per position
    max_daily_loss_pct: float = 5.0      # % of capital daily loss limit
    max_drawdown_pct: float = 15.0       # % maximum drawdown
    max_leverage: float = 3.0            # Maximum leverage allowed
    max_concentration_pct: float = 25.0   # % in single asset
    max_correlation: float = 0.7         # Maximum portfolio correlation

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, capital_usd: float):
        self.capital_usd = capital_usd
        self.daily_pnl = 0.0
        self.peak_capital = capital_usd
        self.current_drawdown = 0.0
        self.positions: Dict[str, Dict] = {}
        self.daily_trades = 0
        self.last_reset_date = datetime.utcnow().date()
        
        # Global risk limits
        self.limits = RiskLimits(
            max_daily_loss_pct=settings.MAX_DAILY_RISK_GLOBAL,
            max_drawdown_pct=settings.MAX_DRAWDOWN_GLOBAL,
            max_leverage=settings.MAX_LEVERAGE_GLOBAL
        )
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss_price: float, risk_pct: float,
                              atr: Optional[float] = None) -> RiskMetrics:
        """Calculate position size based on risk parameters"""
        try:
            # Risk amount in USD
            risk_amount = self.capital_usd * (risk_pct / 100)
            
            # Price risk per unit
            if stop_loss_price > 0:
                price_risk = abs(entry_price - stop_loss_price)
            elif atr:
                # Use ATR-based stop if no explicit stop loss
                price_risk = atr * 2.0  # 2x ATR stop
            else:
                # Fallback to 2% price risk
                price_risk = entry_price * 0.02
            
            # Position size calculation
            if price_risk > 0:
                position_size_units = risk_amount / price_risk
                position_size_usd = position_size_units * entry_price
            else:
                position_size_usd = 0.0
                position_size_units = 0.0
            
            # Check against maximum position size
            max_position_usd = self.capital_usd * (self.limits.max_position_size_pct / 100)
            if position_size_usd > max_position_usd:
                position_size_usd = max_position_usd
                position_size_units = position_size_usd / entry_price
                actual_risk_pct = (position_size_usd * (price_risk / entry_price)) / self.capital_usd * 100
            else:
                actual_risk_pct = risk_pct
            
            # Calculate leverage and margin
            leverage_used = position_size_usd / self.capital_usd
            margin_required = position_size_usd / min(leverage_used, self.limits.max_leverage) if leverage_used > 1 else position_size_usd
            
            # Calculate confidence level based on risk factors
            confidence = self._calculate_confidence(symbol, risk_pct, leverage_used)
            
            return RiskMetrics(
                position_size_usd=position_size_usd,
                max_loss_usd=risk_amount,
                leverage_used=leverage_used,
                margin_required=margin_required,
                risk_percentage=actual_risk_pct,
                confidence_level=confidence
            )
            
        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            # Return minimal risk position
            return RiskMetrics(
                position_size_usd=self.capital_usd * 0.01,  # 1% of capital
                max_loss_usd=self.capital_usd * 0.005,     # 0.5% max loss
                leverage_used=1.0,
                margin_required=self.capital_usd * 0.01,
                risk_percentage=0.5,
                confidence_level=0.3
            )
    
    def validate_trade(self, symbol: str, position_size_usd: float, 
                      side: str, leverage: float = 1.0) -> Tuple[bool, str]:
        """Validate if trade meets risk criteria"""
        try:
            # Check daily loss limit
            if self.daily_pnl < 0 and abs(self.daily_pnl) >= self.capital_usd * (self.limits.max_daily_loss_pct / 100):
                return False, f"Daily loss limit exceeded ({self.limits.max_daily_loss_pct}%)"
            
            # Check drawdown limit
            current_capital = self.capital_usd + self.daily_pnl
            self.current_drawdown = (self.peak_capital - current_capital) / self.peak_capital * 100
            
            if self.current_drawdown >= self.limits.max_drawdown_pct:
                return False, f"Maximum drawdown limit exceeded ({self.limits.max_drawdown_pct}%)"
            
            # Check leverage limit
            if leverage > self.limits.max_leverage:
                return False, f"Leverage {leverage:.1f}x exceeds limit {self.limits.max_leverage}x"
            
            # Check position size limit
            max_position_usd = self.capital_usd * (self.limits.max_position_size_pct / 100)
            if position_size_usd > max_position_usd:
                return False, f"Position size ${position_size_usd:,.0f} exceeds limit ${max_position_usd:,.0f}"
            
            # Check concentration limit
            current_exposure = self._calculate_symbol_exposure(symbol)
            new_exposure = (current_exposure + position_size_usd) / self.capital_usd * 100
            
            if new_exposure > self.limits.max_concentration_pct:
                return False, f"Symbol concentration {new_exposure:.1f}% exceeds limit {self.limits.max_concentration_pct}%"
            
            # Check total portfolio exposure
            total_exposure = self._calculate_total_exposure() + position_size_usd
            if total_exposure > self.capital_usd * 2:  # Max 200% exposure
                return False, "Total portfolio exposure limit exceeded"
            
            return True, "Trade approved"
            
        except Exception as e:
            logger.error(f"Trade validation failed: {e}")
            return False, f"Validation error: {str(e)}"
    
    def update_position(self, symbol: str, position_size_usd: float, 
                       entry_price: float, side: str, leverage: float = 1.0):
        """Update position tracking"""
        try:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "size_usd": 0.0,
                    "avg_price": 0.0,
                    "side": side,
                    "leverage": leverage,
                    "unrealized_pnl": 0.0,
                    "last_update": datetime.utcnow()
                }
            
            position = self.positions[symbol]
            
            # Update position size and average price
            current_size = position["size_usd"]
            current_price = position["avg_price"]
            
            if side == position["side"]:
                # Adding to position
                new_size = current_size + position_size_usd
                new_avg_price = ((current_size * current_price) + (position_size_usd * entry_price)) / new_size
                
                position["size_usd"] = new_size
                position["avg_price"] = new_avg_price
            else:
                # Reducing or reversing position
                position["size_usd"] = abs(current_size - position_size_usd)
                if position["size_usd"] == 0:
                    position["avg_price"] = 0.0
                    position["side"] = None
                elif position_size_usd > current_size:
                    # Position reversal
                    position["side"] = side
                    position["avg_price"] = entry_price
            
            position["leverage"] = leverage
            position["last_update"] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Position update failed: {e}")
    
    def calculate_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized PnL for a position"""
        try:
            if symbol not in self.positions:
                return 0.0
            
            position = self.positions[symbol]
            if position["size_usd"] == 0 or not position["side"]:
                return 0.0
            
            entry_price = position["avg_price"]
            size_usd = position["size_usd"]
            side = position["side"]
            
            if side == "long":
                pnl = (current_price - entry_price) / entry_price * size_usd
            else:  # short
                pnl = (entry_price - current_price) / entry_price * size_usd
            
            position["unrealized_pnl"] = pnl
            return pnl
            
        except Exception as e:
            logger.error(f"PnL calculation failed for {symbol}: {e}")
            return 0.0
    
    def check_kill_switches(self) -> List[str]:
        """Check for kill switch conditions"""
        warnings = []
        
        try:
            # Daily loss kill switch
            daily_loss_pct = abs(self.daily_pnl) / self.capital_usd * 100
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                warnings.append(f"KILL SWITCH: Daily loss {daily_loss_pct:.1f}% exceeds limit")
            
            # Drawdown kill switch
            if self.current_drawdown >= self.limits.max_drawdown_pct:
                warnings.append(f"KILL SWITCH: Drawdown {self.current_drawdown:.1f}% exceeds limit")
            
            # Check individual position losses
            for symbol, position in self.positions.items():
                if position["size_usd"] > 0:
                    position_loss_pct = abs(position["unrealized_pnl"]) / position["size_usd"] * 100
                    if position_loss_pct >= 20:  # 20% position loss limit
                        warnings.append(f"KILL SWITCH: {symbol} position loss {position_loss_pct:.1f}%")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Kill switch check failed: {e}")
            return [f"Kill switch check error: {str(e)}"]
    
    def reset_daily_metrics(self):
        """Reset daily metrics at start of new trading day"""
        current_date = datetime.utcnow().date()
        
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
            
            # Update peak capital if needed
            current_capital = self.capital_usd + sum(
                pos["unrealized_pnl"] for pos in self.positions.values()
            )
            if current_capital > self.peak_capital:
                self.peak_capital = current_capital
    
    def _calculate_confidence(self, symbol: str, risk_pct: float, leverage: float) -> float:
        """Calculate confidence level for position"""
        confidence = 1.0
        
        # Reduce confidence for high risk
        if risk_pct > 2.0:
            confidence *= 0.8
        
        # Reduce confidence for high leverage
        if leverage > 2.0:
            confidence *= 0.7
        
        # Reduce confidence if many positions already open
        if len(self.positions) > 5:
            confidence *= 0.9
        
        # Reduce confidence if in drawdown
        if self.current_drawdown > 5.0:
            confidence *= 0.8
        
        return max(0.3, confidence)
    
    def _calculate_symbol_exposure(self, symbol: str) -> float:
        """Calculate current USD exposure to a symbol"""
        if symbol in self.positions:
            return self.positions[symbol]["size_usd"]
        return 0.0
    
    def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        return sum(pos["size_usd"] for pos in self.positions.values())
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio risk summary"""
        try:
            total_exposure = self._calculate_total_exposure()
            total_unrealized_pnl = sum(pos["unrealized_pnl"] for pos in self.positions.values())
            
            return {
                "capital_usd": self.capital_usd,
                "daily_pnl": self.daily_pnl,
                "unrealized_pnl": total_unrealized_pnl,
                "total_pnl": self.daily_pnl + total_unrealized_pnl,
                "current_drawdown_pct": self.current_drawdown,
                "total_exposure_usd": total_exposure,
                "exposure_ratio": total_exposure / self.capital_usd,
                "positions_count": len([p for p in self.positions.values() if p["size_usd"] > 0]),
                "daily_trades": self.daily_trades,
                "risk_limits": {
                    "max_daily_loss_pct": self.limits.max_daily_loss_pct,
                    "max_drawdown_pct": self.limits.max_drawdown_pct,
                    "max_leverage": self.limits.max_leverage
                }
            }
            
        except Exception as e:
            logger.error(f"Portfolio summary calculation failed: {e}")
            return {"error": str(e)}
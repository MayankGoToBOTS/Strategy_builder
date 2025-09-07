# backend/app/services/risk_guardian/guardian.py
import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from app.deps.redis_client import RedisManager
from app.deps.mongo_client import MongoManager
from app.core.config import settings
from app.core.risk import RiskManager

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class RiskAlert:
    """Risk alert message"""
    strategy_id: str
    alert_level: AlertLevel
    alert_type: str
    message: str
    timestamp: datetime
    data: Dict = None
    
    def to_dict(self):
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat(),
            "alert_level": self.alert_level.value
        }

@dataclass
class PositionUpdate:
    """Position update from trading engine"""
    strategy_id: str
    symbol: str
    side: str  # "long" or "short"
    size_usd: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime

@dataclass
class TradeExecution:
    """Trade execution notification"""
    strategy_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    fee: float
    realized_pnl: Optional[float]
    timestamp: datetime

class RiskGuardian:
    """
    Real-time risk monitoring and enforcement system
    
    Responsibilities:
    - Monitor portfolio PnL and drawdown
    - Enforce daily loss limits
    - Track position concentrations
    - Monitor volatility and market conditions
    - Trigger kill switches when needed
    - Generate risk alerts and notifications
    """
    
    def __init__(self):
        self.running = False
        self.monitored_strategies: Set[str] = set()
        self.strategy_risk_managers: Dict[str, RiskManager] = {}
        self.last_health_check = datetime.utcnow()
        
        # Alert thresholds
        self.alert_thresholds = {
            "daily_loss_warning": 3.0,     # 3% daily loss warning
            "daily_loss_critical": 5.0,    # 5% daily loss critical
            "drawdown_warning": 10.0,      # 10% drawdown warning
            "drawdown_critical": 15.0,     # 15% drawdown critical
            "position_size_warning": 15.0, # 15% position size warning
            "volatility_spike": 2.0,       # 2x normal volatility
        }
        
        # Risk state tracking
        self.portfolio_metrics: Dict[str, Dict] = {}
        self.active_alerts: Dict[str, List[RiskAlert]] = {}
        self.kill_switches_triggered: Set[str] = set()
        
        # Monitoring intervals
        self.monitoring_interval = 5  # seconds
        self.health_check_interval = 60  # seconds
        self.alert_cooldown = 300  # 5 minutes
        
    async def start(self):
        """Start the risk guardian service"""
        if self.running:
            return
            
        self.running = True
        logger.info("Risk Guardian service starting...")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._portfolio_monitor()),
            asyncio.create_task(self._market_condition_monitor()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._health_monitor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Risk Guardian service error: {e}")
        finally:
            self.running = False
    
    async def stop(self):
        """Stop the risk guardian service"""
        self.running = False
        logger.info("Risk Guardian service stopped")
    
    async def register_strategy(self, strategy_id: str, initial_capital: float):
        """Register a strategy for monitoring"""
        try:
            self.monitored_strategies.add(strategy_id)
            self.strategy_risk_managers[strategy_id] = RiskManager(initial_capital)
            self.portfolio_metrics[strategy_id] = {
                "initial_capital": initial_capital,
                "current_capital": initial_capital,
                "daily_pnl": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
                "positions": {},
                "last_update": datetime.utcnow()
            }
            
            logger.info(f"Strategy {strategy_id} registered for risk monitoring")
            
        except Exception as e:
            logger.error(f"Failed to register strategy {strategy_id}: {e}")
    
    async def unregister_strategy(self, strategy_id: str):
        """Unregister a strategy from monitoring"""
        try:
            self.monitored_strategies.discard(strategy_id)
            self.strategy_risk_managers.pop(strategy_id, None)
            self.portfolio_metrics.pop(strategy_id, None)
            self.active_alerts.pop(strategy_id, None)
            self.kill_switches_triggered.discard(strategy_id)
            
            logger.info(f"Strategy {strategy_id} unregistered from risk monitoring")
            
        except Exception as e:
            logger.error(f"Failed to unregister strategy {strategy_id}: {e}")
    
    async def process_position_update(self, update: PositionUpdate):
        """Process position update from trading engine"""
        try:
            strategy_id = update.strategy_id
            
            if strategy_id not in self.monitored_strategies:
                return
            
            # Update portfolio metrics
            metrics = self.portfolio_metrics[strategy_id]
            metrics["positions"][update.symbol] = {
                "side": update.side,
                "size_usd": update.size_usd,
                "entry_price": update.entry_price,
                "current_price": update.current_price,
                "unrealized_pnl": update.unrealized_pnl,
                "timestamp": update.timestamp
            }
            
            # Update total unrealized PnL
            total_unrealized = sum(
                pos["unrealized_pnl"] for pos in metrics["positions"].values()
            )
            
            current_capital = metrics["initial_capital"] + metrics["daily_pnl"] + total_unrealized
            metrics["current_capital"] = current_capital
            metrics["total_pnl"] = current_capital - metrics["initial_capital"]
            metrics["last_update"] = update.timestamp
            
            # Calculate drawdown
            peak_capital = max(metrics.get("peak_capital", metrics["initial_capital"]), current_capital)
            metrics["peak_capital"] = peak_capital
            drawdown = (peak_capital - current_capital) / peak_capital * 100
            metrics["max_drawdown"] = max(metrics["max_drawdown"], drawdown)
            
            # Check risk thresholds
            await self._check_position_risks(strategy_id, update)
            await self._check_portfolio_risks(strategy_id)
            
            # Update Redis with latest metrics
            await self._update_redis_metrics(strategy_id, metrics)
            
        except Exception as e:
            logger.error(f"Failed to process position update: {e}")
    
    async def process_trade_execution(self, trade: TradeExecution):
        """Process trade execution notification"""
        try:
            strategy_id = trade.strategy_id
            
            if strategy_id not in self.monitored_strategies:
                return
            
            # Update daily PnL if this is a closing trade
            if trade.realized_pnl is not None:
                metrics = self.portfolio_metrics[strategy_id]
                metrics["daily_pnl"] += trade.realized_pnl
                
                # Check daily loss limits
                daily_loss_pct = abs(metrics["daily_pnl"]) / metrics["initial_capital"] * 100
                
                if metrics["daily_pnl"] < 0:
                    if daily_loss_pct >= self.alert_thresholds["daily_loss_critical"]:
                        await self._trigger_kill_switch(strategy_id, "Daily loss limit exceeded")
                    elif daily_loss_pct >= self.alert_thresholds["daily_loss_warning"]:
                        await self._generate_alert(
                            strategy_id, AlertLevel.WARNING, "daily_loss",
                            f"Daily loss {daily_loss_pct:.1f}% approaching limit"
                        )
            
            # Store trade in Redis for tracking
            await self._store_trade(trade)
            
        except Exception as e:
            logger.error(f"Failed to process trade execution: {e}")
    
    async def _portfolio_monitor(self):
        """Main portfolio monitoring loop"""
        while self.running:
            try:
                for strategy_id in self.monitored_strategies.copy():
                    await self._check_portfolio_risks(strategy_id)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Portfolio monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _market_condition_monitor(self):
        """Monitor market conditions and volatility"""
        while self.running:
            try:
                # Get latest market data from Redis
                redis = RedisManager.get_redis()
                
                # Check major symbols for volatility spikes
                major_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
                
                for symbol in major_symbols:
                    features_key = f"features:last:{symbol}:1m"
                    features = await RedisManager.get_hash(features_key)
                    
                    if features and "realized_vol_30" in features:
                        try:
                            current_vol = float(features["realized_vol_30"])
                            
                            # Check for volatility spikes
                            if current_vol > 1.0:  # 100% annualized volatility
                                await self._generate_market_alert(
                                    AlertLevel.WARNING, "volatility_spike",
                                    f"{symbol} volatility spike: {current_vol:.1%}"
                                )
                        except (ValueError, TypeError):
                            continue
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Market condition monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _alert_processor(self):
        """Process and distribute alerts"""
        while self.running:
            try:
                # Check for new alerts in Redis
                redis = RedisManager.get_redis()
                
                # Process alerts from the alert stream
                alert_stream = "risk.alerts"
                alerts = await redis.xread({alert_stream: '$'}, count=10, block=1000)
                
                for stream, messages in alerts:
                    for message_id, fields in messages:
                        await self._process_alert_message(fields)
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(5)
    
    async def _health_monitor(self):
        """Monitor service health and reset daily metrics"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Reset daily metrics at midnight UTC
                if (current_time.hour == 0 and 
                    self.last_health_check.date() != current_time.date()):
                    
                    await self._reset_daily_metrics()
                
                # Health check
                if (current_time - self.last_health_check).seconds >= self.health_check_interval:
                    await self._perform_health_check()
                    self.last_health_check = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_position_risks(self, strategy_id: str, update: PositionUpdate):
        """Check position-specific risks"""
        try:
            metrics = self.portfolio_metrics[strategy_id]
            
            # Check position size
            position_pct = update.size_usd / metrics["initial_capital"] * 100
            
            if position_pct > self.alert_thresholds["position_size_warning"]:
                await self._generate_alert(
                    strategy_id, AlertLevel.WARNING, "position_size",
                    f"Large position: {update.symbol} {position_pct:.1f}% of capital"
                )
            
            # Check single position loss
            if update.unrealized_pnl < 0:
                position_loss_pct = abs(update.unrealized_pnl) / update.size_usd * 100
                
                if position_loss_pct > 20:  # 20% position loss
                    await self._generate_alert(
                        strategy_id, AlertLevel.CRITICAL, "position_loss",
                        f"Large position loss: {update.symbol} -{position_loss_pct:.1f}%"
                    )
            
        except Exception as e:
            logger.error(f"Position risk check failed: {e}")
    
    async def _check_portfolio_risks(self, strategy_id: str):
        """Check portfolio-level risks"""
        try:
            if strategy_id not in self.portfolio_metrics:
                return
            
            metrics = self.portfolio_metrics[strategy_id]
            
            # Check drawdown
            drawdown = metrics["max_drawdown"]
            
            if drawdown >= self.alert_thresholds["drawdown_critical"]:
                await self._trigger_kill_switch(strategy_id, f"Maximum drawdown {drawdown:.1f}% exceeded")
            elif drawdown >= self.alert_thresholds["drawdown_warning"]:
                await self._generate_alert(
                    strategy_id, AlertLevel.WARNING, "drawdown",
                    f"High drawdown: {drawdown:.1f}%"
                )
            
            # Check daily P&L
            daily_pnl = metrics["daily_pnl"]
            if daily_pnl < 0:
                daily_loss_pct = abs(daily_pnl) / metrics["initial_capital"] * 100
                
                if daily_loss_pct >= self.alert_thresholds["daily_loss_critical"]:
                    await self._trigger_kill_switch(strategy_id, f"Daily loss {daily_loss_pct:.1f}% exceeded limit")
                elif daily_loss_pct >= self.alert_thresholds["daily_loss_warning"]:
                    await self._generate_alert(
                        strategy_id, AlertLevel.WARNING, "daily_loss",
                        f"Daily loss: {daily_loss_pct:.1f}%"
                    )
            
        except Exception as e:
            logger.error(f"Portfolio risk check failed: {e}")
    
    async def _trigger_kill_switch(self, strategy_id: str, reason: str):
        """Trigger emergency kill switch for a strategy"""
        try:
            if strategy_id in self.kill_switches_triggered:
                return  # Already triggered
            
            self.kill_switches_triggered.add(strategy_id)
            
            # Generate emergency alert
            await self._generate_alert(
                strategy_id, AlertLevel.EMERGENCY, "kill_switch",
                f"KILL SWITCH TRIGGERED: {reason}"
            )
            
            # Publish kill switch signal to Redis
            redis = RedisManager.get_redis()
            kill_switch_data = {
                "strategy_id": strategy_id,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
                "action": "stop_all_trading"
            }
            
            await redis.xadd(f"risk.killswitch.{strategy_id}", kill_switch_data)
            
            logger.critical(f"KILL SWITCH TRIGGERED for strategy {strategy_id}: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to trigger kill switch: {e}")
    
    async def _generate_alert(self, strategy_id: str, level: AlertLevel, 
                            alert_type: str, message: str, data: Dict = None):
        """Generate and distribute risk alert"""
        try:
            alert = RiskAlert(
                strategy_id=strategy_id,
                alert_level=level,
                alert_type=alert_type,
                message=message,
                timestamp=datetime.utcnow(),
                data=data
            )
            
            # Store alert
            if strategy_id not in self.active_alerts:
                self.active_alerts[strategy_id] = []
            
            self.active_alerts[strategy_id].append(alert)
            
            # Keep only recent alerts
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.active_alerts[strategy_id] = [
                a for a in self.active_alerts[strategy_id] 
                if a.timestamp > cutoff_time
            ]
            
            # Publish alert to Redis
            redis = RedisManager.get_redis()
            await redis.xadd("risk.alerts", alert.to_dict())
            
            # Log alert
            log_level = {
                AlertLevel.INFO: logger.info,
                AlertLevel.WARNING: logger.warning,
                AlertLevel.CRITICAL: logger.critical,
                AlertLevel.EMERGENCY: logger.critical
            }.get(level, logger.info)
            
            log_level(f"Risk Alert [{level.value.upper()}] {strategy_id}: {message}")
            
        except Exception as e:
            logger.error(f"Failed to generate alert: {e}")
    
    async def _generate_market_alert(self, level: AlertLevel, alert_type: str, message: str):
        """Generate market-wide alert"""
        await self._generate_alert("MARKET", level, alert_type, message)
    
    async def _update_redis_metrics(self, strategy_id: str, metrics: Dict):
        """Update strategy metrics in Redis"""
        try:
            redis = RedisManager.get_redis()
            metrics_key = f"risk:metrics:{strategy_id}"
            
            # Convert datetime objects to ISO strings
            serializable_metrics = metrics.copy()
            for key, value in serializable_metrics.items():
                if isinstance(value, datetime):
                    serializable_metrics[key] = value.isoformat()
                elif key == "positions":
                    # Convert position timestamps
                    for pos_symbol, pos_data in value.items():
                        if "timestamp" in pos_data and isinstance(pos_data["timestamp"], datetime):
                            pos_data["timestamp"] = pos_data["timestamp"].isoformat()
            
            await RedisManager.set_json(metrics_key, serializable_metrics, ex=3600)  # 1 hour expiry
            
        except Exception as e:
            logger.error(f"Failed to update Redis metrics: {e}")
    
    async def _store_trade(self, trade: TradeExecution):
        """Store trade execution in Redis"""
        try:
            redis = RedisManager.get_redis()
            trade_data = {
                "strategy_id": trade.strategy_id,
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.quantity,
                "price": trade.price,
                "fee": trade.fee,
                "realized_pnl": trade.realized_pnl,
                "timestamp": trade.timestamp.isoformat()
            }
            
            await redis.xadd(f"trades.{trade.strategy_id}", trade_data)
            
        except Exception as e:
            logger.error(f"Failed to store trade: {e}")
    
    async def _reset_daily_metrics(self):
        """Reset daily metrics for all strategies"""
        try:
            for strategy_id in self.monitored_strategies:
                if strategy_id in self.portfolio_metrics:
                    self.portfolio_metrics[strategy_id]["daily_pnl"] = 0.0
            
            logger.info("Daily metrics reset completed")
            
        except Exception as e:
            logger.error(f"Failed to reset daily metrics: {e}")
    
    async def _perform_health_check(self):
        """Perform service health check"""
        try:
            # Check Redis connectivity
            redis = RedisManager.get_redis()
            await redis.ping()
            
            # Check MongoDB connectivity
            strategies_db = MongoManager.get_strategies_db()
            await strategies_db.command("ping")
            
            # Log health status
            logger.info(f"Risk Guardian health check: OK - Monitoring {len(self.monitored_strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _process_alert_message(self, fields: Dict):
        """Process alert message from Redis stream"""
        try:
            # This would handle external alert messages
            # For now, just log them
            logger.info(f"Received alert message: {fields}")
            
        except Exception as e:
            logger.error(f"Failed to process alert message: {e}")
    
    def get_strategy_status(self, strategy_id: str) -> Dict:
        """Get current status for a strategy"""
        try:
            if strategy_id not in self.monitored_strategies:
                return {"error": "Strategy not monitored"}
            
            metrics = self.portfolio_metrics.get(strategy_id, {})
            alerts = self.active_alerts.get(strategy_id, [])
            
            return {
                "strategy_id": strategy_id,
                "monitored": True,
                "kill_switch_triggered": strategy_id in self.kill_switches_triggered,
                "metrics": metrics,
                "active_alerts": len(alerts),
                "recent_alerts": [a.to_dict() for a in alerts[-5:]],  # Last 5 alerts
                "last_update": metrics.get("last_update", datetime.utcnow()).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get strategy status: {e}")
            return {"error": str(e)}
    
    def get_service_status(self) -> Dict:
        """Get overall service status"""
        return {
            "running": self.running,
            "monitored_strategies": len(self.monitored_strategies),
            "kill_switches_active": len(self.kill_switches_triggered),
            "total_alerts": sum(len(alerts) for alerts in self.active_alerts.values()),
            "last_health_check": self.last_health_check.isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.last_health_check).total_seconds()
        }
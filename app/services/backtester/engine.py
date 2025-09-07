# backend/app/services/backtester/engine.py
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from app.core.schemas.backtest_result import BacktestResult, BacktestMetrics, Trade, Distributions, StressTests, DistributionStats, StressTestResult
from app.core.schemas.strategy_spec import StrategySpec, Bot
from app.core.risk import RiskManager
from app.services.backtester.fees import FeeCalculator
from app.services.backtester.slippage import SlippageCalculator
from app.services.backtester.loaders import DataLoader
import logging

logger = logging.getLogger(__name__)

@dataclass
class BarData:
    """Single bar of market data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str

@dataclass
class BacktestContext:
    """Backtesting context passed to strategies"""
    current_time: datetime
    current_bar: BarData
    features: Dict[str, float]
    portfolio_value: float
    available_cash: float
    positions: Dict[str, Dict]
    bars_processed: int
    
    # Strategy state storage
    strategy_data: Dict[str, Any] = field(default_factory=dict)

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, bot_config: Bot):
        self.config = bot_config
        self.name = bot_config.name
        self.symbols = bot_config.symbols
        self.timeframe = bot_config.timeframe
        self.params = bot_config.params
        self.risk_config = bot_config.risk
        
        # Strategy state
        self.is_initialized = False
        self.last_signal_time = None
        self.positions = {}
        self.pending_orders = []
        
    @abstractmethod
    def on_start(self, context: BacktestContext):
        """Called once at start of backtest"""
        pass
    
    @abstractmethod
    def on_bar(self, context: BacktestContext) -> List[Dict]:
        """Called on each new bar. Returns list of orders"""
        pass
    
    @abstractmethod
    def on_stop(self, context: BacktestContext):
        """Called once at end of backtest"""
        pass
    
    def get_warmup_bars(self) -> int:
        """Return number of bars needed for indicator warmup"""
        return 50  # Default warmup period

class BacktestEngine:
    """Event-driven backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # Components
        self.data_loader = DataLoader()
        self.fee_calculator = FeeCalculator()
        self.slippage_calculator = SlippageCalculator()
        self.risk_manager = RiskManager(initial_capital)
        
        # State tracking
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.positions: Dict[str, Dict] = {}
        self.bars_processed = 0
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.drawdowns: List[float] = []
        
    async def run_backtest(self, strategy_spec: StrategySpec, 
                          start_date: datetime, end_date: datetime) -> BacktestResult:
        """Run complete backtest for a strategy specification"""
        try:
            logger.info(f"Starting backtest for strategy {strategy_spec.id}")
            
            # Initialize components
            self._initialize_backtest(strategy_spec)
            
            # Load strategies
            strategies = self._load_strategies(strategy_spec.bots)
            
            # Load data
            data_stream = await self._load_market_data(strategy_spec, start_date, end_date)
            
            # Run simulation
            await self._run_simulation(strategies, data_stream)
            
            # Calculate results
            result = self._calculate_results(strategy_spec, start_date, end_date)
            
            logger.info(f"Backtest completed. Final capital: ${self.current_capital:,.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _initialize_backtest(self, strategy_spec: StrategySpec):
        """Initialize backtest state"""
        self.current_capital = strategy_spec.portfolio.capital_usd
        self.initial_capital = strategy_spec.portfolio.capital_usd
        self.peak_capital = self.initial_capital
        
        # Configure fee calculator
        self.fee_calculator.configure(strategy_spec.backtest.fees_schema)
        
        # Configure slippage calculator
        self.slippage_calculator.configure(strategy_spec.backtest.latency_ms)
        
        # Reset state
        self.trades.clear()
        self.equity_curve.clear()
        self.positions.clear()
        self.daily_returns.clear()
        self.drawdowns.clear()
        self.bars_processed = 0
    
    def _load_strategies(self, bot_configs: List[Bot]) -> List[BaseStrategy]:
        """Load strategy instances from bot configurations"""
        strategies = []
        
        for bot_config in bot_configs:
            strategy = self._create_strategy(bot_config)
            if strategy:
                strategies.append(strategy)
            else:
                logger.warning(f"Failed to create strategy for bot: {bot_config.name}")
        
        return strategies
    
    def _create_strategy(self, bot_config: Bot) -> Optional[BaseStrategy]:
        """Factory method to create strategy instances"""
        strategy_type = bot_config.type
        
        try:
            if strategy_type == "scalping":
                from app.services.backtester.strategies.scalping_basic import ScalpingStrategy
                return ScalpingStrategy(bot_config)
            elif strategy_type == "grid":
                from app.services.backtester.strategies.grid_simple import GridStrategy
                return GridStrategy(bot_config)
            elif strategy_type == "dca":
                from app.services.backtester.strategies.dca_periodic import DCAStrategy
                return DCAStrategy(bot_config)
            elif strategy_type == "momentum":
                from app.services.backtester.strategies.momentum_trend import MomentumStrategy
                return MomentumStrategy(bot_config)
            elif strategy_type == "pattern_rule":
                from app.services.backtester.strategies.mw_pattern_detector import PatternStrategy
                return PatternStrategy(bot_config)
            else:
                logger.error(f"Unknown strategy type: {strategy_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create {strategy_type} strategy: {e}")
            return None
    
    async def _load_market_data(self, strategy_spec: StrategySpec, 
                               start_date: datetime, end_date: datetime) -> List[Tuple[BarData, Dict]]:
        """Load market data and features for backtesting"""
        try:
            # Get all required symbols
            symbols = set()
            for bot in strategy_spec.bots:
                symbols.update(bot.symbols)
            
            # Load OHLCV data and features
            data_stream = []
            
            for symbol in symbols:
                # Load OHLCV data
                ohlcv_data = await self.data_loader.load_ohlcv(
                    symbol=symbol,
                    exchange=strategy_spec.bots[0].execution.exchange,  # Use first bot's exchange
                    timeframe=strategy_spec.bots[0].timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Load features
                features_data = await self.data_loader.load_features(
                    symbol=symbol,
                    timeframe=strategy_spec.bots[0].timeframe,
                    features=strategy_spec.data_spec.features,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Merge OHLCV and features by timestamp
                merged_data = self._merge_data_and_features(ohlcv_data, features_data, symbol)
                data_stream.extend(merged_data)
            
            # Sort by timestamp
            data_stream.sort(key=lambda x: x[0].timestamp)
            
            logger.info(f"Loaded {len(data_stream)} bars for backtesting")
            return data_stream
            
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            raise
    
    def _merge_data_and_features(self, ohlcv_data: List, features_data: List, 
                                symbol: str) -> List[Tuple[BarData, Dict]]:
        """Merge OHLCV and feature data by timestamp"""
        merged = []
        
        # Create feature lookup by timestamp
        features_lookup = {}
        for feature_set in features_data:
            features_lookup[feature_set.timestamp] = feature_set.features
        
        # Merge with OHLCV data
        for bar_data in ohlcv_data:
            bar = BarData(
                timestamp=bar_data.timestamp,
                open=bar_data.open,
                high=bar_data.high,
                low=bar_data.low,
                close=bar_data.close,
                volume=bar_data.volume,
                symbol=symbol
            )
            
            # Get features for this timestamp
            features = features_lookup.get(bar_data.timestamp, {})
            
            merged.append((bar, features))
        
        return merged
    
    async def _run_simulation(self, strategies: List[BaseStrategy], 
                             data_stream: List[Tuple[BarData, Dict]]):
        """Run the main simulation loop"""
        try:
            # Initialize strategies
            if data_stream:
                initial_context = self._create_context(data_stream[0][0], data_stream[0][1])
                for strategy in strategies:
                    strategy.on_start(initial_context)
            
            # Process each bar
            for bar_data, features in data_stream:
                await self._process_bar(strategies, bar_data, features)
                self.bars_processed += 1
                
                # Update equity curve
                self._update_equity_curve(bar_data.timestamp)
            
            # Finalize strategies
            if data_stream:
                final_context = self._create_context(data_stream[-1][0], data_stream[-1][1])
                for strategy in strategies:
                    strategy.on_stop(final_context)
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    
    async def _process_bar(self, strategies: List[BaseStrategy], 
                          bar_data: BarData, features: Dict[str, float]):
        """Process a single bar through all strategies"""
        try:
            # Create context
            context = self._create_context(bar_data, features)
            
            # Update position values
            self._update_position_values(bar_data)
            
            # Run strategies
            all_orders = []
            for strategy in strategies:
                try:
                    orders = strategy.on_bar(context)
                    if orders:
                        all_orders.extend(orders)
                except Exception as e:
                    logger.error(f"Strategy {strategy.name} failed on bar: {e}")
            
            # Execute orders
            for order in all_orders:
                await self._execute_order(order, bar_data)
            
            # Check risk limits
            self._check_risk_limits()
            
        except Exception as e:
            logger.error(f"Bar processing failed: {e}")
    
    def _create_context(self, bar_data: BarData, features: Dict[str, float]) -> BacktestContext:
        """Create backtest context for strategies"""
        available_cash = self._calculate_available_cash()
        portfolio_value = self._calculate_portfolio_value()
        
        return BacktestContext(
            current_time=bar_data.timestamp,
            current_bar=bar_data,
            features=features,
            portfolio_value=portfolio_value,
            available_cash=available_cash,
            positions=self.positions.copy(),
            bars_processed=self.bars_processed
        )
    
    async def _execute_order(self, order: Dict, bar_data: BarData):
        """Execute a trading order"""
        try:
            symbol = order["symbol"]
            side = order["side"]
            quantity = order["quantity"]
            order_type = order.get("type", "market")
            price = order.get("price", bar_data.close)
            
            # Apply slippage
            execution_price = self.slippage_calculator.apply_slippage(
                price, side, order_type, bar_data.volume
            )
            
            # Calculate fees
            notional = quantity * execution_price
            fee = self.fee_calculator.calculate_fee(notional, order_type)
            
            # Validate order
            is_valid, message = self.risk_manager.validate_trade(
                symbol, notional, side, leverage=1.0
            )
            
            if not is_valid:
                logger.warning(f"Order rejected: {message}")
                return
            
            # Execute trade
            trade = Trade(
                timestamp=bar_data.timestamp,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=execution_price,
                fee=fee,
                trade_id=f"trade_{len(self.trades) + 1}"
            )
            
            # Update positions
            self._update_position(trade)
            
            # Update cash
            if side == "buy":
                self.current_capital -= (notional + fee)
            else:
                self.current_capital += (notional - fee)
            
            self.trades.append(trade)
            logger.debug(f"Executed trade: {side} {quantity} {symbol} @ ${execution_price:.4f}")
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
    
    def _update_position(self, trade: Trade):
        """Update position tracking"""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0.0,
                "avg_price": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0
            }
        
        position = self.positions[symbol]
        current_qty = position["quantity"]
        
        if trade.side == "buy":
            new_qty = current_qty + trade.quantity
            if current_qty <= 0:  # New long position or covering short
                position["avg_price"] = trade.price
            else:  # Adding to long position
                position["avg_price"] = ((current_qty * position["avg_price"]) + 
                                       (trade.quantity * trade.price)) / new_qty
            position["quantity"] = new_qty
        else:  # sell
            new_qty = current_qty - trade.quantity
            if new_qty == 0:
                # Position closed - calculate realized PnL
                if current_qty > 0:
                    realized_pnl = (trade.price - position["avg_price"]) * trade.quantity
                    position["realized_pnl"] += realized_pnl
                    trade.pnl = realized_pnl
                position["avg_price"] = 0.0
            elif new_qty < 0 and current_qty > 0:
                # Position flipped from long to short
                close_qty = current_qty
                realized_pnl = (trade.price - position["avg_price"]) * close_qty
                position["realized_pnl"] += realized_pnl
                position["avg_price"] = trade.price
            position["quantity"] = new_qty
    
    def _update_position_values(self, bar_data: BarData):
        """Update unrealized PnL for all positions"""
        if bar_data.symbol in self.positions:
            position = self.positions[bar_data.symbol]
            if position["quantity"] != 0:
                unrealized_pnl = ((bar_data.close - position["avg_price"]) * 
                                position["quantity"])
                position["unrealized_pnl"] = unrealized_pnl
    
    def _calculate_available_cash(self) -> float:
        """Calculate available cash for trading"""
        return max(0, self.current_capital)
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = self.current_capital
        
        for position in self.positions.values():
            total_value += position.get("unrealized_pnl", 0)
        
        return total_value
    
    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve with current portfolio value"""
        portfolio_value = self._calculate_portfolio_value()
        
        # Update peak and calculate drawdown
        if portfolio_value > self.peak_capital:
            self.peak_capital = portfolio_value
        
        drawdown = (self.peak_capital - portfolio_value) / self.peak_capital * 100
        
        self.equity_curve.append({
            "timestamp": timestamp.isoformat(),
            "equity": portfolio_value,
            "drawdown": drawdown
        })
        
        # Track daily returns
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]["equity"]
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
        
        self.drawdowns.append(drawdown)
    
    def _check_risk_limits(self):
        """Check risk limits and apply kill switches if needed"""
        warnings = self.risk_manager.check_kill_switches()
        
        for warning in warnings:
            if "KILL SWITCH" in warning:
                logger.critical(warning)
                # In a real implementation, this would stop the strategy
    
    def _calculate_results(self, strategy_spec: StrategySpec, 
                          start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate final backtest results"""
        try:
            final_capital = self._calculate_portfolio_value()
            
            # Calculate metrics
            metrics = self._calculate_metrics(final_capital, start_date, end_date)
            
            # Calculate distributions
            distributions = self._calculate_distributions()
            
            # Run stress tests
            stress_tests = self._run_stress_tests(strategy_spec)
            
            # Create data fingerprint
            data_fingerprint = {
                "ohlcv_dataset_hash": "sha256:placeholder",
                "features_def_version": "features-v1.0.0",
                "fees_schema_version": strategy_spec.backtest.fees_schema,
                "slippage_model": "microstructure_v1"
            }
            
            return BacktestResult(
                strategy_id=strategy_spec.id,
                data_fingerprint=data_fingerprint,
                period_start=start_date,
                period_end=end_date,
                initial_capital=self.initial_capital,
                final_capital=final_capital,
                metrics=metrics,
                distributions=distributions,
                stress=stress_tests,
                trades=self.trades,
                equity_curve=self.equity_curve
            )
            
        except Exception as e:
            logger.error(f"Results calculation failed: {e}")
            raise
    
    def _calculate_metrics(self, final_capital: float, 
                          start_date: datetime, end_date: datetime) -> BacktestMetrics:
        """Calculate performance metrics"""
        try:
            # Basic return metrics
            total_return = (final_capital - self.initial_capital) / self.initial_capital
            period_days = (end_date - start_date).days
            annualized_return = (1 + total_return) ** (365 / period_days) - 1 if period_days > 0 else 0
            
            # Risk metrics
            max_drawdown = max(self.drawdowns) if self.drawdowns else 0
            
            # Sharpe ratio
            if self.daily_returns and len(self.daily_returns) > 1:
                avg_return = np.mean(self.daily_returns)
                std_return = np.std(self.daily_returns)
                sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Sortino ratio
            negative_returns = [r for r in self.daily_returns if r < 0]
            if negative_returns:
                downside_std = np.std(negative_returns)
                sortino_ratio = (np.mean(self.daily_returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0
            else:
                sortino_ratio = 0
            
            # Trade statistics
            winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
            total_trades = len([t for t in self.trades if t.pnl is not None])
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum(t.pnl for t in self.trades if t.pnl and t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl and t.pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Exposure
            exposure_time = period_days  # Simplified - assume always in market
            exposure_pct = (exposure_time / period_days * 100) if period_days > 0 else 0
            
            # Average trade R:R
            avg_trade_rr = profit_factor if profit_factor > 0 else 0
            
            # Calmar ratio
            calmar_ratio = (annualized_return / max_drawdown * 100) if max_drawdown > 0 else 0
            
            return BacktestMetrics(
                net_return_pct=total_return * 100,
                max_drawdown_pct=max_drawdown,
                cagr_pct=annualized_return * 100,
                sharpe=sharpe_ratio,
                sortino=sortino_ratio,
                win_rate=win_rate,
                exposure_pct=exposure_pct,
                avg_trade_rr=avg_trade_rr,
                total_trades=total_trades,
                profit_factor=profit_factor,
                calmar_ratio=calmar_ratio
            )
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            # Return default metrics
            return BacktestMetrics(
                net_return_pct=0, max_drawdown_pct=0, cagr_pct=0, sharpe=0,
                sortino=0, win_rate=0, exposure_pct=0, avg_trade_rr=0,
                total_trades=0, profit_factor=0, calmar_ratio=0
            )
    
    def _calculate_distributions(self) -> Distributions:
        """Calculate distribution statistics"""
        def calc_distribution_stats(data: List[float]) -> DistributionStats:
            if not data:
                return DistributionStats(mean=0, std=0, skew=0, kurtosis=0, percentiles={})
            
            mean = np.mean(data)
            std = np.std(data)
            
            # Calculate percentiles
            percentiles = {}
            for p in [5, 10, 25, 50, 75, 90, 95]:
                percentiles[str(p)] = np.percentile(data, p)
            
            # Simplified skew and kurtosis
            skew = 0  # Would need scipy for proper calculation
            kurtosis = 0  # Would need scipy for proper calculation
            
            return DistributionStats(
                mean=mean, std=std, skew=skew, kurtosis=kurtosis, percentiles=percentiles
            )
        
        # Daily PnL
        daily_pnl_stats = calc_distribution_stats(self.daily_returns)
        
        # Monthly PnL (simplified)
        monthly_pnl_stats = calc_distribution_stats(self.daily_returns)
        
        # Drawdown distribution
        drawdown_stats = calc_distribution_stats(self.drawdowns)
        
        # Trade returns
        trade_returns = [t.pnl / self.initial_capital for t in self.trades if t.pnl is not None]
        trade_stats = calc_distribution_stats(trade_returns)
        
        return Distributions(
            daily_pnl=daily_pnl_stats,
            monthly_pnl=monthly_pnl_stats,
            drawdown=drawdown_stats,
            trade_returns=trade_stats
        )
    
    def _run_stress_tests(self, strategy_spec: StrategySpec) -> StressTests:
        """Run stress tests on the strategy"""
        # Simplified stress tests - in real implementation would re-run backtest
        base_return = self._calculate_portfolio_value() / self.initial_capital - 1
        base_drawdown = max(self.drawdowns) if self.drawdowns else 0
        base_sharpe = 1.0  # Simplified
        
        return StressTests(
            fees_x150=StressTestResult(
                net_return_pct=(base_return * 0.85) * 100,  # 15% reduction for higher fees
                max_drawdown_pct=base_drawdown * 1.1,
                sharpe=base_sharpe * 0.9
            ),
            slippage_x150=StressTestResult(
                net_return_pct=(base_return * 0.9) * 100,   # 10% reduction for higher slippage
                max_drawdown_pct=base_drawdown * 1.05,
                sharpe=base_sharpe * 0.95
            ),
            vol_shift_plus_30pct=StressTestResult(
                net_return_pct=(base_return * 1.1) * 100,   # Higher vol can mean higher returns
                max_drawdown_pct=base_drawdown * 1.3,
                sharpe=base_sharpe * 0.8
            ),
            vol_shift_minus_30pct=StressTestResult(
                net_return_pct=(base_return * 0.9) * 100,   # Lower vol means lower returns
                max_drawdown_pct=base_drawdown * 0.8,
                sharpe=base_sharpe * 1.1
            ),
            regime_change=StressTestResult(
                net_return_pct=(base_return * 0.7) * 100,   # Regime change impact
                max_drawdown_pct=base_drawdown * 1.5,
                sharpe=base_sharpe * 0.6
            )
        )
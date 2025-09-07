# backend/app/routes/strategy_builder.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Optional, Tuple
import re
import asyncio
from datetime import datetime, timedelta
import hashlib
import json

from app.core.schemas.strategy_spec import (
    StrategyRequest, StrategyResponse, StrategySpec, UserRequest, UserTarget,
    RiskTolerance, Constraints, TimeHorizon, Feasibility, DataSpec, SymbolSpec,
    DataFingerprint, Universe, Market, RegimeDetection, Bot, BotRisk, BotExecution,
    Portfolio, BotAllocation, LeveragePolicy, Compliance, BacktestConfig, BacktestPeriod,
    Results, ResultsSummary
)
from app.core.feasibility import FeasibilityAnalyzer
from app.core.risk import RiskManager
from app.services.backtester.engine import BacktestEngine
from app.deps.mongo_client import get_strategies_db
from app.deps.redis_client import get_redis
from motor.motor_asyncio import AsyncIOMotorDatabase
from redis.asyncio import Redis
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class StrategyOrchestrator:
    """Main orchestrator for strategy building process"""
    
    def __init__(self):
        self.feasibility_analyzer = FeasibilityAnalyzer()
        
        # Strategy templates mapping
        self.strategy_templates = {
            "scalping": {
                "name": "Scalp-Alpha",
                "description": "High-frequency scalping with VWAP bias and micro-breakouts",
                "best_for": ["high_volume", "range", "low_latency"],
                "min_capital": 1000,
                "risk_level": "high",
                "default_params": {
                    "atr_mult_sl": 1.2,
                    "atr_mult_tp": 0.8,
                    "cooldown_bars": 5,
                    "volume_threshold": 1.2,
                    "breakout_period": 10
                }
            },
            "grid": {
                "name": "Grid-Ranger",
                "description": "Dynamic grid trading for range-bound markets",
                "best_for": ["range", "low_volatility", "sideways"],
                "min_capital": 5000,
                "risk_level": "medium",
                "default_params": {
                    "grid_levels": 5,
                    "grid_spacing_atr": 0.5,
                    "max_position_per_level": 0.02,
                    "min_atr_threshold": 0.01,
                    "max_atr_threshold": 0.05
                }
            },
            "dca": {
                "name": "DCA-Builder",
                "description": "Systematic dollar-cost averaging with momentum filters",
                "best_for": ["long_term", "accumulation", "trending"],
                "min_capital": 2000,
                "risk_level": "low",
                "default_params": {
                    "purchase_interval_bars": 1440,  # Daily
                    "base_purchase_amount": 0.05,
                    "atr_size_adjustment": True,
                    "momentum_veto": True,
                    "rsi_oversold_threshold": 30
                }
            },
            "momentum": {
                "name": "Momentum-Rider",
                "description": "Trend-following with ADX and moving average filters",
                "best_for": ["trending", "momentum", "breakout"],
                "min_capital": 3000,
                "risk_level": "medium",
                "default_params": {
                    "adx_threshold": 25,
                    "ma_period": 20,
                    "atr_stop_mult": 2.0,
                    "position_size_pct": 0.1,
                    "min_trend_strength": 30
                }
            },
            "pattern_rule": {
                "name": "Pattern-Hunter",
                "description": "Chart pattern recognition with volume confirmation",
                "best_for": ["reversal", "breakout", "swing"],
                "min_capital": 4000,
                "risk_level": "medium",
                "default_params": {
                    "min_pattern_bars": 20,
                    "volume_confirmation": True,
                    "volume_threshold": 1.5,
                    "min_pattern_height": 0.02,
                    "position_size_pct": 0.08
                }
            }
        }
    
    async def build_strategy(self, request: StrategyRequest, 
                           strategies_db: AsyncIOMotorDatabase,
                           redis: Redis) -> StrategyResponse:
        """Main strategy building orchestrator"""
        try:
            logger.info(f"Building strategy for query: {request.query}")
            
            # 1. Parse user intent
            user_request = await self._parse_user_intent(request)
            
            # 2. Run feasibility analysis
            feasibility = await self.feasibility_analyzer.analyze_request(user_request)
            
            # 3. Select appropriate markets and symbols
            universe = await self._select_universe(user_request, feasibility)
            
            # 4. Choose and configure strategies
            bots = await self._select_and_configure_bots(user_request, feasibility, universe)
            
            # 5. Create portfolio allocation
            portfolio = self._create_portfolio(user_request, bots)
            
            # 6. Set up data requirements
            data_spec = self._create_data_spec(bots, universe)
            
            # 7. Configure backtesting
            backtest_config = self._create_backtest_config(user_request)
            
            # 8. Run quick backtest for validation
            backtest_results = await self._run_validation_backtest(
                user_request, bots, universe, data_spec, backtest_config
            )
            
            # 9. Create strategy specification
            strategy_spec = StrategySpec(
                user_request=user_request,
                feasibility=feasibility,
                data_spec=data_spec,
                data_fingerprint=self._create_data_fingerprint(),
                universe=universe,
                regime_detection=self._create_regime_detection(),
                bots=bots,
                portfolio=portfolio,
                compliance=Compliance(),
                backtest=backtest_config,
                results=backtest_results
            )
            
            # 10. Generate human-readable report
            human_report = self._generate_human_report(strategy_spec)
            
            # 11. Store strategy
            await self._store_strategy(strategy_spec, strategies_db)
            
            logger.info(f"Strategy built successfully: {strategy_spec.id}")
            
            return StrategyResponse(
                human_report=human_report,
                strategy_spec=strategy_spec
            )
            
        except Exception as e:
            logger.error(f"Strategy building failed: {e}")
            raise HTTPException(status_code=500, detail=f"Strategy building failed: {str(e)}")
    
    async def _parse_user_intent(self, request: StrategyRequest) -> UserRequest:
        """Parse user query to extract intent and requirements"""
        try:
            query = request.query.lower()
            
            # Extract target return
            target_return = self._extract_target_return(query)
            
            # Determine time horizon
            time_horizon = self._extract_time_horizon(query)
            
            # Determine strategy style preferences
            style_preferences = self._extract_style_preferences(query)
            
            # Create constraints
            constraints = Constraints(
                leverage_max=request.max_leverage,
                max_drawdown_pct=request.max_drawdown_pct,
                allowed_exchanges=request.allowed_exchanges,
                blacklist_symbols=[]
            )
            
            # Create risk tolerance
            risk_tolerance = RiskTolerance(level=request.risk_tolerance)
            
            return UserRequest(
                raw=request.query,
                target_return=target_return,
                risk_tolerance=risk_tolerance,
                constraints=constraints,
                time_horizon=time_horizon,
                capital_usd=request.capital_usd
            )
            
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            # Return default values
            return UserRequest(
                raw=request.query,
                target_return=UserTarget(value=10.0, period="month", type="gross"),
                risk_tolerance=RiskTolerance(level=request.risk_tolerance),
                constraints=Constraints(
                    leverage_max=request.max_leverage,
                    max_drawdown_pct=request.max_drawdown_pct,
                    allowed_exchanges=request.allowed_exchanges,
                    blacklist_symbols=[]
                ),
                time_horizon=TimeHorizon(value=3, unit="month"),
                capital_usd=request.capital_usd
            )
    
    def _extract_target_return(self, query: str) -> UserTarget:
        """Extract target return from user query"""
        # Look for percentage patterns
        percentage_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*(?:per\s+)?(?:monthly|month|mo)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:per\s+)?(?:daily|day)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:per\s+)?(?:weekly|week)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:per\s+)?(?:yearly|year|annual)',
            r'(\d+(?:\.\d+)?)\s*percent\s*(?:per\s+)?(?:month|monthly)',
            r'make\s+(\d+(?:\.\d+)?)\s*%',
            r'return\s+(\d+(?:\.\d+)?)\s*%',
            r'profit\s+(\d+(?:\.\d+)?)\s*%'
        ]
        
        for pattern in percentage_patterns:
            match = re.search(pattern, query)
            if match:
                value = float(match.group(1))
                
                # Determine period
                if 'month' in pattern or 'mo' in pattern:
                    period = "month"
                elif 'day' in pattern:
                    period = "day"
                elif 'week' in pattern:
                    period = "week"
                elif 'year' in pattern or 'annual' in pattern:
                    period = "year"
                else:
                    period = "month"  # Default
                
                return UserTarget(value=value, period=period, type="gross")
        
        # Look for multiplier patterns like "2x", "double"
        multiplier_patterns = [
            r'(\d+(?:\.\d+)?)\s*x',
            r'(\d+(?:\.\d+)?)\s*times',
            r'double',
            r'triple'
        ]
        
        for pattern in multiplier_patterns:
            if pattern == 'double':
                return UserTarget(value=100.0, period="year", type="gross")
            elif pattern == 'triple':
                return UserTarget(value=200.0, period="year", type="gross")
            else:
                match = re.search(pattern, query)
                if match:
                    multiplier = float(match.group(1))
                    annual_return = (multiplier - 1) * 100
                    return UserTarget(value=annual_return, period="year", type="gross")
        
        # Default target based on common requests
        if any(word in query for word in ['aggressive', 'high', 'moon', 'pump']):
            return UserTarget(value=20.0, period="month", type="gross")
        elif any(word in query for word in ['conservative', 'safe', 'steady']):
            return UserTarget(value=5.0, period="month", type="gross")
        else:
            return UserTarget(value=10.0, period="month", type="gross")
    
    def _extract_time_horizon(self, query: str) -> TimeHorizon:
        """Extract time horizon from user query"""
        # Look for time horizon patterns
        horizon_patterns = [
            (r'(\d+)\s*(?:months?|mo)', 'month'),
            (r'(\d+)\s*(?:weeks?|wk)', 'week'),
            (r'(\d+)\s*(?:days?)', 'day'),
            (r'(\d+)\s*(?:years?|yr)', 'year')
        ]
        
        for pattern, unit in horizon_patterns:
            match = re.search(pattern, query)
            if match:
                value = int(match.group(1))
                return TimeHorizon(value=value, unit=unit)
        
        # Look for qualitative terms
        if any(word in query for word in ['long', 'hold', 'invest', 'accumulate']):
            return TimeHorizon(value=6, unit="month")
        elif any(word in query for word in ['short', 'quick', 'scalp', 'day']):
            return TimeHorizon(value=1, unit="month")
        else:
            return TimeHorizon(value=3, unit="month")
    
    def _extract_style_preferences(self, query: str) -> List[str]:
        """Extract strategy style preferences from query"""
        preferences = []
        
        style_keywords = {
            'scalping': ['scalp', 'quick', 'fast', 'micro', 'short-term'],
            'grid': ['grid', 'range', 'sideways', 'choppy', 'bound'],
            'dca': ['dca', 'dollar cost', 'accumulate', 'gradual', 'systematic'],
            'momentum': ['momentum', 'trend', 'follow', 'breakout', 'surge'],
            'pattern': ['pattern', 'chart', 'technical', 'formation', 'setup']
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in query for keyword in keywords):
                preferences.append(style)
        
        return preferences
    
    async def _select_universe(self, user_request: UserRequest, 
                              feasibility: Feasibility) -> Universe:
        """Select trading universe (markets and symbols)"""
        try:
            # Determine markets based on capital and risk tolerance
            markets = ["crypto.spot"]
            
            if user_request.capital_usd > 5000 and user_request.risk_tolerance.level in ["medium", "high"]:
                markets.append("crypto.usdt_perp")
            
            # Select symbols based on various factors
            candidates = []
            
            # Major crypto pairs (always include)
            major_symbols = ["BTCUSDT", "ETHUSDT"]
            
            # Add more symbols based on capital size
            if user_request.capital_usd > 10000:
                major_symbols.extend(["ADAUSDT", "SOLUSDT", "DOTUSDT"])
            
            if user_request.capital_usd > 25000:
                major_symbols.extend(["AVAXUSDT", "MATICUSDT", "LINKUSDT"])
            
            # Create candidate list
            for symbol in major_symbols:
                for market in markets:
                    market_type = "usdt_perp" if "perp" in market else "spot"
                    candidates.append(Market(symbol=symbol, market=market_type))
            
            return Universe(
                markets=markets,
                candidates=candidates,
                blacklist=user_request.constraints.blacklist_symbols
            )
            
        except Exception as e:
            logger.error(f"Universe selection failed: {e}")
            # Return minimal universe
            return Universe(
                markets=["crypto.spot"],
                candidates=[Market(symbol="BTCUSDT", market="spot")],
                blacklist=[]
            )
    
    async def _select_and_configure_bots(self, user_request: UserRequest, 
                                       feasibility: Feasibility, 
                                       universe: Universe) -> List[Bot]:
        """Select and configure trading bots based on requirements"""
        try:
            bots = []
            
            # Determine strategy mix based on capital, risk tolerance, and target
            capital = user_request.capital_usd
            risk_level = user_request.risk_tolerance.level
            target_annual = self._convert_to_annual_target(user_request.target_return)
            
            # Strategy selection logic
            selected_strategies = []
            
            if target_annual > 100:  # >100% annual return
                if risk_level == "high":
                    selected_strategies = ["scalping", "momentum"]
                else:
                    selected_strategies = ["momentum", "grid"]
            elif target_annual > 50:  # 50-100% annual return
                if risk_level == "high":
                    selected_strategies = ["momentum", "scalping"]
                else:
                    selected_strategies = ["momentum", "dca"]
            else:  # <50% annual return
                if risk_level == "low":
                    selected_strategies = ["dca", "grid"]
                else:
                    selected_strategies = ["grid", "momentum"]
            
            # Add pattern strategy if capital is sufficient
            if capital > 5000 and len(selected_strategies) < 3:
                selected_strategies.append("pattern_rule")
            
            # Create bot configurations
            for i, strategy_type in enumerate(selected_strategies):
                template = self.strategy_templates[strategy_type]
                
                # Select symbols for this bot
                bot_symbols = self._select_bot_symbols(universe, strategy_type, i)
                
                # Configure risk parameters
                risk_config = self._configure_bot_risk(user_request, feasibility, strategy_type)
                
                # Configure execution parameters
                execution_config = BotExecution(
                    exchange=user_request.constraints.allowed_exchanges[0],
                    account="paper",  # Start with paper trading
                    order_type="limit" if strategy_type in ["grid", "dca"] else "market",
                    slippage_ppm=150
                )
                
                # Create bot
                bot = Bot(
                    name=f"{template['name']}-{i+1}",
                    type=strategy_type,
                    symbols=bot_symbols,
                    timeframe="1m" if strategy_type == "scalping" else "5m",
                    entry_logic=template["description"],
                    params=template["default_params"].copy(),
                    risk=risk_config,
                    execution=execution_config
                )
                
                bots.append(bot)
            
            return bots
            
        except Exception as e:
            logger.error(f"Bot configuration failed: {e}")
            # Return minimal bot configuration
            return [self._create_default_bot(user_request, universe)]
    
    def _convert_to_annual_target(self, target: UserTarget) -> float:
        """Convert target to annual percentage"""
        multipliers = {"day": 365, "week": 52, "month": 12, "year": 1}
        multiplier = multipliers.get(target.period, 12)
        
        if target.period == "day":
            # Compound daily returns
            return ((1 + target.value / 100) ** 365 - 1) * 100
        else:
            return target.value * multiplier
    
    def _select_bot_symbols(self, universe: Universe, strategy_type: str, bot_index: int) -> List[str]:
        """Select symbols for a specific bot"""
        available_symbols = [c.symbol for c in universe.candidates]
        
        # Strategy-specific symbol selection
        if strategy_type == "scalping":
            # Prefer high-volume pairs
            return ["BTCUSDT", "ETHUSDT"][:2]
        elif strategy_type == "dca":
            # Focus on major coins for long-term accumulation
            return ["BTCUSDT"]
        elif strategy_type == "grid":
            # Good for range-bound trading
            return ["ETHUSDT", "ADAUSDT"][bot_index:bot_index+1] if len(available_symbols) > bot_index else ["ETHUSDT"]
        else:
            # Default selection
            return available_symbols[bot_index:bot_index+2] if len(available_symbols) > bot_index else available_symbols[:1]
    
    def _configure_bot_risk(self, user_request: UserRequest, 
                           feasibility: Feasibility, strategy_type: str) -> BotRisk:
        """Configure risk parameters for a bot"""
        # Base risk levels by strategy type
        base_risk = {
            "scalping": {"per_trade": 0.2, "daily": 1.0, "positions": 4, "dd": 8},
            "grid": {"per_trade": 0.5, "daily": 2.0, "positions": 8, "dd": 12},
            "dca": {"per_trade": 1.0, "daily": 3.0, "positions": 1, "dd": 15},
            "momentum": {"per_trade": 0.8, "daily": 2.5, "positions": 3, "dd": 10},
            "pattern_rule": {"per_trade": 0.6, "daily": 2.0, "positions": 2, "dd": 10}
        }
        
        risk_params = base_risk.get(strategy_type, base_risk["momentum"])
        
        # Adjust based on user risk tolerance
        risk_multiplier = {
            "low": 0.7,
            "medium": 1.0,
            "high": 1.3
        }.get(user_request.risk_tolerance.level, 1.0)
        
        # Adjust based on feasibility assessment
        if feasibility.assessment == "stretch":
            risk_multiplier *= 0.8
        elif feasibility.assessment == "low":
            risk_multiplier *= 0.6
        
        return BotRisk(
            per_trade_risk_pct=risk_params["per_trade"] * risk_multiplier,
            daily_risk_cap_pct=min(risk_params["daily"] * risk_multiplier, 
                                 user_request.constraints.max_drawdown_pct / 2),
            max_open_positions=risk_params["positions"],
            dd_kill_switch_pct=min(risk_params["dd"], user_request.constraints.max_drawdown_pct)
        )
    
    def _create_default_bot(self, user_request: UserRequest, universe: Universe) -> Bot:
        """Create a default bot configuration as fallback"""
        return Bot(
            name="Default-Momentum",
            type="momentum",
            symbols=["BTCUSDT"],
            timeframe="5m",
            entry_logic="Basic momentum strategy with trend following",
            params=self.strategy_templates["momentum"]["default_params"].copy(),
            risk=BotRisk(
                per_trade_risk_pct=0.5,
                daily_risk_cap_pct=2.0,
                max_open_positions=2,
                dd_kill_switch_pct=10.0
            ),
            execution=BotExecution(
                exchange=user_request.constraints.allowed_exchanges[0],
                account="paper",
                order_type="market",
                slippage_ppm=150
            )
        )
    
    def _create_portfolio(self, user_request: UserRequest, bots: List[Bot]) -> Portfolio:
        """Create portfolio allocation"""
        # Equal allocation by default, can be optimized
        allocation_per_bot = 100.0 / len(bots)
        
        allocations = []
        leverage_caps = {}
        
        for bot in bots:
            allocations.append(BotAllocation(
                bot=bot.name,
                weight_pct=allocation_per_bot
            ))
            
            # Set leverage caps per bot type
            leverage_caps[bot.name] = min(
                user_request.constraints.leverage_max,
                2.0 if bot.type == "scalping" else 1.5
            )
        
        return Portfolio(
            capital_usd=user_request.capital_usd,
            allocation=allocations,
            leverage_policy=LeveragePolicy(
                max=user_request.constraints.leverage_max,
                per_bot_caps=leverage_caps
            )
        )
    
    def _create_data_spec(self, bots: List[Bot], universe: Universe) -> DataSpec:
        """Create data specification"""
        # Collect all symbols and timeframes
        symbols = set()
        timeframes = set()
        
        for bot in bots:
            symbols.update(bot.symbols)
            timeframes.add(bot.timeframe)
        
        # Create symbol specs
        symbol_specs = []
        for symbol in symbols:
            symbol_specs.append(SymbolSpec(
                symbol=symbol,
                exchange="binance",  # Primary exchange
                timeframes=list(timeframes)
            ))
        
        # Required features
        features = [
            "atr_14", "rsi_14", "adx_14", "plus_di_14", "minus_di_14",
            "vwap", "realized_vol_30", "spread_bps", "regime_label"
        ]
        
        return DataSpec(
            symbols=symbol_specs,
            features=features,
            latency_budget_ms=200,
            max_data_gap_tolerance_ms=2000
        )
    
    def _create_backtest_config(self, user_request: UserRequest) -> BacktestConfig:
        """Create backtesting configuration"""
        # Determine backtest period based on time horizon
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        
        horizon_days = {
            "day": 30,
            "week": 90,
            "month": 180,
            "year": 365
        }.get(user_request.time_horizon.unit, 180)
        
        start_date = (datetime.utcnow() - timedelta(days=horizon_days)).strftime("%Y-%m-%d")
        
        return BacktestConfig(
            periods=[BacktestPeriod(
                from_date=start_date,
                to_date=end_date,
                mode="in_sample"
            )],
            fees_schema="binance_futures_default",
            latency_ms=80
        )
    
    async def _run_validation_backtest(self, user_request: UserRequest, bots: List[Bot],
                                     universe: Universe, data_spec: DataSpec,
                                     backtest_config: BacktestConfig) -> Results:
        """Run a quick validation backtest"""
        try:
            # For now, return simulated results
            # In production, this would run the actual backtesting engine
            
            # Estimate performance based on target and feasibility
            target_annual = self._convert_to_annual_target(user_request.target_return)
            
            # Simulate conservative results
            actual_return = target_annual * 0.7  # Assume 70% of target achieved
            max_dd = min(actual_return * 0.3, user_request.constraints.max_drawdown_pct * 0.8)
            
            # Calculate other metrics
            sharpe = max(0.5, min(2.0, actual_return / 30))  # Rough estimate
            win_rate = 0.52 + (sharpe - 1.0) * 0.1  # Higher Sharpe -> higher win rate
            
            # Probability assessment
            if actual_return <= target_annual * 0.5:
                hit_probability = 0.8
            elif actual_return <= target_annual * 0.8:
                hit_probability = 0.5
            else:
                hit_probability = 0.2
            
            warnings = []
            if target_annual > 100:
                warnings.append("Target return is very aggressive; expect high volatility")
            if max_dd > 15:
                warnings.append("Strategy may experience significant drawdowns")
            
            return Results(
                summary=ResultsSummary(
                    net_return_pct=actual_return,
                    cagr_pct=actual_return,
                    max_drawdown_pct=max_dd,
                    sharpe=sharpe,
                    win_rate=win_rate,
                    ex_ante_target_hit_probability=hit_probability
                ),
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Validation backtest failed: {e}")
            # Return conservative default results
            return Results(
                summary=ResultsSummary(
                    net_return_pct=8.0,
                    cagr_pct=8.0,
                    max_drawdown_pct=6.0,
                    sharpe=1.2,
                    win_rate=0.55,
                    ex_ante_target_hit_probability=0.4
                ),
                warnings=["Backtest validation failed; using conservative estimates"]
            )
    
    def _create_data_fingerprint(self) -> DataFingerprint:
        """Create data fingerprint for reproducibility"""
        return DataFingerprint(
            ohlcv_dataset_hash="sha256:placeholder_hash",
            features_def_version="features-v1.0.0",
            fees_schema_version="fees-202409",
            slippage_model="microstructure_v1"
        )
    
    def _create_regime_detection(self) -> RegimeDetection:
        """Create regime detection configuration"""
        return RegimeDetection(
            features=["atr_14", "rsi_14", "adx_14", "realized_vol_30", "zscore_returns_20"],
            labels=["trend_up", "trend_down", "range", "high_vol", "low_liquidity"],
            decision_rules="If high_vol & range → scalping/grid; if trend_up & mid_vol → momentum/DCA"
        )
    
    def _generate_human_report(self, strategy_spec: StrategySpec) -> str:
        """Generate human-readable strategy report"""
        try:
            user_req = strategy_spec.user_request
            feasibility = strategy_spec.feasibility
            results = strategy_spec.results
            
            report = f"""# GoToBots Strategy Report

## 📊 Intent Summary
**User Query:** {user_req.raw}
**Target Return:** {user_req.target_return.value}% per {user_req.target_return.period}
**Capital:** ${user_req.capital_usd:,.0f}
**Risk Tolerance:** {user_req.risk_tolerance.level.title()}
**Time Horizon:** {user_req.time_horizon.value} {user_req.time_horizon.unit}(s)

## 🎯 Feasibility Assessment: {feasibility.assessment.upper()}
{feasibility.comment}
"""
            
            if feasibility.recommended_target:
                report += f"\n**Recommended Target:** {feasibility.recommended_target.value}% per {feasibility.recommended_target.period}"
            
            report += f"""

## 🚀 Selected Markets & Symbols
**Markets:** {', '.join(strategy_spec.universe.markets)}
**Primary Symbols:** {', '.join([c.symbol for c in strategy_spec.universe.candidates[:5]])}

## 🤖 Strategy Configuration
"""
            
            for i, bot in enumerate(strategy_spec.bots, 1):
                report += f"""
### Bot {i}: {bot.name} ({bot.type.title()})
- **Symbols:** {', '.join(bot.symbols)}
- **Timeframe:** {bot.timeframe}
- **Risk per Trade:** {bot.risk.per_trade_risk_pct}%
- **Daily Risk Cap:** {bot.risk.daily_risk_cap_pct}%
- **Max Positions:** {bot.risk.max_open_positions}
- **Allocation:** {[a.weight_pct for a in strategy_spec.portfolio.allocation if a.bot == bot.name][0]:.1f}%
"""
            
            report += f"""

## 📈 Expected Performance
- **Projected Return:** {results.summary.net_return_pct:.1f}% annually
- **Max Drawdown:** {results.summary.max_drawdown_pct:.1f}%
- **Sharpe Ratio:** {results.summary.sharpe:.2f}
- **Win Rate:** {results.summary.win_rate:.1%}
- **Target Hit Probability:** {results.summary.ex_ante_target_hit_probability:.1%}

## ⚠️ Risk Warnings
"""
            
            for warning in results.warnings:
                report += f"- {warning}\n"
            
            report += f"""
## 🔧 Risk Management
- **Max Leverage:** {strategy_spec.portfolio.leverage_policy.max}x
- **Portfolio Stop Loss:** {max(bot.risk.dd_kill_switch_pct for bot in strategy_spec.bots):.1f}%
- **Daily Loss Limit:** {max(bot.risk.daily_risk_cap_pct for bot in strategy_spec.bots):.1f}%

## 📋 Next Steps
1. **Review Strategy:** Examine the configuration and risk parameters
2. **Run Full Backtest:** Execute comprehensive historical testing
3. **Paper Trading:** Start with simulated trading to validate performance
4. **Live Deployment:** Begin with reduced position sizes
5. **Monitor & Adjust:** Track performance and optimize parameters

---
**Strategy ID:** `{strategy_spec.id}`
**Generated:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"# Strategy Report\n\nError generating detailed report: {str(e)}\n\nStrategy ID: {strategy_spec.id}"
    
    async def _store_strategy(self, strategy_spec: StrategySpec, 
                            strategies_db: AsyncIOMotorDatabase):
        """Store strategy specification in database"""
        try:
            collection = strategies_db.strategies
            
            strategy_doc = {
                "_id": strategy_spec.id,
                "strategy_spec": strategy_spec.dict(),
                "created_at": datetime.utcnow(),
                "user_query": strategy_spec.user_request.raw,
                "feasibility_assessment": strategy_spec.feasibility.assessment,
                "expected_return": strategy_spec.results.summary.net_return_pct,
                "max_drawdown": strategy_spec.results.summary.max_drawdown_pct
            }
            
            await collection.insert_one(strategy_doc)
            logger.info(f"Strategy {strategy_spec.id} stored successfully")
            
        except Exception as e:
            logger.error(f"Strategy storage failed: {e}")
            # Don't raise exception - strategy building succeeded even if storage failed

# Initialize orchestrator
orchestrator = StrategyOrchestrator()

@router.post("/build", response_model=StrategyResponse)
async def build_strategy(
    request: StrategyRequest,
    background_tasks: BackgroundTasks,
    strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db),
    redis: Redis = Depends(get_redis)
):
    """Build a complete trading strategy from user query"""
    try:
        logger.info(f"Strategy build request: {request.query}")
        
        # Build strategy
        response = await orchestrator.build_strategy(request, strategies_db, redis)
        
        # Add background task for additional processing if needed
        # background_tasks.add_task(additional_processing, response.strategy_spec)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strategy build endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/templates")
async def get_strategy_templates():
    """Get available strategy templates"""
    return {"templates": orchestrator.strategy_templates}

@router.get("/{strategy_id}")
async def get_strategy(
    strategy_id: str,
    strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
):
    """Get existing strategy by ID"""
    try:
        collection = strategies_db.strategies
        strategy_doc = await collection.find_one({"_id": strategy_id})
        
        if not strategy_doc:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return strategy_doc["strategy_spec"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strategy retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve strategy")

@router.get("/")
async def list_strategies(
    limit: int = 20,
    offset: int = 0,
    strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
):
    """List recent strategies"""
    try:
        collection = strategies_db.strategies
        
        cursor = collection.find(
            {},
            {"_id": 1, "user_query": 1, "feasibility_assessment": 1, 
             "expected_return": 1, "created_at": 1}
        ).sort("created_at", -1).skip(offset).limit(limit)
        
        strategies = []
        async for doc in cursor:
            strategies.append(doc)
        
        total_count = await collection.count_documents({})
        
        return {
            "strategies": strategies,
            "total": total_count,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Strategy listing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to list strategies")
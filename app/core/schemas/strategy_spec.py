# backend/app/core/schemas/strategy_spec.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Literal
from datetime import datetime
import uuid

class UserTarget(BaseModel):
    value: float
    period: Literal["day", "week", "month", "year"]
    type: Literal["gross", "net"] = "gross"

class RiskTolerance(BaseModel):
    level: Literal["low", "medium", "high"]
    
class Constraints(BaseModel):
    leverage_max: float = 2.0
    max_drawdown_pct: float = 12.0
    allowed_exchanges: List[str] = ["binance"]
    blacklist_symbols: List[str] = []

class TimeHorizon(BaseModel):
    value: int
    unit: Literal["day", "week", "month", "year"]

class UserRequest(BaseModel):
    raw: str
    target_return: UserTarget
    risk_tolerance: RiskTolerance
    constraints: Constraints
    time_horizon: TimeHorizon
    capital_usd: float

class Feasibility(BaseModel):
    assessment: Literal["viable", "stretch", "low"]
    comment: str
    recommended_target: Optional[UserTarget] = None

class SymbolSpec(BaseModel):
    symbol: str
    exchange: str
    timeframes: List[str]

class DataSpec(BaseModel):
    source: str = "gotobots-data-gateway"
    symbols: List[SymbolSpec]
    features: List[str]
    latency_budget_ms: int = 200
    max_data_gap_tolerance_ms: int = 2000

class DataFingerprint(BaseModel):
    ohlcv_dataset_hash: str
    features_def_version: str
    fees_schema_version: str
    slippage_model: str

class Market(BaseModel):
    symbol: str
    market: str
    
class Universe(BaseModel):
    markets: List[str]
    candidates: List[Market]
    blacklist: List[str] = []

class RegimeDetection(BaseModel):
    features: List[str]
    labels: List[str]
    decision_rules: str

class BotRisk(BaseModel):
    per_trade_risk_pct: float
    daily_risk_cap_pct: float
    max_open_positions: int
    dd_kill_switch_pct: float

class BotExecution(BaseModel):
    exchange: str
    account: Literal["paper", "live"]
    order_type: Literal["limit", "limit_post_only", "market"]
    slippage_ppm: int = 150

class Bot(BaseModel):
    name: str
    type: Literal["scalping", "grid", "dca", "momentum", "pattern_rule", "custom"]
    symbols: List[str]
    timeframe: str
    entry_logic: str
    params: Dict[str, Union[float, int, str]]
    risk: BotRisk
    execution: BotExecution

class BotAllocation(BaseModel):
    bot: str
    weight_pct: float

class LeveragePolicy(BaseModel):
    max: float
    per_bot_caps: Dict[str, float]

class Portfolio(BaseModel):
    capital_usd: float
    allocation: List[BotAllocation]
    leverage_policy: LeveragePolicy

class Compliance(BaseModel):
    max_position_hours: int = 6
    no_overnight_futures: bool = False

# Fixed BacktestPeriod: Removed aliases to avoid confusion
class BacktestPeriod(BaseModel):
    from_date: str = Field(alias="from")
    to_date: str = Field(alias="to")
    mode: Literal["in_sample", "oos"]
    
    class Config:
        allow_population_by_field_name = True  # Allow both field name and alias

class BacktestConfig(BaseModel):
    periods: List[BacktestPeriod]
    fees_schema: str
    latency_ms: int = 80

class ResultsSummary(BaseModel):
    net_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    win_rate: float
    ex_ante_target_hit_probability: float

class Results(BaseModel):
    summary: ResultsSummary
    warnings: List[str] = []

class StrategySpec(BaseModel):
    id: str = Field(default_factory=lambda: f"auto-strat-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}")
    user_request: UserRequest
    feasibility: Feasibility
    data_spec: DataSpec
    data_fingerprint: DataFingerprint
    universe: Universe
    regime_detection: RegimeDetection
    bots: List[Bot]
    portfolio: Portfolio
    compliance: Compliance
    backtest: BacktestConfig
    results: Results
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Additional schemas for API requests
class StrategyRequest(BaseModel):
    query: str
    capital_usd: float = 10000
    risk_tolerance: Literal["low", "medium", "high"] = "medium"
    max_leverage: float = 2.0
    max_drawdown_pct: float = 12.0
    allowed_exchanges: List[str] = ["binance"]
    
class StrategyResponse(BaseModel):
    human_report: str
    strategy_spec: StrategySpec
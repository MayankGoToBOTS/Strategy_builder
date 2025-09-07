# backend/app/core/schemas/backtest_result.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import uuid

class BacktestMetrics(BaseModel):
    net_return_pct: float
    max_drawdown_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    win_rate: float
    exposure_pct: float
    avg_trade_rr: float
    total_trades: int
    profit_factor: float
    calmar_ratio: float

class DistributionStats(BaseModel):
    mean: float
    std: float
    skew: float
    kurtosis: float
    percentiles: Dict[str, float]  # "5": -2.1, "25": -0.5, etc.

class Distributions(BaseModel):
    daily_pnl: DistributionStats
    monthly_pnl: DistributionStats
    drawdown: DistributionStats
    trade_returns: DistributionStats

class StressTestResult(BaseModel):
    net_return_pct: float
    max_drawdown_pct: float
    sharpe: float

class StressTests(BaseModel):
    fees_x150: StressTestResult
    slippage_x150: StressTestResult
    vol_shift_plus_30pct: StressTestResult
    vol_shift_minus_30pct: StressTestResult
    regime_change: StressTestResult

class Trade(BaseModel):
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    fee: float
    pnl: Optional[float] = None
    trade_id: str

class BacktestResult(BaseModel):
    run_id: str = Field(default_factory=lambda: f"bt-{str(uuid.uuid4())}")
    strategy_id: str
    data_fingerprint: Dict[str, str]  # From DataFingerprint schema
    period_start: datetime
    period_end: datetime
    initial_capital: float
    final_capital: float
    metrics: BacktestMetrics
    distributions: Distributions
    stress: StressTests
    trades: List[Trade]
    equity_curve: List[Dict[str, float]]  # [{"timestamp": ts, "equity": val, "drawdown": dd}]
    created_at: datetime = Field(default_factory=datetime.utcnow)


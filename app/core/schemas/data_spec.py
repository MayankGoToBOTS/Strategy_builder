# backend/app/core/schemas/data_spec.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class OHLCVBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    exchange: str
    timeframe: str

class FeatureSet(BaseModel):
    timestamp: datetime
    symbol: str
    timeframe: str
    features: Dict[str, float]  # {"atr_14": 123.45, "rsi_14": 67.8, ...}

class ExchangeFilter(BaseModel):
    exchange: str
    symbol: str
    tick_size: float
    step_size: float
    price_precision: int
    qty_precision: int
    min_notional: float
    max_leverage: float
    as_of: datetime

class RegimeSnapshot(BaseModel):
    symbol: str
    timeframe: str
    timestamp: datetime
    regime: str  # "trend_up", "range", "high_vol", "low_liquidity"
    confidence: float
    features: Dict[str, float]

# Request/Response models for API
class DataRequest(BaseModel):
    exchange: str
    symbol: str
    timeframe: str
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    limit: Optional[int] = 1000

class FeatureRequest(BaseModel):
    symbol: str
    timeframe: str
    features: List[str]
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    limit: Optional[int] = 1000

class DataResponse(BaseModel):
    data: List[OHLCVBar]
    total_count: int
    has_more: bool

class FeatureResponse(BaseModel):
    data: List[FeatureSet]
    total_count: int
    has_more: bool
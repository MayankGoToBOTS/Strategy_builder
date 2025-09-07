# backend/app/core/config.py
from pydantic_settings import BaseSettings
from pydantic import AnyUrl
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # App Configuration
    APP_NAME: str = "GoToBots Strategy Builder"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # MongoDB Configuration
    MONGODB_URL: AnyUrl | str = "mongodb://127.0.0.1:27017"
    MONGODB_DB_HISTORICAL: str = "historical"
    MONGODB_DB_STRATEGIES: str = "strategies"
    # MONGODB_URL="mongodb://40.172.248.81:9131"
    # MONGODB_DB_HISTORICAL="trading_bot_db" 
    # MONGODB_DB_STRATEGIES="strategies_builder"
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 0
    
    # Binance API Configuration
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_SECRET_KEY: Optional[str] = None
    BINANCE_TESTNET: bool = True
    
    # Risk Management Globals
    MAX_LEVERAGE_GLOBAL: float = 5.0
    MAX_DRAWDOWN_GLOBAL: float = 20.0
    MAX_DAILY_RISK_GLOBAL: float = 5.0
    
    # Backtesting Configuration
    DEFAULT_COMMISSION: float = 0.001  # 0.1%
    DEFAULT_SLIPPAGE_PPM: int = 150   # 150 parts per million
    MAX_BACKTEST_DURATION_DAYS: int = 365
    
    # WebSocket Configuration
    WS_RECONNECT_DELAY: int = 5
    WS_PING_INTERVAL: int = 20
    WS_PING_TIMEOUT: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
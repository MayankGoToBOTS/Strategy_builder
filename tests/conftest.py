# backend/tests/conftest.py
import pytest
import asyncio
import os
from typing import Dict, Any
from datetime import datetime, timedelta
import numpy as np

# Set test environment
os.environ["TESTING"] = "1"
os.environ["MONGODB_DB_HISTORICAL"] = "test_gotobots_historical"
os.environ["MONGODB_DB_STRATEGIES"] = "test_gotobots_strategies"
os.environ["REDIS_DB"] = "1"

from app.main import app
from app.deps.mongo_client import MongoManager
from app.deps.redis_client import RedisManager
from app.core.schemas.strategy_spec import StrategyRequest, UserTarget, RiskTolerance, Constraints, TimeHorizon
from app.core.schemas.data_spec import OHLCVBar, FeatureSet
from httpx import AsyncClient

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def setup_test_db():
    """Set up test databases"""
    await MongoManager.initialize()
    await RedisManager.initialize()
    
    # Clean test databases
    historical_db = MongoManager.get_historical_db()
    strategies_db = MongoManager.get_strategies_db()
    redis = RedisManager.get_redis()
    
    # Drop test collections
    await historical_db.drop_collection("ohlcv_1m")
    await historical_db.drop_collection("features_1m")
    await strategies_db.drop_collection("strategies")
    await strategies_db.drop_collection("backtests")
    await redis.flushdb()
    
    yield
    
    # Cleanup
    await MongoManager.close()
    await RedisManager.close()

@pytest.fixture
async def client(setup_test_db):
    """Create test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def sample_strategy_request():
    """Sample strategy request for testing"""
    return StrategyRequest(
        query="Create a scalping strategy that makes 10% monthly returns",
        capital_usd=10000,
        risk_tolerance="medium",
        max_leverage=2.0,
        max_drawdown_pct=12.0,
        allowed_exchanges=["binance"]
    )

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    data = []
    base_price = 50000.0
    base_time = datetime.utcnow() - timedelta(days=30)
    
    for i in range(1000):  # 1000 bars
        timestamp = base_time + timedelta(minutes=i)
        
        # Generate realistic price movement
        price_change = np.random.normal(0, 0.02)  # 2% volatility
        price = base_price * (1 + price_change)
        
        # Generate OHLC from close price
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = price * (1 + np.random.normal(0, 0.005))
        volume = np.random.uniform(1000, 10000)
        
        bar = OHLCVBar(
            timestamp=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=price,
            volume=volume,
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1m"
        )
        
        data.append(bar)
        base_price = price  # Price drift
    
    return data

@pytest.fixture
def sample_features_data():
    """Generate sample features data for testing"""
    data = []
    base_time = datetime.utcnow() - timedelta(days=30)
    
    for i in range(1000):
        timestamp = base_time + timedelta(minutes=i)
        
        features = FeatureSet(
            timestamp=timestamp,
            symbol="BTCUSDT",
            timeframe="1m",
            features={
                "atr_14": np.random.uniform(500, 2000),
                "rsi_14": np.random.uniform(20, 80),
                "adx_14": np.random.uniform(10, 50),
                "plus_di_14": np.random.uniform(10, 40),
                "minus_di_14": np.random.uniform(10, 40),
                "vwap": np.random.uniform(49000, 51000),
                "realized_vol_30": np.random.uniform(0.3, 1.2),
                "spread_bps": np.random.uniform(1, 10)
            }
        )
        
        data.append(features)
    
    return data

@pytest.fixture
async def populate_test_data(sample_ohlcv_data, sample_features_data):
    """Populate test database with sample data"""
    await MongoManager.initialize()
    await RedisManager.initialize()
    
    historical_db = MongoManager.get_historical_db()
    
    # Insert OHLCV data
    ohlcv_docs = [bar.dict() for bar in sample_ohlcv_data]
    await historical_db.ohlcv_1m.insert_many(ohlcv_docs)
    
    # Insert features data
    features_docs = [fs.dict() for fs in sample_features_data]
    await historical_db.features_1m.insert_many(features_docs)
    
    yield
    
    # Cleanup
    await historical_db.drop_collection("ohlcv_1m")
    await historical_db.drop_collection("features_1m")
    await MongoManager.close()
    await RedisManager.close()












# # backend/tests/conftest.py
# import pytest
# import asyncio
# import os
# from typing import Dict, Any
# from datetime import datetime, timedelta
# import numpy as np

# # Set test environment
# os.environ["TESTING"] = "1"
# os.environ["MONGODB_DB_HISTORICAL"] = "test_gotobots_historical"
# os.environ["MONGODB_DB_STRATEGIES"] = "test_gotobots_strategies"
# os.environ["REDIS_DB"] = "1"

# from app.main import app
# from app.deps.mongo_client import MongoManager
# from app.deps.redis_client import RedisManager
# from app.core.schemas.strategy_spec import StrategyRequest, UserTarget, RiskTolerance, Constraints, TimeHorizon
# from app.core.schemas.data_spec import OHLCVBar, FeatureSet
# from httpx import AsyncClient

# @pytest.fixture(scope="session")
# def event_loop():
#     """Create an instance of the default event loop for the test session."""
#     loop = asyncio.get_event_loop_policy().new_event_loop()
#     yield loop
#     loop.close()

# @pytest.fixture(scope="session")
# async def setup_test_db():
#     """Set up test databases"""
#     await MongoManager.initialize()
#     await RedisManager.initialize()
    
#     # Clean test databases
#     historical_db = MongoManager.get_historical_db()
#     strategies_db = MongoManager.get_strategies_db()
#     redis = RedisManager.get_redis()
    
#     # Drop test collections
#     await historical_db.drop_collection("ohlcv_1m")
#     await historical_db.drop_collection("features_1m")
#     await strategies_db.drop_collection("strategies")
#     await strategies_db.drop_collection("backtests")
#     await redis.flushdb()
    
#     yield
    
#     # Cleanup
#     await MongoManager.close()
#     await RedisManager.close()

# @pytest.fixture
# async def client(setup_test_db):
#     """Create test client"""
#     async with AsyncClient(app=app, base_url="http://test") as ac:
#         yield ac

# @pytest.fixture
# def sample_strategy_request():
#     """Sample strategy request for testing"""
#     return StrategyRequest(
#         query="Create a scalping strategy that makes 10% monthly returns",
#         capital_usd=10000,
#         risk_tolerance="medium",
#         max_leverage=2.0,
#         max_drawdown_pct=12.0,
#         allowed_exchanges=["binance"]
#     )

# @pytest.fixture
# def sample_ohlcv_data():
#     """Generate sample OHLCV data for testing"""
#     data = []
#     base_price = 50000.0
#     base_time = datetime.utcnow() - timedelta(days=30)
    
#     for i in range(1000):  # 1000 bars
#         timestamp = base_time + timedelta(minutes=i)
        
#         # Generate realistic price movement
#         price_change = np.random.normal(0, 0.02)  # 2% volatility
#         price = base_price * (1 + price_change)
        
#         # Generate OHLC from close price
#         high = price * (1 + abs(np.random.normal(0, 0.01)))
#         low = price * (1 - abs(np.random.normal(0, 0.01)))
#         open_price = price * (1 + np.random.normal(0, 0.005))
#         volume = np.random.uniform(1000, 10000)
        
#         bar = OHLCVBar(
#             timestamp=timestamp,
#             open=open_price,
#             high=high,
#             low=low,
#             close=price,
#             volume=volume,
#             symbol="BTCUSDT",
#             exchange="binance",
#             timeframe="1m"
#         )
        
#         data.append(bar)
#         base_price = price  # Price drift
    
#     return data

# @pytest.fixture
# def sample_features_data():
#     """Generate sample features data for testing"""
#     data = []
#     base_time = datetime.utcnow() - timedelta(days=30)
    
#     for i in range(1000):
#         timestamp = base_time + timedelta(minutes=i)
        
#         features = FeatureSet(
#             timestamp=timestamp,
#             symbol="BTCUSDT",
#             timeframe="1m",
#             features={
#                 "atr_14": np.random.uniform(500, 2000),
#                 "rsi_14": np.random.uniform(20, 80),
#                 "adx_14": np.random.uniform(10, 50),
#                 "plus_di_14": np.random.uniform(10, 40),
#                 "minus_di_14": np.random.uniform(10, 40),
#                 "vwap": np.random.uniform(49000, 51000),
#                 "realized_vol_30": np.random.uniform(0.3, 1.2),
#                 "spread_bps": np.random.uniform(1, 10)
#             }
#         )
        
#         data.append(features)
    
#     return data

# @pytest.fixture
# async def populate_test_data(setup_test_db, sample_ohlcv_data, sample_features_data):
#     """Populate test database with sample data"""
#     historical_db = MongoManager.get_historical_db()
    
#     # Insert OHLCV data
#     ohlcv_docs = [bar.dict() for bar in sample_ohlcv_data]
#     await historical_db.ohlcv_1m.insert_many(ohlcv_docs)
    
#     # Insert features data
#     features_docs = [fs.dict() for fs in sample_features_data]
#     await historical_db.features_1m.insert_many(features_docs)
    
#     yield
    
#     # Cleanup
#     await historical_db.drop_collection("ohlcv_1m")
#     await historical_db.drop_collection("features_1m")


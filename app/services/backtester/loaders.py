# backend/app/services/backtester/loaders.py
from typing import List, Dict, Optional
from datetime import datetime
from app.deps.mongo_client import MongoManager
from app.core.schemas.data_spec import OHLCVBar, FeatureSet
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Load historical data for backtesting"""
    
    def __init__(self):
        self.cache = {}
    
    async def load_ohlcv(self, symbol: str, exchange: str, timeframe: str,
                        start_date: datetime, end_date: datetime) -> List[OHLCVBar]:
        """Load OHLCV data from MongoDB"""
        try:
            historical_db = MongoManager.get_historical_db()
            collection_name = f"ohlcv_{timeframe}"
            collection = historical_db[collection_name]
            
            # Build query
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            # Execute query
            cursor = collection.find(query).sort("timestamp", 1)
            
            bars = []
            async for doc in cursor:
                bar = OHLCVBar(
                    timestamp=doc["timestamp"],
                    open=doc["open"],
                    high=doc["high"],
                    low=doc["low"],
                    close=doc["close"],
                    volume=doc["volume"],
                    symbol=doc["symbol"],
                    exchange=doc["exchange"],
                    timeframe=timeframe
                )
                bars.append(bar)
            
            logger.info(f"Loaded {len(bars)} OHLCV bars for {symbol} ({timeframe})")
            return bars
            
        except Exception as e:
            logger.error(f"Failed to load OHLCV data: {e}")
            return []
    
    async def load_features(self, symbol: str, timeframe: str, features: List[str],
                           start_date: datetime, end_date: datetime) -> List[FeatureSet]:
        """Load feature data from MongoDB"""
        try:
            historical_db = MongoManager.get_historical_db()
            collection_name = f"features_{timeframe}"
            collection = historical_db[collection_name]
            
            # Build query
            query = {
                "symbol": symbol,
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            # Execute query
            cursor = collection.find(query).sort("timestamp", 1)
            
            feature_sets = []
            async for doc in cursor:
                # Extract requested features
                feature_dict = {}
                for feature in features:
                    if feature in doc:
                        feature_dict[feature] = doc[feature]
                
                if feature_dict:  # Only add if we have some features
                    feature_set = FeatureSet(
                        timestamp=doc["timestamp"],
                        symbol=symbol,
                        timeframe=timeframe,
                        features=feature_dict
                    )
                    feature_sets.append(feature_set)
            
            logger.info(f"Loaded {len(feature_sets)} feature sets for {symbol} ({timeframe})")
            return feature_sets
            
        except Exception as e:
            logger.error(f"Failed to load feature data: {e}")
            return []
    
    async def preload_data(self, symbols: List[str], exchange: str, timeframe: str,
                          start_date: datetime, end_date: datetime):
        """Preload data into cache for faster access"""
        try:
            for symbol in symbols:
                cache_key = f"{symbol}_{exchange}_{timeframe}_{start_date}_{end_date}"
                
                if cache_key not in self.cache:
                    ohlcv_data = await self.load_ohlcv(symbol, exchange, timeframe, start_date, end_date)
                    self.cache[cache_key] = ohlcv_data
                    
                    logger.info(f"Preloaded data for {symbol} into cache")
            
        except Exception as e:
            logger.error(f"Data preloading failed: {e}")
    
    def get_cached_data(self, symbol: str, exchange: str, timeframe: str,
                       start_date: datetime, end_date: datetime) -> Optional[List[OHLCVBar]]:
        """Get data from cache if available"""
        cache_key = f"{symbol}_{exchange}_{timeframe}_{start_date}_{end_date}"
        return self.cache.get(cache_key)
    
    def clear_cache(self):
        """Clear data cache"""
        self.cache.clear()
        logger.info("Data cache cleared")
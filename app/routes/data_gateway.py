# backend/app/routes/data_gateway.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from app.deps.mongo_client import get_historical_db, get_strategies_db
from app.deps.redis_client import get_redis, RedisManager
from app.deps.ccxt_filters import get_ccxt_filters, CCXTFiltersManager
from app.core.schemas.data_spec import (
    DataRequest, FeatureRequest, DataResponse, FeatureResponse,
    OHLCVBar, FeatureSet, ExchangeFilter, RegimeSnapshot
)
from motor.motor_asyncio import AsyncIOMotorDatabase
from redis.asyncio import Redis
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/ohlcv", response_model=DataResponse)
async def get_ohlcv_data(
    exchange: str = Query(..., description="Exchange name (e.g., binance)"),
    symbol: str = Query(..., description="Trading symbol (e.g., BTCUSDT)"),
    timeframe: str = Query("1m", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)"),
    from_date: Optional[datetime] = Query(None, description="Start date (ISO format)"),
    to_date: Optional[datetime] = Query(None, description="End date (ISO format)"),
    limit: int = Query(1000, description="Maximum number of records", le=10000),
    historical_db: AsyncIOMotorDatabase = Depends(get_historical_db)
):
    """Get historical OHLCV data from MongoDB"""
    try:
        # Build collection name
        collection_name = f"ohlcv_{timeframe}"
        collection = historical_db[collection_name]
        
        # Build query
        query = {
            "exchange": exchange,
            "symbol": symbol
        }
        
        # Add date filters
        if from_date or to_date:
            date_filter = {}
            if from_date:
                date_filter["$gte"] = from_date
            if to_date:
                date_filter["$lte"] = to_date
            query["timestamp"] = date_filter
        
        # Count total documents
        total_count = await collection.count_documents(query)
        
        # Get data with limit
        cursor = collection.find(query).sort("timestamp", 1).limit(limit)
        
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
        
        has_more = total_count > limit
        
        logger.info(f"Retrieved {len(bars)} OHLCV records for {exchange}:{symbol}:{timeframe}")
        
        return DataResponse(
            data=bars,
            total_count=total_count,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"Error retrieving OHLCV data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve OHLCV data: {str(e)}")

@router.get("/features", response_model=FeatureResponse)
async def get_feature_data(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query("1m", description="Timeframe"),
    features: List[str] = Query([], description="Feature names to retrieve"),
    from_date: Optional[datetime] = Query(None, description="Start date"),
    to_date: Optional[datetime] = Query(None, description="End date"),
    limit: int = Query(1000, description="Maximum number of records", le=10000),
    use_cache: bool = Query(True, description="Use Redis cache for recent data"),
    historical_db: AsyncIOMotorDatabase = Depends(get_historical_db),
    redis: Redis = Depends(get_redis)
):
    """Get feature data from MongoDB or Redis cache"""
    try:
        feature_sets = []
        
        # If no date filters and use_cache, try Redis first for recent data
        if use_cache and not from_date and not to_date:
            try:
                # Get latest features from Redis
                redis_key = f"features:last:{symbol}:{timeframe}"
                cached_features = await RedisManager.get_hash(redis_key)
                
                if cached_features:
                    # Parse cached features
                    timestamp_str = cached_features.get("timestamp")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        
                        feature_dict = {}
                        for feature in features:
                            value_str = cached_features.get(feature)
                            if value_str:
                                try:
                                    feature_dict[feature] = float(value_str)
                                except ValueError:
                                    pass
                        
                        if feature_dict:
                            feature_set = FeatureSet(
                                timestamp=timestamp,
                                symbol=symbol,
                                timeframe=timeframe,
                                features=feature_dict
                            )
                            feature_sets.append(feature_set)
                            
                            logger.info(f"Retrieved latest features from cache for {symbol}:{timeframe}")
                            return FeatureResponse(
                                data=feature_sets,
                                total_count=1,
                                has_more=False
                            )
            
            except Exception as e:
                logger.warning(f"Failed to get features from cache: {e}")
        
        # Get features from MongoDB
        collection_name = f"features_{timeframe}"
        collection = historical_db[collection_name]
        
        # Build query
        query = {"symbol": symbol}
        
        # Add date filters
        if from_date or to_date:
            date_filter = {}
            if from_date:
                date_filter["$gte"] = from_date
            if to_date:
                date_filter["$lte"] = to_date
            query["timestamp"] = date_filter
        
        # Count total documents
        total_count = await collection.count_documents(query)
        
        # Get data with limit
        cursor = collection.find(query).sort("timestamp", 1).limit(limit)
        
        async for doc in cursor:
            # Filter requested features
            feature_dict = {}
            if features:
                for feature in features:
                    if feature in doc:
                        feature_dict[feature] = doc[feature]
            else:
                # Include all features except metadata
                exclude_fields = {"_id", "symbol", "timestamp", "timeframe"}
                feature_dict = {k: v for k, v in doc.items() if k not in exclude_fields}
            
            if feature_dict:
                feature_set = FeatureSet(
                    timestamp=doc["timestamp"],
                    symbol=doc["symbol"],
                    timeframe=timeframe,
                    features=feature_dict
                )
                feature_sets.append(feature_set)
        
        has_more = total_count > limit
        
        logger.info(f"Retrieved {len(feature_sets)} feature records for {symbol}:{timeframe}")
        
        return FeatureResponse(
            data=feature_sets,
            total_count=total_count,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"Error retrieving feature data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feature data: {str(e)}")

@router.get("/regime/snapshot", response_model=RegimeSnapshot)
async def get_regime_snapshot(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query("1m", description="Timeframe"),
    redis: Redis = Depends(get_redis)
):
    """Get current regime snapshot from Redis"""
    try:
        # Get latest features for regime detection
        features_key = f"features:last:{symbol}:{timeframe}"
        cached_features = await RedisManager.get_hash(features_key)
        
        if not cached_features:
            raise HTTPException(status_code=404, detail="No recent feature data found")
        
        # Parse timestamp
        timestamp_str = cached_features.get("timestamp")
        if not timestamp_str:
            raise HTTPException(status_code=404, detail="No timestamp in feature data")
        
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        # Parse features
        features = {}
        for key, value in cached_features.items():
            if key != "timestamp" and key != "symbol":
                try:
                    features[key] = float(value)
                except ValueError:
                    pass
        
        # Simple regime detection based on available features
        regime = "range"  # Default
        confidence = 0.5
        
        # Basic regime detection logic
        if "adx_14" in features and "rsi_14" in features:
            adx = features["adx_14"]
            rsi = features["rsi_14"]
            
            if adx > 25:
                if "plus_di_14" in features and "minus_di_14" in features:
                    plus_di = features["plus_di_14"]
                    minus_di = features["minus_di_14"]
                    
                    if plus_di > minus_di:
                        regime = "trend_up"
                        confidence = min(0.9, adx / 50.0)
                    else:
                        regime = "trend_down"
                        confidence = min(0.9, adx / 50.0)
                else:
                    regime = "trend_up" if rsi > 50 else "trend_down"
                    confidence = min(0.7, adx / 40.0)
            
            elif "realized_vol_30" in features and features["realized_vol_30"] > 0.5:
                regime = "high_vol"
                confidence = 0.8
        
        return RegimeSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            regime=regime,
            confidence=confidence,
            features=features
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting regime snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get regime snapshot: {str(e)}")

@router.get("/exchange-filters", response_model=List[ExchangeFilter])
async def get_exchange_filters(
    exchange: str = Query(..., description="Exchange name"),
    symbol: Optional[str] = Query(None, description="Specific symbol (optional)"),
    refresh: bool = Query(False, description="Force refresh from exchange"),
    ccxt_filters: CCXTFiltersManager = Depends(get_ccxt_filters),
    strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
):
    """Get exchange filters for order validation"""
    try:
        if refresh:
            # Force refresh from exchange
            await ccxt_filters.refresh_filters(exchange)
        
        # Try to get cached filters first
        cached_filters = await ccxt_filters.get_cached_filters(exchange, symbol, max_age_hours=24)
        
        if cached_filters:
            logger.info(f"Retrieved {len(cached_filters)} cached filters for {exchange}")
            return cached_filters
        
        # If no cached filters, get fresh from exchange
        fresh_filters = await ccxt_filters.get_exchange_filters(exchange, symbol)
        
        # Cache the fresh filters
        if fresh_filters:
            collection = strategies_db.exchange_filters
            filter_docs = [filter_obj.dict() for filter_obj in fresh_filters]
            await collection.insert_many(filter_docs)
        
        logger.info(f"Retrieved {len(fresh_filters)} fresh filters for {exchange}")
        return fresh_filters
        
    except Exception as e:
        logger.error(f"Error getting exchange filters: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get exchange filters: {str(e)}")

@router.post("/validate-order")
async def validate_order(
    exchange: str,
    symbol: str,
    side: str,
    amount: float,
    price: float,
    ccxt_filters: CCXTFiltersManager = Depends(get_ccxt_filters)
):
    """Validate order parameters against exchange filters"""
    try:
        validation_result = await ccxt_filters.validate_order(
            exchange, symbol, side, amount, price
        )
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating order: {e}")
        raise HTTPException(status_code=500, detail=f"Order validation failed: {str(e)}")

@router.get("/symbols/{exchange}")
async def get_exchange_symbols(
    exchange: str,
    market_type: Optional[str] = Query(None, description="spot, future, etc."),
    ccxt_filters: CCXTFiltersManager = Depends(get_ccxt_filters)
):
    """Get available symbols for an exchange"""
    try:
        if exchange not in ccxt_filters.exchanges:
            raise HTTPException(status_code=400, detail=f"Exchange {exchange} not supported")
        
        exchange_instance = ccxt_filters.exchanges[exchange]
        
        # Load markets if not already loaded
        if not exchange_instance.markets:
            import asyncio
            await asyncio.to_thread(exchange_instance.load_markets)
        
        symbols = []
        for symbol, market in exchange_instance.markets.items():
            # Filter by market type if specified
            if market_type and market.get('type') != market_type:
                continue
            
            symbols.append({
                "symbol": symbol,
                "base": market.get('base'),
                "quote": market.get('quote'),
                "type": market.get('type'),
                "active": market.get('active', True),
                "spot": market.get('spot', False),
                "future": market.get('future', False),
                "margin": market.get('margin', False)
            })
        
        logger.info(f"Retrieved {len(symbols)} symbols for {exchange}")
        return {"symbols": symbols, "count": len(symbols)}
        
    except Exception as e:
        logger.error(f"Error getting exchange symbols: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get symbols: {str(e)}")

@router.get("/stream/{symbol}/{timeframe}")
async def get_live_stream_data(
    symbol: str,
    timeframe: str,
    count: int = Query(10, description="Number of recent records", le=100),
    redis: Redis = Depends(get_redis)
):
    """Get recent data from Redis streams"""
    try:
        stream_key = f"ohlcv.binance.{symbol}.{timeframe}"
        
        # Read recent entries from stream
        entries = await redis.xrevrange(stream_key, count=count)
        
        stream_data = []
        for entry_id, fields in entries:
            stream_data.append({
                "id": entry_id,
                "timestamp": fields.get("timestamp"),
                "open": float(fields.get("open", 0)),
                "high": float(fields.get("high", 0)),
                "low": float(fields.get("low", 0)),
                "close": float(fields.get("close", 0)),
                "volume": float(fields.get("volume", 0))
            })
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": stream_data,
            "count": len(stream_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting stream data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stream data: {str(e)}")











# # backend/app/routes/data_gateway.py
# from fastapi import APIRouter, Depends, HTTPException, Query
# from typing import List, Optional, Dict, Any
# from datetime import datetime, timedelta
# from app.deps.mongo_client import get_historical_db, get_strategies_db
# from app.deps.redis_client import get_redis, RedisManager
# from app.deps.ccxt_filters import get_ccxt_filters, CCXTFiltersManager
# from app.core.schemas.data_spec import (
#     DataRequest, FeatureRequest, DataResponse, FeatureResponse,
#     OHLCVBar, FeatureSet, ExchangeFilter, RegimeSnapshot
# )
# from motor.motor_asyncio import AsyncIOMotorDatabase
# from aioredis import Redis
# import logging

# logger = logging.getLogger(__name__)
# router = APIRouter()

# @router.get("/ohlcv", response_model=DataResponse)
# async def get_ohlcv_data(
#     exchange: str = Query(..., description="Exchange name (e.g., binance)"),
#     symbol: str = Query(..., description="Trading symbol (e.g., BTCUSDT)"),
#     timeframe: str = Query("1m", description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)"),
#     from_date: Optional[datetime] = Query(None, description="Start date (ISO format)"),
#     to_date: Optional[datetime] = Query(None, description="End date (ISO format)"),
#     limit: int = Query(1000, description="Maximum number of records", le=10000),
#     historical_db: AsyncIOMotorDatabase = Depends(get_historical_db)
# ):
#     """Get historical OHLCV data from MongoDB"""
#     try:
#         # Build collection name
#         collection_name = f"ohlcv_{timeframe}"
#         collection = historical_db[collection_name]
        
#         # Build query
#         query = {
#             "exchange": exchange,
#             "symbol": symbol
#         }
        
#         # Add date filters
#         if from_date or to_date:
#             date_filter = {}
#             if from_date:
#                 date_filter["$gte"] = from_date
#             if to_date:
#                 date_filter["$lte"] = to_date
#             query["timestamp"] = date_filter
        
#         # Count total documents
#         total_count = await collection.count_documents(query)
        
#         # Get data with limit
#         cursor = collection.find(query).sort("timestamp", 1).limit(limit)
        
#         bars = []
#         async for doc in cursor:
#             bar = OHLCVBar(
#                 timestamp=doc["timestamp"],
#                 open=doc["open"],
#                 high=doc["high"],
#                 low=doc["low"],
#                 close=doc["close"],
#                 volume=doc["volume"],
#                 symbol=doc["symbol"],
#                 exchange=doc["exchange"],
#                 timeframe=timeframe
#             )
#             bars.append(bar)
        
#         has_more = total_count > limit
        
#         logger.info(f"Retrieved {len(bars)} OHLCV records for {exchange}:{symbol}:{timeframe}")
        
#         return DataResponse(
#             data=bars,
#             total_count=total_count,
#             has_more=has_more
#         )
        
#     except Exception as e:
#         logger.error(f"Error retrieving OHLCV data: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to retrieve OHLCV data: {str(e)}")

# @router.get("/features", response_model=FeatureResponse)
# async def get_feature_data(
#     symbol: str = Query(..., description="Trading symbol"),
#     timeframe: str = Query("1m", description="Timeframe"),
#     features: List[str] = Query([], description="Feature names to retrieve"),
#     from_date: Optional[datetime] = Query(None, description="Start date"),
#     to_date: Optional[datetime] = Query(None, description="End date"),
#     limit: int = Query(1000, description="Maximum number of records", le=10000),
#     use_cache: bool = Query(True, description="Use Redis cache for recent data"),
#     historical_db: AsyncIOMotorDatabase = Depends(get_historical_db),
#     redis: Redis = Depends(get_redis)
# ):
#     """Get feature data from MongoDB or Redis cache"""
#     try:
#         feature_sets = []
        
#         # If no date filters and use_cache, try Redis first for recent data
#         if use_cache and not from_date and not to_date:
#             try:
#                 # Get latest features from Redis
#                 redis_key = f"features:last:{symbol}:{timeframe}"
#                 cached_features = await RedisManager.get_hash(redis_key)
                
#                 if cached_features:
#                     # Parse cached features
#                     timestamp_str = cached_features.get("timestamp")
#                     if timestamp_str:
#                         timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        
#                         feature_dict = {}
#                         for feature in features:
#                             value_str = cached_features.get(feature)
#                             if value_str:
#                                 try:
#                                     feature_dict[feature] = float(value_str)
#                                 except ValueError:
#                                     pass
                        
#                         if feature_dict:
#                             feature_set = FeatureSet(
#                                 timestamp=timestamp,
#                                 symbol=symbol,
#                                 timeframe=timeframe,
#                                 features=feature_dict
#                             )
#                             feature_sets.append(feature_set)
                            
#                             logger.info(f"Retrieved latest features from cache for {symbol}:{timeframe}")
#                             return FeatureResponse(
#                                 data=feature_sets,
#                                 total_count=1,
#                                 has_more=False
#                             )
            
#             except Exception as e:
#                 logger.warning(f"Failed to get features from cache: {e}")
        
#         # Get features from MongoDB
#         collection_name = f"features_{timeframe}"
#         collection = historical_db[collection_name]
        
#         # Build query
#         query = {"symbol": symbol}
        
#         # Add date filters
#         if from_date or to_date:
#             date_filter = {}
#             if from_date:
#                 date_filter["$gte"] = from_date
#             if to_date:
#                 date_filter["$lte"] = to_date
#             query["timestamp"] = date_filter
        
#         # Count total documents
#         total_count = await collection.count_documents(query)
        
#         # Get data with limit
#         cursor = collection.find(query).sort("timestamp", 1).limit(limit)
        
#         async for doc in cursor:
#             # Filter requested features
#             feature_dict = {}
#             if features:
#                 for feature in features:
#                     if feature in doc:
#                         feature_dict[feature] = doc[feature]
#             else:
#                 # Include all features except metadata
#                 exclude_fields = {"_id", "symbol", "timestamp", "timeframe"}
#                 feature_dict = {k: v for k, v in doc.items() if k not in exclude_fields}
            
#             if feature_dict:
#                 feature_set = FeatureSet(
#                     timestamp=doc["timestamp"],
#                     symbol=doc["symbol"],
#                     timeframe=timeframe,
#                     features=feature_dict
#                 )
#                 feature_sets.append(feature_set)
        
#         has_more = total_count > limit
        
#         logger.info(f"Retrieved {len(feature_sets)} feature records for {symbol}:{timeframe}")
        
#         return FeatureResponse(
#             data=feature_sets,
#             total_count=total_count,
#             has_more=has_more
#         )
        
#     except Exception as e:
#         logger.error(f"Error retrieving feature data: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to retrieve feature data: {str(e)}")

# @router.get("/regime/snapshot", response_model=RegimeSnapshot)
# async def get_regime_snapshot(
#     symbol: str = Query(..., description="Trading symbol"),
#     timeframe: str = Query("1m", description="Timeframe"),
#     redis: Redis = Depends(get_redis)
# ):
#     """Get current regime snapshot from Redis"""
#     try:
#         # Get latest features for regime detection
#         features_key = f"features:last:{symbol}:{timeframe}"
#         cached_features = await RedisManager.get_hash(features_key)
        
#         if not cached_features:
#             raise HTTPException(status_code=404, detail="No recent feature data found")
        
#         # Parse timestamp
#         timestamp_str = cached_features.get("timestamp")
#         if not timestamp_str:
#             raise HTTPException(status_code=404, detail="No timestamp in feature data")
        
#         timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
#         # Parse features
#         features = {}
#         for key, value in cached_features.items():
#             if key != "timestamp" and key != "symbol":
#                 try:
#                     features[key] = float(value)
#                 except ValueError:
#                     pass
        
#         # Simple regime detection based on available features
#         regime = "range"  # Default
#         confidence = 0.5
        
#         # Basic regime detection logic
#         if "adx_14" in features and "rsi_14" in features:
#             adx = features["adx_14"]
#             rsi = features["rsi_14"]
            
#             if adx > 25:
#                 if "plus_di_14" in features and "minus_di_14" in features:
#                     plus_di = features["plus_di_14"]
#                     minus_di = features["minus_di_14"]
                    
#                     if plus_di > minus_di:
#                         regime = "trend_up"
#                         confidence = min(0.9, adx / 50.0)
#                     else:
#                         regime = "trend_down"
#                         confidence = min(0.9, adx / 50.0)
#                 else:
#                     regime = "trend_up" if rsi > 50 else "trend_down"
#                     confidence = min(0.7, adx / 40.0)
            
#             elif "realized_vol_30" in features and features["realized_vol_30"] > 0.5:
#                 regime = "high_vol"
#                 confidence = 0.8
        
#         return RegimeSnapshot(
#             symbol=symbol,
#             timeframe=timeframe,
#             timestamp=timestamp,
#             regime=regime,
#             confidence=confidence,
#             features=features
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error getting regime snapshot: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to get regime snapshot: {str(e)}")

# @router.get("/exchange-filters", response_model=List[ExchangeFilter])
# async def get_exchange_filters(
#     exchange: str = Query(..., description="Exchange name"),
#     symbol: Optional[str] = Query(None, description="Specific symbol (optional)"),
#     refresh: bool = Query(False, description="Force refresh from exchange"),
#     ccxt_filters: CCXTFiltersManager = Depends(get_ccxt_filters),
#     strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
# ):
#     """Get exchange filters for order validation"""
#     try:
#         if refresh:
#             # Force refresh from exchange
#             await ccxt_filters.refresh_filters(exchange)
        
#         # Try to get cached filters first
#         cached_filters = await ccxt_filters.get_cached_filters(exchange, symbol, max_age_hours=24)
        
#         if cached_filters:
#             logger.info(f"Retrieved {len(cached_filters)} cached filters for {exchange}")
#             return cached_filters
        
#         # If no cached filters, get fresh from exchange
#         fresh_filters = await ccxt_filters.get_exchange_filters(exchange, symbol)
        
#         # Cache the fresh filters
#         if fresh_filters:
#             collection = strategies_db.exchange_filters
#             filter_docs = [filter_obj.dict() for filter_obj in fresh_filters]
#             await collection.insert_many(filter_docs)
        
#         logger.info(f"Retrieved {len(fresh_filters)} fresh filters for {exchange}")
#         return fresh_filters
        
#     except Exception as e:
#         logger.error(f"Error getting exchange filters: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to get exchange filters: {str(e)}")

# @router.post("/validate-order")
# async def validate_order(
#     exchange: str,
#     symbol: str,
#     side: str,
#     amount: float,
#     price: float,
#     ccxt_filters: CCXTFiltersManager = Depends(get_ccxt_filters)
# ):
#     """Validate order parameters against exchange filters"""
#     try:
#         validation_result = await ccxt_filters.validate_order(
#             exchange, symbol, side, amount, price
#         )
        
#         return validation_result
        
#     except Exception as e:
#         logger.error(f"Error validating order: {e}")
#         raise HTTPException(status_code=500, detail=f"Order validation failed: {str(e)}")

# @router.get("/symbols/{exchange}")
# async def get_exchange_symbols(
#     exchange: str,
#     market_type: Optional[str] = Query(None, description="spot, future, etc."),
#     ccxt_filters: CCXTFiltersManager = Depends(get_ccxt_filters)
# ):
#     """Get available symbols for an exchange"""
#     try:
#         if exchange not in ccxt_filters.exchanges:
#             raise HTTPException(status_code=400, detail=f"Exchange {exchange} not supported")
        
#         exchange_instance = ccxt_filters.exchanges[exchange]
        
#         # Load markets if not already loaded
#         if not exchange_instance.markets:
#             import asyncio
#             await asyncio.to_thread(exchange_instance.load_markets)
        
#         symbols = []
#         for symbol, market in exchange_instance.markets.items():
#             # Filter by market type if specified
#             if market_type and market.get('type') != market_type:
#                 continue
            
#             symbols.append({
#                 "symbol": symbol,
#                 "base": market.get('base'),
#                 "quote": market.get('quote'),
#                 "type": market.get('type'),
#                 "active": market.get('active', True),
#                 "spot": market.get('spot', False),
#                 "future": market.get('future', False),
#                 "margin": market.get('margin', False)
#             })
        
#         logger.info(f"Retrieved {len(symbols)} symbols for {exchange}")
#         return {"symbols": symbols, "count": len(symbols)}
        
#     except Exception as e:
#         logger.error(f"Error getting exchange symbols: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to get symbols: {str(e)}")

# @router.get("/stream/{symbol}/{timeframe}")
# async def get_live_stream_data(
#     symbol: str,
#     timeframe: str,
#     count: int = Query(10, description="Number of recent records", le=100),
#     redis: Redis = Depends(get_redis)
# ):
#     """Get recent data from Redis streams"""
#     try:
#         stream_key = f"ohlcv.binance.{symbol}.{timeframe}"
        
#         # Read recent entries from stream
#         entries = await redis.xrevrange(stream_key, count=count)
        
#         stream_data = []
#         for entry_id, fields in entries:
#             stream_data.append({
#                 "id": entry_id,
#                 "timestamp": fields.get("timestamp"),
#                 "open": float(fields.get("open", 0)),
#                 "high": float(fields.get("high", 0)),
#                 "low": float(fields.get("low", 0)),
#                 "close": float(fields.get("close", 0)),
#                 "volume": float(fields.get("volume", 0))
#             })
        
#         return {
#             "symbol": symbol,
#             "timeframe": timeframe,
#             "data": stream_data,
#             "count": len(stream_data)
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting stream data: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to get stream data: {str(e)}")
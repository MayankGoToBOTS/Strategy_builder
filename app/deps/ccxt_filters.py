# backend/app/deps/ccxt_filters.py
import ccxt
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from app.core.config import settings
from app.core.schemas.data_spec import ExchangeFilter
from app.deps.mongo_client import MongoManager
import logging

logger = logging.getLogger(__name__)

class CCXTFiltersManager:
    """Manages exchange filters and market data using CCXT"""
    
    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """Initialize CCXT exchange instances"""
        try:
            # Binance
            binance_config = {
                'apiKey': settings.BINANCE_API_KEY,
                'secret': settings.BINANCE_SECRET_KEY,
                'sandbox': settings.BINANCE_TESTNET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # or 'future' for derivatives
                }
            }
            
            self.exchanges['binance'] = ccxt.binance(binance_config)
            logger.info("CCXT exchanges initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CCXT exchanges: {e}")
    
    async def get_exchange_filters(self, exchange: str, symbol: Optional[str] = None) -> List[ExchangeFilter]:
        """Get exchange filters for symbol(s)"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not supported")
            
            exchange_instance = self.exchanges[exchange]
            
            # Load markets if not already loaded
            if not exchange_instance.markets:
                await asyncio.to_thread(exchange_instance.load_markets)
            
            filters = []
            
            if symbol:
                # Get filters for specific symbol
                if symbol in exchange_instance.markets:
                    market = exchange_instance.markets[symbol]
                    filters.append(self._market_to_filter(exchange, symbol, market))
            else:
                # Get filters for all symbols
                for symbol, market in exchange_instance.markets.items():
                    filters.append(self._market_to_filter(exchange, symbol, market))
            
            return filters
            
        except Exception as e:
            logger.error(f"Failed to get exchange filters: {e}")
            raise
    
    def _market_to_filter(self, exchange: str, symbol: str, market: dict) -> ExchangeFilter:
        """Convert CCXT market info to ExchangeFilter"""
        precision = market.get('precision', {})
        limits = market.get('limits', {})
        
        return ExchangeFilter(
            exchange=exchange,
            symbol=symbol,
            tick_size=precision.get('price', 1e-8),
            step_size=precision.get('amount', 1e-8),
            price_precision=precision.get('price', 8),
            qty_precision=precision.get('amount', 8),
            min_notional=limits.get('cost', {}).get('min', 0.0),
            max_leverage=market.get('info', {}).get('maxLeverage', 1.0),
            as_of=datetime.utcnow()
        )
    
    async def validate_order(self, exchange: str, symbol: str, side: str, 
                           amount: float, price: float) -> Dict[str, any]:
        """Validate order parameters against exchange filters"""
        try:
            filters = await self.get_exchange_filters(exchange, symbol)
            if not filters:
                raise ValueError(f"No filters found for {exchange}:{symbol}")
            
            filter_obj = filters[0]
            
            # Round price to tick size
            rounded_price = self._round_to_tick_size(price, filter_obj.tick_size)
            
            # Round amount to step size
            rounded_amount = self._round_to_step_size(amount, filter_obj.step_size)
            
            # Check minimum notional
            notional = rounded_price * rounded_amount
            if notional < filter_obj.min_notional:
                raise ValueError(f"Order notional {notional} below minimum {filter_obj.min_notional}")
            
            return {
                "valid": True,
                "original_price": price,
                "rounded_price": rounded_price,
                "original_amount": amount,
                "rounded_amount": rounded_amount,
                "notional": notional,
                "filter": filter_obj
            }
            
        except Exception as e:
            logger.error(f"Order validation failed: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    def _round_to_tick_size(self, price: float, tick_size: float) -> float:
        """Round price to exchange tick size"""
        if tick_size == 0:
            return price
        return round(price / tick_size) * tick_size
    
    def _round_to_step_size(self, amount: float, step_size: float) -> float:
        """Round amount to exchange step size"""
        if step_size == 0:
            return amount
        return round(amount / step_size) * step_size
    
    async def refresh_filters(self, exchange: str) -> int:
        """Refresh exchange filters and store in MongoDB"""
        try:
            filters = await self.get_exchange_filters(exchange)
            
            # Store in MongoDB
            strategies_db = MongoManager.get_strategies_db()
            collection = strategies_db.exchange_filters
            
            # Clear old filters for this exchange
            await collection.delete_many({"exchange": exchange})
            
            # Insert new filters
            if filters:
                filter_docs = [filter_obj.dict() for filter_obj in filters]
                await collection.insert_many(filter_docs)
            
            logger.info(f"Refreshed {len(filters)} filters for {exchange}")
            return len(filters)
            
        except Exception as e:
            logger.error(f"Failed to refresh filters for {exchange}: {e}")
            raise
    
    async def get_cached_filters(self, exchange: str, symbol: Optional[str] = None, 
                               max_age_hours: int = 24) -> List[ExchangeFilter]:
        """Get cached exchange filters from MongoDB"""
        try:
            strategies_db = MongoManager.get_strategies_db()
            collection = strategies_db.exchange_filters
            
            # Build query
            query = {
                "exchange": exchange,
                "as_of": {"$gte": datetime.utcnow() - timedelta(hours=max_age_hours)}
            }
            
            if symbol:
                query["symbol"] = symbol
            
            cursor = collection.find(query).sort("as_of", -1)
            
            filters = []
            async for doc in cursor:
                filters.append(ExchangeFilter(**doc))
            
            return filters
            
        except Exception as e:
            logger.error(f"Failed to get cached filters: {e}")
            return []
    
    async def get_symbol_info(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Get detailed symbol information"""
        try:
            if exchange not in self.exchanges:
                return None
            
            exchange_instance = self.exchanges[exchange]
            
            if not exchange_instance.markets:
                await asyncio.to_thread(exchange_instance.load_markets)
            
            if symbol in exchange_instance.markets:
                return exchange_instance.markets[symbol]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get symbol info: {e}")
            return None

# Global instance
ccxt_filters = CCXTFiltersManager()

# Dependency function for FastAPI
async def get_ccxt_filters() -> CCXTFiltersManager:
    return ccxt_filters
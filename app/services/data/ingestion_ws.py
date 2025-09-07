# backend/app/services/data/ingestion_ws.py
import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
from app.core.config import settings
from app.deps.redis_client import RedisManager
from app.core.schemas.data_spec import OHLCVBar, FeatureSet
import traceback

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time market data"""
    
    def __init__(self):
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # exchange -> [symbols]
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Binance WebSocket URLs
        self.binance_spot_url = "wss://stream.binance.com:9443/ws"
        self.binance_futures_url = "wss://fstream.binance.com/ws"
    
    async def start(self):
        """Start WebSocket connections"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting WebSocket manager...")
        
        # Start default subscriptions
        default_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
        
        # Start Binance spot connection
        task = asyncio.create_task(
            self._run_binance_connection("spot", default_symbols)
        )
        self.tasks.append(task)
        
        logger.info(f"WebSocket manager started with {len(default_symbols)} default symbols")
    
    async def stop(self):
        """Stop all WebSocket connections"""
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close all connections
        for connection in self.connections.values():
            await connection.close()
        
        self.connections.clear()
        self.tasks.clear()
        logger.info("WebSocket manager stopped")
    
    async def _run_binance_connection(self, market: str, symbols: List[str]):
        """Run Binance WebSocket connection with automatic reconnection"""
        url = self.binance_spot_url if market == "spot" else self.binance_futures_url
        
        while self.running:
            try:
                await self._connect_binance(url, market, symbols)
            except Exception as e:
                logger.error(f"Binance {market} connection error: {e}")
                if self.running:
                    logger.info(f"Reconnecting in {settings.WS_RECONNECT_DELAY} seconds...")
                    await asyncio.sleep(settings.WS_RECONNECT_DELAY)
    
    async def _connect_binance(self, url: str, market: str, symbols: List[str]):
        """Connect to Binance WebSocket and handle messages"""
        # Create subscription message for kline (candlestick) data
        streams = []
        for symbol in symbols:
            streams.append(f"{symbol.lower()}@kline_1m")
            streams.append(f"{symbol.lower()}@ticker")
        
        params = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        
        logger.info(f"Connecting to Binance {market} WebSocket with {len(streams)} streams...")
        
        async with websockets.connect(
            url,
            ping_interval=settings.WS_PING_INTERVAL,
            ping_timeout=settings.WS_PING_TIMEOUT
        ) as websocket:
            
            self.connections[f"binance_{market}"] = websocket
            
            # Send subscription message
            await websocket.send(json.dumps(params))
            logger.info(f"Subscribed to Binance {market} streams")
            
            # Listen for messages
            async for message in websocket:
                if not self.running:
                    break
                
                try:
                    await self._handle_binance_message(json.loads(message), market)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
    
    async def _handle_binance_message(self, message: dict, market: str):
        """Handle incoming Binance WebSocket message"""
        if "stream" not in message:
            return
        
        stream = message["stream"]
        data = message["data"]
        
        if "@kline_" in stream:
            await self._process_kline_data(data, market)
        elif "@ticker" in stream:
            await self._process_ticker_data(data, market)
    
    async def _process_kline_data(self, kline_data: dict, market: str):
        """Process kline (candlestick) data"""
        try:
            k = kline_data["k"]
            
            # Only process closed candles
            if not k["x"]:  # x indicates if kline is closed
                return
            
            symbol = k["s"]
            
            # Create OHLCV bar
            bar = OHLCVBar(
                timestamp=datetime.fromtimestamp(k["t"] / 1000),
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                symbol=symbol,
                exchange="binance",
                timeframe="1m"
            )
            
            # Store in Redis stream
            redis = RedisManager.get_redis()
            stream_key = f"ohlcv.binance.{symbol}.1m"
            
            await redis.xadd(stream_key, {
                "timestamp": bar.timestamp.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "symbol": symbol,
                "exchange": "binance",
                "timeframe": "1m"
            })
            
            # Update latest snapshot
            snapshot_key = f"ohlcv:last:{symbol}:1m"
            await RedisManager.set_hash(snapshot_key, {
                "timestamp": bar.timestamp.isoformat(),
                "open": str(bar.open),
                "high": str(bar.high),
                "low": str(bar.low),
                "close": str(bar.close),
                "volume": str(bar.volume)
            })
            
            # Trigger feature calculation
            await self._calculate_features(bar)
            
            logger.debug(f"Processed kline data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")
            logger.error(traceback.format_exc())
    
    async def _process_ticker_data(self, ticker_data: dict, market: str):
        """Process 24hr ticker data"""
        try:
            symbol = ticker_data["s"]
            
            # Store ticker info in Redis
            redis = RedisManager.get_redis()
            ticker_key = f"ticker:last:{symbol}"
            
            await RedisManager.set_hash(ticker_key, {
                "symbol": symbol,
                "price_change": ticker_data["P"],
                "price_change_percent": ticker_data["P"],
                "last_price": ticker_data["c"],
                "bid_price": ticker_data.get("b", "0"),
                "ask_price": ticker_data.get("a", "0"),
                "volume": ticker_data["v"],
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error processing ticker data: {e}")
    
    async def _calculate_features(self, bar: OHLCVBar):
        """Calculate and store technical features"""
        try:
            redis = RedisManager.get_redis()
            
            # Get recent bars for feature calculation
            stream_key = f"ohlcv.binance.{bar.symbol}.1m"
            recent_data = await redis.xrevrange(stream_key, count=50)
            
            if len(recent_data) < 20:  # Need minimum data for indicators
                return
            
            # Convert to price list for calculations
            closes = []
            highs = []
            lows = []
            volumes = []
            
            for entry_id, fields in reversed(recent_data):
                closes.append(float(fields["close"]))
                highs.append(float(fields["high"]))
                lows.append(float(fields["low"]))
                volumes.append(float(fields["volume"]))
            
            # Calculate basic features
            features = await self._compute_technical_indicators(
                closes, highs, lows, volumes
            )
            
            # Store features in Redis stream
            features_stream = f"features.{bar.symbol}.1m"
            feature_data = {
                "timestamp": bar.timestamp.isoformat(),
                "symbol": bar.symbol,
                **{k: str(v) for k, v in features.items()}
            }
            
            await redis.xadd(features_stream, feature_data)
            
            # Update latest features snapshot
            features_key = f"features:last:{bar.symbol}:1m"
            await RedisManager.set_hash(features_key, feature_data)
            
            logger.debug(f"Calculated features for {bar.symbol}")
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
    
    async def _compute_technical_indicators(self, closes: List[float], 
                                          highs: List[float], lows: List[float], 
                                          volumes: List[float]) -> Dict[str, float]:
        """Compute technical indicators"""
        import numpy as np
        
        try:
            features = {}
            
            if len(closes) >= 14:
                # RSI
                features["rsi_14"] = self._calculate_rsi(closes, 14)
                
                # ATR
                features["atr_14"] = self._calculate_atr(highs, lows, closes, 14)
            
            if len(closes) >= 20:
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes, 20, 2)
                features["bb_upper_20"] = bb_upper
                features["bb_middle_20"] = bb_middle
                features["bb_lower_20"] = bb_lower
            
            # Current price metrics
            features["price"] = closes[-1]
            features["volume"] = volumes[-1]
            
            # Volatility
            if len(closes) >= 30:
                returns = np.diff(np.log(closes[-30:]))
                features["realized_vol_30"] = np.std(returns) * np.sqrt(1440)  # Annualized
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing indicators: {e}")
            return {}
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate RSI indicator"""
        import numpy as np
        
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_atr(self, highs: List[float], lows: List[float], 
                      closes: List[float], period: int) -> float:
        """Calculate Average True Range"""
        import numpy as np
        
        if len(highs) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        return float(np.mean(true_ranges[-period:]))
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int, 
                                 std_dev: float) -> tuple:
        """Calculate Bollinger Bands"""
        import numpy as np
        
        if len(prices) < period:
            price = prices[-1]
            return price, price, price
        
        prices_array = np.array(prices[-period:])
        middle = np.mean(prices_array)
        std = np.std(prices_array)
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return float(upper), float(middle), float(lower)
    
    async def subscribe_symbol(self, exchange: str, symbol: str, market: str = "spot"):
        """Subscribe to a new symbol"""
        # This would implement dynamic subscription
        # For now, log the request
        logger.info(f"Subscription request: {exchange} {symbol} {market}")
    
    async def unsubscribe_symbol(self, exchange: str, symbol: str):
        """Unsubscribe from a symbol"""
        logger.info(f"Unsubscription request: {exchange} {symbol}")
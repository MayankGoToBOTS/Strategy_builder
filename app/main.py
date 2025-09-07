# backend/app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from app.core.config import settings
from app.deps.mongo_client import MongoManager
from app.deps.redis_client import RedisManager
from app.routes import strategy_builder, data_gateway, backtests
from app.services.data.ingestion_ws import WebSocketManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting GoToBots Strategy Builder...")
    
    # Initialize MongoDB
    await MongoManager.initialize()
    logger.info("MongoDB connected successfully")
    
    # Initialize Redis
    await RedisManager.initialize()
    logger.info("Redis connected successfully")
    
    # Initialize WebSocket Manager for real-time data
    websocket_manager = WebSocketManager()
    app.state.websocket_manager = websocket_manager
    
    # Start background tasks
    await websocket_manager.start()
    logger.info("WebSocket manager started")
    
    yield
    
    # Cleanup
    logger.info("Shutting down GoToBots Strategy Builder...")
    await websocket_manager.stop()
    await RedisManager.close()
    await MongoManager.close()
    logger.info("Cleanup completed")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Advanced Trading Strategy Builder with Real-time Backtesting",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(strategy_builder.router, prefix="/api/v1/strategy", tags=["Strategy Builder"])
app.include_router(data_gateway.router, prefix="/api/v1/data", tags=["Data Gateway"])
app.include_router(backtests.router, prefix="/api/v1/backtests", tags=["Backtests"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "GoToBots Strategy Builder API",
        "version": settings.APP_VERSION,
        "status": "healthy"
    }

@app.get("/healthz")
async def health_check():
    """Detailed health check"""
    try:
        # Check MongoDB
        mongo_status = await MongoManager.health_check()
        
        # Check Redis
        redis_status = await RedisManager.health_check()
        
        return {
            "status": "healthy",
            "services": {
                "mongodb": mongo_status,
                "redis": redis_status,
                "websocket": "active" if hasattr(app.state, 'websocket_manager') else "inactive"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
# backend/app/deps/mongo_client.py
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ServerSelectionTimeoutError
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class MongoManager:
    client: Optional[AsyncIOMotorClient] = None
    historical_db: Optional[AsyncIOMotorDatabase] = None
    strategies_db: Optional[AsyncIOMotorDatabase] = None

    @classmethod
    async def initialize(cls):
        """Initialize MongoDB connection"""
        try:
            cls.client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=10
            )
            
            # Test the connection
            await cls.client.admin.command('ping')
            
            # Initialize databases
            cls.historical_db = cls.client[settings.MONGODB_DB_HISTORICAL]
            cls.strategies_db = cls.client[settings.MONGODB_DB_STRATEGIES]
            
            # Create indexes for better performance
            await cls._create_indexes()
            
            logger.info("MongoDB connection established successfully")
            
        except ServerSelectionTimeoutError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"MongoDB initialization error: {e}")
            raise

    @classmethod
    async def _create_indexes(cls):
        """Create necessary indexes for collections"""
        try:
            # Historical data indexes
            await cls.historical_db.ohlcv_1m.create_index([
                ("exchange", 1), ("symbol", 1), ("timestamp", 1)
            ], unique=True)
            
            await cls.historical_db.features_1m.create_index([
                ("symbol", 1), ("timestamp", 1)
            ])
            
            # Strategy database indexes
            await cls.strategies_db.strategies.create_index([
                ("created_at", -1)
            ])
            
            await cls.strategies_db.backtests.create_index([
                ("strategy_id", 1), ("created_at", -1)
            ])
            
            await cls.strategies_db.exchange_filters.create_index([
                ("exchange", 1), ("symbol", 1), ("as_of", -1)
            ])
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")

    @classmethod
    async def close(cls):
        """Close MongoDB connection"""
        if cls.client:
            cls.client.close()
            logger.info("MongoDB connection closed")

    @classmethod
    async def health_check(cls) -> str:
        """Check MongoDB health"""
        try:
            await cls.client.admin.command('ping')
            return "healthy"
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return "unhealthy"

    @classmethod
    def get_historical_db(cls) -> AsyncIOMotorDatabase:
        """Get historical database instance"""
        if cls.historical_db is None:
            raise RuntimeError("MongoDB not initialized. Call initialize() first.")
        return cls.historical_db

    @classmethod
    def get_strategies_db(cls) -> AsyncIOMotorDatabase:
        """Get strategies database instance"""
        if cls.strategies_db is None:
            raise RuntimeError("MongoDB not initialized. Call initialize() first.")
        return cls.strategies_db

# Dependency functions for FastAPI
async def get_historical_db() -> AsyncIOMotorDatabase:
    return MongoManager.get_historical_db()

async def get_strategies_db() -> AsyncIOMotorDatabase:
    return MongoManager.get_strategies_db()

























# # backend/app/deps/mongo_client.py
# from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
# from pymongo.errors import ServerSelectionTimeoutError
# from app.core.config import settings
# import logging

# logger = logging.getLogger(__name__)

# class MongoManager:
#     client: AsyncIOMotorClient = None
#     historical_db: AsyncIOMotorDatabase = None
#     strategies_db: AsyncIOMotorDatabase = None

#     @classmethod
#     async def initialize(cls):
#         """Initialize MongoDB connection"""
#         try:
#             cls.client = AsyncIOMotorClient(
#                 settings.MONGODB_URL,
#                 serverSelectionTimeoutMS=5000,
#                 maxPoolSize=10
#             )
            
#             # Test the connection
#             await cls.client.admin.command('ping')
            
#             # Initialize databases
#             cls.historical_db = cls.client[settings.MONGODB_DB_HISTORICAL]
#             cls.strategies_db = cls.client[settings.MONGODB_DB_STRATEGIES]
            
#             # Create indexes for better performance
#             await cls._create_indexes()
            
#             logger.info("MongoDB connection established successfully")
            
#         except ServerSelectionTimeoutError as e:
#             logger.error(f"Failed to connect to MongoDB: {e}")
#             raise
#         except Exception as e:
#             logger.error(f"MongoDB initialization error: {e}")
#             raise

#     @classmethod
#     async def _create_indexes(cls):
#         """Create necessary indexes for collections"""
#         try:
#             # Historical data indexes
#             await cls.historical_db.ohlcv_1m.create_index([
#                 ("exchange", 1), ("symbol", 1), ("timestamp", 1)
#             ], unique=True)
            
#             await cls.historical_db.features_1m.create_index([
#                 ("symbol", 1), ("timestamp", 1)
#             ])
            
#             # Strategy database indexes
#             await cls.strategies_db.strategies.create_index([
#                 ("created_at", -1)
#             ])
            
#             await cls.strategies_db.backtests.create_index([
#                 ("strategy_id", 1), ("created_at", -1)
#             ])
            
#             await cls.strategies_db.exchange_filters.create_index([
#                 ("exchange", 1), ("symbol", 1), ("as_of", -1)
#             ])
            
#             logger.info("MongoDB indexes created successfully")
            
#         except Exception as e:
#             logger.warning(f"Failed to create some indexes: {e}")

#     @classmethod
#     async def close(cls):
#         """Close MongoDB connection"""
#         if cls.client:
#             cls.client.close()
#             logger.info("MongoDB connection closed")

#     @classmethod
#     async def health_check(cls) -> str:
#         """Check MongoDB health"""
#         try:
#             await cls.client.admin.command('ping')
#             return "healthy"
#         except Exception as e:
#             logger.error(f"MongoDB health check failed: {e}")
#             return "unhealthy"

#     @classmethod
#     def get_historical_db(cls) -> AsyncIOMotorDatabase:
#         """Get historical database instance"""
#         if cls.historical_db is None:
#             raise RuntimeError("MongoDB not initialized. Call initialize() first.")
#         return cls.historical_db

#     @classmethod
#     def get_strategies_db(cls) -> AsyncIOMotorDatabase:
#         """Get strategies database instance"""
#         if cls.strategies_db is None:
#             raise RuntimeError("MongoDB not initialized. Call initialize() first.")
#         return cls.strategies_db

# # Dependency functions for FastAPI
# async def get_historical_db() -> AsyncIOMotorDatabase:
#     return MongoManager.get_historical_db()

# async def get_strategies_db() -> AsyncIOMotorDatabase:
#     return MongoManager.get_strategies_db()


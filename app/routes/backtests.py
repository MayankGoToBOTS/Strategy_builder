# backend/app/routes/backtests.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from app.core.schemas.backtest_result import BacktestResult, BacktestMetrics
from app.core.schemas.strategy_spec import StrategySpec
from app.services.backtester.engine import BacktestEngine
from app.deps.mongo_client import get_strategies_db
from app.deps.redis_client import get_redis
from motor.motor_asyncio import AsyncIOMotorDatabase
from redis.asyncio import Redis
import asyncio
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class BacktestManager:
    """Manages backtest execution and results"""
    
    def __init__(self):
        self.running_backtests: Dict[str, asyncio.Task] = {}
        self.backtest_queue: List[str] = []
        self.max_concurrent_backtests = 3
    
    async def run_backtest(self, strategy_spec: StrategySpec, 
                          custom_start_date: Optional[datetime] = None,
                          custom_end_date: Optional[datetime] = None,
                          strategies_db: AsyncIOMotorDatabase = None) -> BacktestResult:
        """Execute a backtest for a strategy"""
        try:
            logger.info(f"Starting backtest for strategy {strategy_spec.id}")
            
            # Initialize backtest engine
            engine = BacktestEngine(initial_capital=strategy_spec.portfolio.capital_usd)
            
            # Determine date range
            if custom_start_date and custom_end_date:
                start_date = custom_start_date
                end_date = custom_end_date
            else:
                # Use dates from strategy config
                backtest_period = strategy_spec.backtest.periods[0]
                start_date = datetime.fromisoformat(backtest_period.from_date)
                end_date = datetime.fromisoformat(backtest_period.to_date)
            
            # Run backtest
            result = await engine.run_backtest(strategy_spec, start_date, end_date)
            
            # Store result in database
            if strategies_db:
                await self._store_backtest_result(result, strategies_db)
            
            logger.info(f"Backtest completed for strategy {strategy_spec.id}")
            return result
            
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            raise
    
    async def _store_backtest_result(self, result: BacktestResult, 
                                   strategies_db: AsyncIOMotorDatabase):
        """Store backtest result in database"""
        try:
            collection = strategies_db.backtests
            
            result_doc = {
                "_id": result.run_id,
                "strategy_id": result.strategy_id,
                "backtest_result": result.dict(),
                "created_at": result.created_at,
                "period_start": result.period_start,
                "period_end": result.period_end,
                "final_return_pct": result.metrics.net_return_pct,
                "max_drawdown_pct": result.metrics.max_drawdown_pct,
                "sharpe_ratio": result.metrics.sharpe,
                "total_trades": result.metrics.total_trades
            }
            
            await collection.insert_one(result_doc)
            logger.info(f"Backtest result {result.run_id} stored successfully")
            
        except Exception as e:
            logger.error(f"Failed to store backtest result: {e}")
    
    async def queue_backtest(self, strategy_id: str, backtest_params: Dict) -> str:
        """Queue a backtest for execution"""
        backtest_id = f"bt-{strategy_id}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        if len(self.running_backtests) < self.max_concurrent_backtests:
            # Start immediately
            task = asyncio.create_task(self._execute_queued_backtest(backtest_id, strategy_id, backtest_params))
            self.running_backtests[backtest_id] = task
        else:
            # Add to queue
            self.backtest_queue.append((backtest_id, strategy_id, backtest_params))
        
        return backtest_id
    
    async def _execute_queued_backtest(self, backtest_id: str, strategy_id: str, params: Dict):
        """Execute a queued backtest"""
        try:
            # This would load strategy and run backtest
            logger.info(f"Executing queued backtest {backtest_id}")
            
            # Simulate backtest execution
            await asyncio.sleep(2)  # Placeholder for actual backtest
            
        except Exception as e:
            logger.error(f"Queued backtest {backtest_id} failed: {e}")
        finally:
            # Remove from running tasks
            if backtest_id in self.running_backtests:
                del self.running_backtests[backtest_id]
            
            # Start next queued backtest if any
            if self.backtest_queue:
                next_backtest = self.backtest_queue.pop(0)
                task = asyncio.create_task(
                    self._execute_queued_backtest(next_backtest[0], next_backtest[1], next_backtest[2])
                )
                self.running_backtests[next_backtest[0]] = task
    
    def get_backtest_status(self, backtest_id: str) -> Dict:
        """Get status of a backtest"""
        if backtest_id in self.running_backtests:
            task = self.running_backtests[backtest_id]
            return {
                "status": "running",
                "done": task.done(),
                "cancelled": task.cancelled()
            }
        
        # Check if in queue
        for queued_id, _, _ in self.backtest_queue:
            if queued_id == backtest_id:
                return {"status": "queued", "position": self.backtest_queue.index((queued_id, _, _)) + 1}
        
        return {"status": "not_found"}

# Global backtest manager
backtest_manager = BacktestManager()

@router.post("/run/{strategy_id}")
async def run_backtest(
    strategy_id: str,
    background_tasks: BackgroundTasks,
    start_date: Optional[datetime] = Query(None, description="Custom start date"),
    end_date: Optional[datetime] = Query(None, description="Custom end date"),
    async_execution: bool = Query(False, description="Run backtest asynchronously"),
    strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
):
    """Run backtest for a specific strategy"""
    try:
        # Load strategy
        collection = strategies_db.strategies
        strategy_doc = await collection.find_one({"_id": strategy_id})
        
        if not strategy_doc:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy_spec = StrategySpec(**strategy_doc["strategy_spec"])
        
        if async_execution:
            # Queue backtest for async execution
            backtest_params = {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
            
            backtest_id = await backtest_manager.queue_backtest(strategy_id, backtest_params)
            
            return {
                "message": "Backtest queued successfully",
                "backtest_id": backtest_id,
                "status": "queued"
            }
        else:
            # Run backtest synchronously
            result = await backtest_manager.run_backtest(
                strategy_spec, start_date, end_date, strategies_db
            )
            
            return {
                "message": "Backtest completed successfully",
                "result": result
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest execution request failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute backtest")

@router.get("/status/{backtest_id}")
async def get_backtest_status(backtest_id: str):
    """Get status of a running backtest"""
    status = backtest_manager.get_backtest_status(backtest_id)
    return status

@router.get("/results/{strategy_id}")
async def get_backtest_results(
    strategy_id: str,
    limit: int = Query(10, description="Number of results to return"),
    strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
):
    """Get backtest results for a strategy"""
    try:
        collection = strategies_db.backtests
        
        cursor = collection.find(
            {"strategy_id": strategy_id}
        ).sort("created_at", -1).limit(limit)
        
        results = []
        async for doc in cursor:
            results.append({
                "run_id": doc["_id"],
                "created_at": doc["created_at"],
                "period_start": doc["period_start"],
                "period_end": doc["period_end"],
                "final_return_pct": doc["final_return_pct"],
                "max_drawdown_pct": doc["max_drawdown_pct"],
                "sharpe_ratio": doc["sharpe_ratio"],
                "total_trades": doc["total_trades"]
            })
        
        return {
            "strategy_id": strategy_id,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve backtest results")

@router.get("/result/{run_id}")
async def get_backtest_result_detail(
    run_id: str,
    strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
):
    """Get detailed backtest result"""
    try:
        collection = strategies_db.backtests
        result_doc = await collection.find_one({"_id": run_id})
        
        if not result_doc:
            raise HTTPException(status_code=404, detail="Backtest result not found")
        
        return result_doc["backtest_result"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest result detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve backtest result")

@router.post("/compare")
async def compare_backtests(
    run_ids: List[str],
    strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
):
    """Compare multiple backtest results"""
    try:
        collection = strategies_db.backtests
        
        comparison_data = []
        
        for run_id in run_ids:
            result_doc = await collection.find_one({"_id": run_id})
            if result_doc:
                backtest_result = BacktestResult(**result_doc["backtest_result"])
                
                comparison_data.append({
                    "run_id": run_id,
                    "strategy_id": result_doc["strategy_id"],
                    "period": f"{result_doc['period_start'].strftime('%Y-%m-%d')} to {result_doc['period_end'].strftime('%Y-%m-%d')}",
                    "metrics": {
                        "return_pct": backtest_result.metrics.net_return_pct,
                        "cagr_pct": backtest_result.metrics.cagr_pct,
                        "max_drawdown_pct": backtest_result.metrics.max_drawdown_pct,
                        "sharpe": backtest_result.metrics.sharpe,
                        "sortino": backtest_result.metrics.sortino,
                        "win_rate": backtest_result.metrics.win_rate,
                        "total_trades": backtest_result.metrics.total_trades,
                        "profit_factor": backtest_result.metrics.profit_factor
                    }
                })
        
        if not comparison_data:
            raise HTTPException(status_code=404, detail="No valid backtest results found")
        
        # Calculate comparison statistics
        returns = [d["metrics"]["return_pct"] for d in comparison_data]
        sharpes = [d["metrics"]["sharpe"] for d in comparison_data]
        drawdowns = [d["metrics"]["max_drawdown_pct"] for d in comparison_data]
        
        summary = {
            "best_return": max(returns),
            "worst_return": min(returns),
            "best_sharpe": max(sharpes),
            "worst_sharpe": min(sharpes),
            "best_drawdown": min(drawdowns),  # Lower is better
            "worst_drawdown": max(drawdowns),
            "avg_return": sum(returns) / len(returns),
            "avg_sharpe": sum(sharpes) / len(sharpes)
        }
        
        return {
            "comparison_data": comparison_data,
            "summary": summary,
            "count": len(comparison_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest comparison failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare backtests")

@router.delete("/result/{run_id}")
async def delete_backtest_result(
    run_id: str,
    strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
):
    """Delete a backtest result"""
    try:
        collection = strategies_db.backtests
        result = await collection.delete_one({"_id": run_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Backtest result not found")
        
        return {"message": "Backtest result deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete backtest result: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete backtest result")

@router.get("/analytics/performance-metrics")
async def get_performance_analytics(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
):
    """Get performance analytics across backtests"""
    try:
        collection = strategies_db.backtests
        
        # Build query
        query = {}
        if strategy_id:
            query["strategy_id"] = strategy_id
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            query["created_at"] = date_filter
        
        # Aggregate performance data
        pipeline = [
            {"$match": query},
            {"$group": {
                "_id": None,
                "total_backtests": {"$sum": 1},
                "avg_return": {"$avg": "$final_return_pct"},
                "avg_drawdown": {"$avg": "$max_drawdown_pct"},
                "avg_sharpe": {"$avg": "$sharpe_ratio"},
                "avg_trades": {"$avg": "$total_trades"},
                "best_return": {"$max": "$final_return_pct"},
                "worst_return": {"$min": "$final_return_pct"},
                "best_sharpe": {"$max": "$sharpe_ratio"},
                "worst_sharpe": {"$min": "$sharpe_ratio"}
            }}
        ]
        
        result = await collection.aggregate(pipeline).to_list(length=1)
        
        if not result:
            return {
                "message": "No backtest data found",
                "analytics": None
            }
        
        analytics = result[0]
        analytics.pop("_id", None)
        
        # Get strategy breakdown if not filtered by strategy
        if not strategy_id:
            strategy_pipeline = [
                {"$match": query},
                {"$group": {
                    "_id": "$strategy_id",
                    "backtest_count": {"$sum": 1},
                    "avg_return": {"$avg": "$final_return_pct"},
                    "avg_sharpe": {"$avg": "$sharpe_ratio"}
                }},
                {"$sort": {"avg_return": -1}}
            ]
            
            strategy_breakdown = await collection.aggregate(strategy_pipeline).to_list(length=None)
            analytics["strategy_breakdown"] = strategy_breakdown
        
        return {
            "analytics": analytics,
            "query_params": {
                "strategy_id": strategy_id,
                "start_date": start_date,
                "end_date": end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Performance analytics failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance analytics")

@router.get("/health")
async def backtest_service_health():
    """Health check for backtest service"""
    return {
        "status": "healthy",
        "running_backtests": len(backtest_manager.running_backtests),
        "queued_backtests": len(backtest_manager.backtest_queue),
        "max_concurrent": backtest_manager.max_concurrent_backtests
    }














# # backend/app/routes/backtests.py
# from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
# from typing import List, Dict, Optional
# from datetime import datetime, timedelta
# from app.core.schemas.backtest_result import BacktestResult, BacktestMetrics
# from app.core.schemas.strategy_spec import StrategySpec
# from app.services.backtester.engine import BacktestEngine
# from app.deps.mongo_client import get_strategies_db
# from app.deps.redis_client import get_redis
# from motor.motor_asyncio import AsyncIOMotorDatabase
# from aioredis import Redis
# import asyncio
# import logging

# logger = logging.getLogger(__name__)
# router = APIRouter()

# class BacktestManager:
#     """Manages backtest execution and results"""
    
#     def __init__(self):
#         self.running_backtests: Dict[str, asyncio.Task] = {}
#         self.backtest_queue: List[str] = []
#         self.max_concurrent_backtests = 3
    
#     async def run_backtest(self, strategy_spec: StrategySpec, 
#                           custom_start_date: Optional[datetime] = None,
#                           custom_end_date: Optional[datetime] = None,
#                           strategies_db: AsyncIOMotorDatabase = None) -> BacktestResult:
#         """Execute a backtest for a strategy"""
#         try:
#             logger.info(f"Starting backtest for strategy {strategy_spec.id}")
            
#             # Initialize backtest engine
#             engine = BacktestEngine(initial_capital=strategy_spec.portfolio.capital_usd)
            
#             # Determine date range
#             if custom_start_date and custom_end_date:
#                 start_date = custom_start_date
#                 end_date = custom_end_date
#             else:
#                 # Use dates from strategy config
#                 backtest_period = strategy_spec.backtest.periods[0]
#                 start_date = datetime.fromisoformat(backtest_period.from_date)
#                 end_date = datetime.fromisoformat(backtest_period.to_date)
            
#             # Run backtest
#             result = await engine.run_backtest(strategy_spec, start_date, end_date)
            
#             # Store result in database
#             if strategies_db:
#                 await self._store_backtest_result(result, strategies_db)
            
#             logger.info(f"Backtest completed for strategy {strategy_spec.id}")
#             return result
            
#         except Exception as e:
#             logger.error(f"Backtest execution failed: {e}")
#             raise
    
#     async def _store_backtest_result(self, result: BacktestResult, 
#                                    strategies_db: AsyncIOMotorDatabase):
#         """Store backtest result in database"""
#         try:
#             collection = strategies_db.backtests
            
#             result_doc = {
#                 "_id": result.run_id,
#                 "strategy_id": result.strategy_id,
#                 "backtest_result": result.dict(),
#                 "created_at": result.created_at,
#                 "period_start": result.period_start,
#                 "period_end": result.period_end,
#                 "final_return_pct": result.metrics.net_return_pct,
#                 "max_drawdown_pct": result.metrics.max_drawdown_pct,
#                 "sharpe_ratio": result.metrics.sharpe,
#                 "total_trades": result.metrics.total_trades
#             }
            
#             await collection.insert_one(result_doc)
#             logger.info(f"Backtest result {result.run_id} stored successfully")
            
#         except Exception as e:
#             logger.error(f"Failed to store backtest result: {e}")
    
#     async def queue_backtest(self, strategy_id: str, backtest_params: Dict) -> str:
#         """Queue a backtest for execution"""
#         backtest_id = f"bt-{strategy_id}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
#         if len(self.running_backtests) < self.max_concurrent_backtests:
#             # Start immediately
#             task = asyncio.create_task(self._execute_queued_backtest(backtest_id, strategy_id, backtest_params))
#             self.running_backtests[backtest_id] = task
#         else:
#             # Add to queue
#             self.backtest_queue.append((backtest_id, strategy_id, backtest_params))
        
#         return backtest_id
    
#     async def _execute_queued_backtest(self, backtest_id: str, strategy_id: str, params: Dict):
#         """Execute a queued backtest"""
#         try:
#             # This would load strategy and run backtest
#             logger.info(f"Executing queued backtest {backtest_id}")
            
#             # Simulate backtest execution
#             await asyncio.sleep(2)  # Placeholder for actual backtest
            
#         except Exception as e:
#             logger.error(f"Queued backtest {backtest_id} failed: {e}")
#         finally:
#             # Remove from running tasks
#             if backtest_id in self.running_backtests:
#                 del self.running_backtests[backtest_id]
            
#             # Start next queued backtest if any
#             if self.backtest_queue:
#                 next_backtest = self.backtest_queue.pop(0)
#                 task = asyncio.create_task(
#                     self._execute_queued_backtest(next_backtest[0], next_backtest[1], next_backtest[2])
#                 )
#                 self.running_backtests[next_backtest[0]] = task
    
#     def get_backtest_status(self, backtest_id: str) -> Dict:
#         """Get status of a backtest"""
#         if backtest_id in self.running_backtests:
#             task = self.running_backtests[backtest_id]
#             return {
#                 "status": "running",
#                 "done": task.done(),
#                 "cancelled": task.cancelled()
#             }
        
#         # Check if in queue
#         for queued_id, _, _ in self.backtest_queue:
#             if queued_id == backtest_id:
#                 return {"status": "queued", "position": self.backtest_queue.index((queued_id, _, _)) + 1}
        
#         return {"status": "not_found"}

# # Global backtest manager
# backtest_manager = BacktestManager()

# @router.post("/run/{strategy_id}")
# async def run_backtest(
#     strategy_id: str,
#     background_tasks: BackgroundTasks,
#     start_date: Optional[datetime] = Query(None, description="Custom start date"),
#     end_date: Optional[datetime] = Query(None, description="Custom end date"),
#     async_execution: bool = Query(False, description="Run backtest asynchronously"),
#     strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
# ):
#     """Run backtest for a specific strategy"""
#     try:
#         # Load strategy
#         collection = strategies_db.strategies
#         strategy_doc = await collection.find_one({"_id": strategy_id})
        
#         if not strategy_doc:
#             raise HTTPException(status_code=404, detail="Strategy not found")
        
#         strategy_spec = StrategySpec(**strategy_doc["strategy_spec"])
        
#         if async_execution:
#             # Queue backtest for async execution
#             backtest_params = {
#                 "start_date": start_date.isoformat() if start_date else None,
#                 "end_date": end_date.isoformat() if end_date else None
#             }
            
#             backtest_id = await backtest_manager.queue_backtest(strategy_id, backtest_params)
            
#             return {
#                 "message": "Backtest queued successfully",
#                 "backtest_id": backtest_id,
#                 "status": "queued"
#             }
#         else:
#             # Run backtest synchronously
#             result = await backtest_manager.run_backtest(
#                 strategy_spec, start_date, end_date, strategies_db
#             )
            
#             return {
#                 "message": "Backtest completed successfully",
#                 "result": result
#             }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Backtest execution request failed: {e}")
#         raise HTTPException(status_code=500, detail="Failed to execute backtest")

# @router.get("/status/{backtest_id}")
# async def get_backtest_status(backtest_id: str):
#     """Get status of a running backtest"""
#     status = backtest_manager.get_backtest_status(backtest_id)
#     return status

# @router.get("/results/{strategy_id}")
# async def get_backtest_results(
#     strategy_id: str,
#     limit: int = Query(10, description="Number of results to return"),
#     strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
# ):
#     """Get backtest results for a strategy"""
#     try:
#         collection = strategies_db.backtests
        
#         cursor = collection.find(
#             {"strategy_id": strategy_id}
#         ).sort("created_at", -1).limit(limit)
        
#         results = []
#         async for doc in cursor:
#             results.append({
#                 "run_id": doc["_id"],
#                 "created_at": doc["created_at"],
#                 "period_start": doc["period_start"],
#                 "period_end": doc["period_end"],
#                 "final_return_pct": doc["final_return_pct"],
#                 "max_drawdown_pct": doc["max_drawdown_pct"],
#                 "sharpe_ratio": doc["sharpe_ratio"],
#                 "total_trades": doc["total_trades"]
#             })
        
#         return {
#             "strategy_id": strategy_id,
#             "results": results,
#             "count": len(results)
#         }
        
#     except Exception as e:
#         logger.error(f"Failed to get backtest results: {e}")
#         raise HTTPException(status_code=500, detail="Failed to retrieve backtest results")

# @router.get("/result/{run_id}")
# async def get_backtest_result_detail(
#     run_id: str,
#     strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
# ):
#     """Get detailed backtest result"""
#     try:
#         collection = strategies_db.backtests
#         result_doc = await collection.find_one({"_id": run_id})
        
#         if not result_doc:
#             raise HTTPException(status_code=404, detail="Backtest result not found")
        
#         return result_doc["backtest_result"]
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Failed to get backtest result detail: {e}")
#         raise HTTPException(status_code=500, detail="Failed to retrieve backtest result")

# @router.post("/compare")
# async def compare_backtests(
#     run_ids: List[str],
#     strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
# ):
#     """Compare multiple backtest results"""
#     try:
#         collection = strategies_db.backtests
        
#         comparison_data = []
        
#         for run_id in run_ids:
#             result_doc = await collection.find_one({"_id": run_id})
#             if result_doc:
#                 backtest_result = BacktestResult(**result_doc["backtest_result"])
                
#                 comparison_data.append({
#                     "run_id": run_id,
#                     "strategy_id": result_doc["strategy_id"],
#                     "period": f"{result_doc['period_start'].strftime('%Y-%m-%d')} to {result_doc['period_end'].strftime('%Y-%m-%d')}",
#                     "metrics": {
#                         "return_pct": backtest_result.metrics.net_return_pct,
#                         "cagr_pct": backtest_result.metrics.cagr_pct,
#                         "max_drawdown_pct": backtest_result.metrics.max_drawdown_pct,
#                         "sharpe": backtest_result.metrics.sharpe,
#                         "sortino": backtest_result.metrics.sortino,
#                         "win_rate": backtest_result.metrics.win_rate,
#                         "total_trades": backtest_result.metrics.total_trades,
#                         "profit_factor": backtest_result.metrics.profit_factor
#                     }
#                 })
        
#         if not comparison_data:
#             raise HTTPException(status_code=404, detail="No valid backtest results found")
        
#         # Calculate comparison statistics
#         returns = [d["metrics"]["return_pct"] for d in comparison_data]
#         sharpes = [d["metrics"]["sharpe"] for d in comparison_data]
#         drawdowns = [d["metrics"]["max_drawdown_pct"] for d in comparison_data]
        
#         summary = {
#             "best_return": max(returns),
#             "worst_return": min(returns),
#             "best_sharpe": max(sharpes),
#             "worst_sharpe": min(sharpes),
#             "best_drawdown": min(drawdowns),  # Lower is better
#             "worst_drawdown": max(drawdowns),
#             "avg_return": sum(returns) / len(returns),
#             "avg_sharpe": sum(sharpes) / len(sharpes)
#         }
        
#         return {
#             "comparison_data": comparison_data,
#             "summary": summary,
#             "count": len(comparison_data)
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Backtest comparison failed: {e}")
#         raise HTTPException(status_code=500, detail="Failed to compare backtests")

# @router.delete("/result/{run_id}")
# async def delete_backtest_result(
#     run_id: str,
#     strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
# ):
#     """Delete a backtest result"""
#     try:
#         collection = strategies_db.backtests
#         result = await collection.delete_one({"_id": run_id})
        
#         if result.deleted_count == 0:
#             raise HTTPException(status_code=404, detail="Backtest result not found")
        
#         return {"message": "Backtest result deleted successfully"}
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Failed to delete backtest result: {e}")
#         raise HTTPException(status_code=500, detail="Failed to delete backtest result")

# @router.get("/analytics/performance-metrics")
# async def get_performance_analytics(
#     strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
#     start_date: Optional[datetime] = Query(None, description="Filter by start date"),
#     end_date: Optional[datetime] = Query(None, description="Filter by end date"),
#     strategies_db: AsyncIOMotorDatabase = Depends(get_strategies_db)
# ):
#     """Get performance analytics across backtests"""
#     try:
#         collection = strategies_db.backtests
        
#         # Build query
#         query = {}
#         if strategy_id:
#             query["strategy_id"] = strategy_id
#         if start_date or end_date:
#             date_filter = {}
#             if start_date:
#                 date_filter["$gte"] = start_date
#             if end_date:
#                 date_filter["$lte"] = end_date
#             query["created_at"] = date_filter
        
#         # Aggregate performance data
#         pipeline = [
#             {"$match": query},
#             {"$group": {
#                 "_id": None,
#                 "total_backtests": {"$sum": 1},
#                 "avg_return": {"$avg": "$final_return_pct"},
#                 "avg_drawdown": {"$avg": "$max_drawdown_pct"},
#                 "avg_sharpe": {"$avg": "$sharpe_ratio"},
#                 "avg_trades": {"$avg": "$total_trades"},
#                 "best_return": {"$max": "$final_return_pct"},
#                 "worst_return": {"$min": "$final_return_pct"},
#                 "best_sharpe": {"$max": "$sharpe_ratio"},
#                 "worst_sharpe": {"$min": "$sharpe_ratio"}
#             }}
#         ]
        
#         result = await collection.aggregate(pipeline).to_list(length=1)
        
#         if not result:
#             return {
#                 "message": "No backtest data found",
#                 "analytics": None
#             }
        
#         analytics = result[0]
#         analytics.pop("_id", None)
        
#         # Get strategy breakdown if not filtered by strategy
#         if not strategy_id:
#             strategy_pipeline = [
#                 {"$match": query},
#                 {"$group": {
#                     "_id": "$strategy_id",
#                     "backtest_count": {"$sum": 1},
#                     "avg_return": {"$avg": "$final_return_pct"},
#                     "avg_sharpe": {"$avg": "$sharpe_ratio"}
#                 }},
#                 {"$sort": {"avg_return": -1}}
#             ]
            
#             strategy_breakdown = await collection.aggregate(strategy_pipeline).to_list(length=None)
#             analytics["strategy_breakdown"] = strategy_breakdown
        
#         return {
#             "analytics": analytics,
#             "query_params": {
#                 "strategy_id": strategy_id,
#                 "start_date": start_date,
#                 "end_date": end_date
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"Performance analytics failed: {e}")
#         raise HTTPException(status_code=500, detail="Failed to get performance analytics")

# @router.get("/health")
# async def backtest_service_health():
#     """Health check for backtest service"""
#     return {
#         "status": "healthy",
#         "running_backtests": len(backtest_manager.running_backtests),
#         "queued_backtests": len(backtest_manager.backtest_queue),
#         "max_concurrent": backtest_manager.max_concurrent_backtests
#     }
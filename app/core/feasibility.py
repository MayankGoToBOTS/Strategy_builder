# backend/app/core/feasibility.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from app.core.schemas.strategy_spec import UserTarget, Feasibility, UserRequest
from app.deps.mongo_client import MongoManager
import logging

logger = logging.getLogger(__name__)

class FeasibilityAnalyzer:
    """Analyze feasibility of user-requested trading targets"""
    
    def __init__(self):
        self.period_multipliers = {
            "day": 365,
            "week": 52,
            "month": 12,
            "year": 1
        }
        
        # Realistic performance bands based on market analysis
        self.performance_bands = {
            "conservative": {"daily": 0.5, "monthly": 3.0, "annual": 20.0},
            "moderate": {"daily": 1.0, "monthly": 8.0, "annual": 50.0},
            "aggressive": {"daily": 2.0, "monthly": 15.0, "annual": 100.0},
            "extreme": {"daily": 5.0, "monthly": 25.0, "annual": 200.0}
        }
    
    async def analyze_request(self, user_request: UserRequest) -> Feasibility:
        """Analyze feasibility of user request"""
        try:
            # Convert target to annualized return
            annual_target = self._convert_to_annual(user_request.target_return)
            
            # Get historical volatility for symbols/markets
            avg_volatility = await self._estimate_market_volatility(
                user_request.constraints.allowed_exchanges,
                user_request.time_horizon
            )
            
            # Calculate implied Sharpe ratio
            implied_sharpe = self._calculate_implied_sharpe(annual_target, avg_volatility)
            
            # Assess feasibility
            assessment, comment, recommended_target = self._assess_feasibility(
                annual_target, implied_sharpe, avg_volatility, user_request
            )
            
            return Feasibility(
                assessment=assessment,
                comment=comment,
                recommended_target=recommended_target
            )
            
        except Exception as e:
            logger.error(f"Feasibility analysis failed: {e}")
            # Return conservative fallback
            return Feasibility(
                assessment="low",
                comment=f"Analysis failed due to technical error: {str(e)}. Using conservative assessment.",
                recommended_target=UserTarget(value=5.0, period="month", type="gross")
            )
    
    def _convert_to_annual(self, target: UserTarget) -> float:
        """Convert target return to annualized percentage"""
        multiplier = self.period_multipliers.get(target.period, 1)
        
        if target.period == "day":
            # Daily compounding: (1 + daily_rate)^365 - 1
            daily_rate = target.value / 100
            annual_rate = (1 + daily_rate) ** 365 - 1
            return annual_rate * 100
        else:
            # Simple multiplication for other periods
            return target.value * multiplier
    
    async def _estimate_market_volatility(self, exchanges: List[str], 
                                        time_horizon) -> float:
        """Estimate average market volatility from historical data"""
        try:
            historical_db = MongoManager.get_historical_db()
            
            # Default symbols to analyze
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
            
            # Calculate lookback period - Fixed: access TimeHorizon attributes properly
            horizon_days = self._convert_horizon_to_days(time_horizon)
            lookback_start = datetime.utcnow() - timedelta(days=min(horizon_days * 2, 365))
            
            volatilities = []
            
            for symbol in symbols:
                try:
                    # Get recent price data
                    collection = historical_db.ohlcv_1m
                    cursor = collection.find({
                        "symbol": symbol,
                        "exchange": {"$in": exchanges},
                        "timestamp": {"$gte": lookback_start}
                    }).sort("timestamp", 1)
                    
                    prices = []
                    async for doc in cursor:
                        prices.append(doc["close"])
                    
                    if len(prices) > 100:  # Need sufficient data
                        # Calculate returns and volatility
                        returns = np.diff(np.log(prices))
                        vol = np.std(returns) * np.sqrt(525600)  # Annualized (minutes per year)
                        volatilities.append(vol)
                        
                except Exception as e:
                    logger.warning(f"Failed to get volatility for {symbol}: {e}")
                    continue
            
            if volatilities:
                avg_vol = np.mean(volatilities)
                logger.info(f"Estimated average market volatility: {avg_vol:.2%}")
                return avg_vol
            else:
                # Fallback to typical crypto volatility
                logger.warning("Using fallback volatility estimate")
                return 0.8  # 80% annual volatility
                
        except Exception as e:
            logger.error(f"Volatility estimation failed: {e}")
            return 0.8  # Conservative fallback
    
    def _convert_horizon_to_days(self, time_horizon) -> int:
        """Convert time horizon to days - Fixed: access TimeHorizon attributes properly"""
        # TimeHorizon is a Pydantic model, not a dict
        value = time_horizon.value
        unit = time_horizon.unit
        
        unit_to_days = {
            "day": 1,
            "week": 7,
            "month": 30,
            "year": 365
        }
        
        return value * unit_to_days.get(unit, 30)
    
    def _calculate_implied_sharpe(self, annual_return: float, volatility: float) -> float:
        """Calculate implied Sharpe ratio"""
        if volatility <= 0:
            return 0.0
        
        # Assume risk-free rate of 5% for crypto (USDT staking rates)
        risk_free_rate = 0.05
        excess_return = (annual_return / 100) - risk_free_rate
        
        return excess_return / volatility
    
    def _assess_feasibility(self, annual_target: float, implied_sharpe: float, 
                          volatility: float, user_request: UserRequest) -> Tuple[str, str, Optional[UserTarget]]:
        """Assess feasibility and provide recommendations"""
        
        # Risk tolerance factor
        risk_factor = {
            "low": 0.7,
            "medium": 1.0,
            "high": 1.3
        }.get(user_request.risk_tolerance.level, 1.0)
        
        # Adjusted thresholds based on risk tolerance
        conservative_annual = self.performance_bands["conservative"]["annual"] * risk_factor
        moderate_annual = self.performance_bands["moderate"]["annual"] * risk_factor
        aggressive_annual = self.performance_bands["aggressive"]["annual"] * risk_factor
        extreme_annual = self.performance_bands["extreme"]["annual"] * risk_factor
        
        # Assessment logic
        if annual_target <= conservative_annual:
            assessment = "viable"
            comment = f"Target return of {annual_target:.1f}% annually is achievable with conservative strategies."
            recommended_target = None
            
        elif annual_target <= moderate_annual:
            assessment = "viable" if implied_sharpe <= 2.0 else "stretch"
            if assessment == "viable":
                comment = f"Target return of {annual_target:.1f}% annually is achievable with moderate risk strategies."
                recommended_target = None
            else:
                comment = f"Target return of {annual_target:.1f}% annually requires high Sharpe ratio ({implied_sharpe:.1f}). Consider more conservative target."
                recommended_target = self._generate_recommended_target(moderate_annual * 0.7, user_request.target_return.period)
                
        elif annual_target <= aggressive_annual:
            if implied_sharpe <= 3.0:
                assessment = "stretch"
                comment = f"Target return of {annual_target:.1f}% annually is aggressive but possible with high-risk strategies and exceptional execution."
                recommended_target = self._generate_recommended_target(moderate_annual, user_request.target_return.period)
            else:
                assessment = "low"
                comment = f"Target return of {annual_target:.1f}% annually requires unrealistic Sharpe ratio ({implied_sharpe:.1f}). Strongly recommend lower target."
                recommended_target = self._generate_recommended_target(conservative_annual, user_request.target_return.period)
                
        else:
            assessment = "low"
            comment = f"Target return of {annual_target:.1f}% annually is unrealistic for retail trading. Expected volatility is {volatility:.1%}. Recommend focusing on risk-adjusted returns."
            recommended_target = self._generate_recommended_target(conservative_annual, user_request.target_return.period)
        
        return assessment, comment, recommended_target
    
    def _generate_recommended_target(self, annual_return: float, original_period: str) -> UserTarget:
        """Generate recommended target based on annual return"""
        period_divisors = {
            "day": 365,
            "week": 52,
            "month": 12,
            "year": 1
        }
        
        divisor = period_divisors.get(original_period, 12)
        
        if original_period == "day":
            # Convert from annual to daily with compounding
            daily_rate = (1 + annual_return / 100) ** (1/365) - 1
            target_value = daily_rate * 100
        else:
            target_value = annual_return / divisor
        
        return UserTarget(
            value=round(target_value, 1),
            period=original_period,
            type="gross"
        )






# # backend/app/core/feasibility.py
# import numpy as np
# from typing import Dict, List, Tuple, Optional
# from datetime import datetime, timedelta
# from app.core.schemas.strategy_spec import UserTarget, Feasibility, UserRequest
# from app.deps.mongo_client import MongoManager
# import logging

# logger = logging.getLogger(__name__)

# class FeasibilityAnalyzer:
#     """Analyze feasibility of user-requested trading targets"""
    
#     def __init__(self):
#         self.period_multipliers = {
#             "day": 365,
#             "week": 52,
#             "month": 12,
#             "year": 1
#         }
        
#         # Realistic performance bands based on market analysis
#         self.performance_bands = {
#             "conservative": {"daily": 0.5, "monthly": 3.0, "annual": 20.0},
#             "moderate": {"daily": 1.0, "monthly": 8.0, "annual": 50.0},
#             "aggressive": {"daily": 2.0, "monthly": 15.0, "annual": 100.0},
#             "extreme": {"daily": 5.0, "monthly": 25.0, "annual": 200.0}
#         }
    
#     async def analyze_request(self, user_request: UserRequest) -> Feasibility:
#         """Analyze feasibility of user request"""
#         try:
#             # Convert target to annualized return
#             annual_target = self._convert_to_annual(user_request.target_return)
            
#             # Get historical volatility for symbols/markets
#             avg_volatility = await self._estimate_market_volatility(
#                 user_request.constraints.allowed_exchanges,
#                 user_request.time_horizon
#             )
            
#             # Calculate implied Sharpe ratio
#             implied_sharpe = self._calculate_implied_sharpe(annual_target, avg_volatility)
            
#             # Assess feasibility
#             assessment, comment, recommended_target = self._assess_feasibility(
#                 annual_target, implied_sharpe, avg_volatility, user_request
#             )
            
#             return Feasibility(
#                 assessment=assessment,
#                 comment=comment,
#                 recommended_target=recommended_target
#             )
            
#         except Exception as e:
#             logger.error(f"Feasibility analysis failed: {e}")
#             # Return conservative fallback
#             return Feasibility(
#                 assessment="low",
#                 comment=f"Analysis failed due to technical error: {str(e)}. Using conservative assessment.",
#                 recommended_target=UserTarget(value=5.0, period="month", type="gross")
#             )
    
#     def _convert_to_annual(self, target: UserTarget) -> float:
#         """Convert target return to annualized percentage"""
#         multiplier = self.period_multipliers.get(target.period, 1)
        
#         if target.period == "day":
#             # Daily compounding: (1 + daily_rate)^365 - 1
#             daily_rate = target.value / 100
#             annual_rate = (1 + daily_rate) ** 365 - 1
#             return annual_rate * 100
#         else:
#             # Simple multiplication for other periods
#             return target.value * multiplier
    
#     async def _estimate_market_volatility(self, exchanges: List[str], 
#                                         time_horizon: Dict) -> float:
#         """Estimate average market volatility from historical data"""
#         try:
#             historical_db = MongoManager.get_historical_db()
            
#             # Default symbols to analyze
#             symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
            
#             # Calculate lookback period
#             horizon_days = self._convert_horizon_to_days(time_horizon)
#             lookback_start = datetime.utcnow() - timedelta(days=min(horizon_days * 2, 365))
            
#             volatilities = []
            
#             for symbol in symbols:
#                 try:
#                     # Get recent price data
#                     collection = historical_db.ohlcv_1m
#                     cursor = collection.find({
#                         "symbol": symbol,
#                         "exchange": {"$in": exchanges},
#                         "timestamp": {"$gte": lookback_start}
#                     }).sort("timestamp", 1)
                    
#                     prices = []
#                     async for doc in cursor:
#                         prices.append(doc["close"])
                    
#                     if len(prices) > 100:  # Need sufficient data
#                         # Calculate returns and volatility
#                         returns = np.diff(np.log(prices))
#                         vol = np.std(returns) * np.sqrt(525600)  # Annualized (minutes per year)
#                         volatilities.append(vol)
                        
#                 except Exception as e:
#                     logger.warning(f"Failed to get volatility for {symbol}: {e}")
#                     continue
            
#             if volatilities:
#                 avg_vol = np.mean(volatilities)
#                 logger.info(f"Estimated average market volatility: {avg_vol:.2%}")
#                 return avg_vol
#             else:
#                 # Fallback to typical crypto volatility
#                 logger.warning("Using fallback volatility estimate")
#                 return 0.8  # 80% annual volatility
                
#         except Exception as e:
#             logger.error(f"Volatility estimation failed: {e}")
#             return 0.8  # Conservative fallback
    
#     def _convert_horizon_to_days(self, time_horizon: Dict) -> int:
#         """Convert time horizon to days"""
#         value = time_horizon.get("value", 1)
#         unit = time_horizon.get("unit", "month")
        
#         unit_to_days = {
#             "day": 1,
#             "week": 7,
#             "month": 30,
#             "year": 365
#         }
        
#         return value * unit_to_days.get(unit, 30)
    
#     def _calculate_implied_sharpe(self, annual_return: float, volatility: float) -> float:
#         """Calculate implied Sharpe ratio"""
#         if volatility <= 0:
#             return 0.0
        
#         # Assume risk-free rate of 5% for crypto (USDT staking rates)
#         risk_free_rate = 0.05
#         excess_return = (annual_return / 100) - risk_free_rate
        
#         return excess_return / volatility
    
#     def _assess_feasibility(self, annual_target: float, implied_sharpe: float, 
#                           volatility: float, user_request: UserRequest) -> Tuple[str, str, Optional[UserTarget]]:
#         """Assess feasibility and provide recommendations"""
        
#         # Risk tolerance factor
#         risk_factor = {
#             "low": 0.7,
#             "medium": 1.0,
#             "high": 1.3
#         }.get(user_request.risk_tolerance.level, 1.0)
        
#         # Adjusted thresholds based on risk tolerance
#         conservative_annual = self.performance_bands["conservative"]["annual"] * risk_factor
#         moderate_annual = self.performance_bands["moderate"]["annual"] * risk_factor
#         aggressive_annual = self.performance_bands["aggressive"]["annual"] * risk_factor
#         extreme_annual = self.performance_bands["extreme"]["annual"] * risk_factor
        
#         # Assessment logic
#         if annual_target <= conservative_annual:
#             assessment = "viable"
#             comment = f"Target return of {annual_target:.1f}% annually is achievable with conservative strategies."
#             recommended_target = None
            
#         elif annual_target <= moderate_annual:
#             assessment = "viable" if implied_sharpe <= 2.0 else "stretch"
#             if assessment == "viable":
#                 comment = f"Target return of {annual_target:.1f}% annually is achievable with moderate risk strategies."
#                 recommended_target = None
#             else:
#                 comment = f"Target return of {annual_target:.1f}% annually requires high Sharpe ratio ({implied_sharpe:.1f}). Consider more conservative target."
#                 recommended_target = self._generate_recommended_target(moderate_annual * 0.7, user_request.target_return.period)
                
#         elif annual_target <= aggressive_annual:
#             if implied_sharpe <= 3.0:
#                 assessment = "stretch"
#                 comment = f"Target return of {annual_target:.1f}% annually is aggressive but possible with high-risk strategies and exceptional execution."
#                 recommended_target = self._generate_recommended_target(moderate_annual, user_request.target_return.period)
#             else:
#                 assessment = "low"
#                 comment = f"Target return of {annual_target:.1f}% annually requires unrealistic Sharpe ratio ({implied_sharpe:.1f}). Strongly recommend lower target."
#                 recommended_target = self._generate_recommended_target(conservative_annual, user_request.target_return.period)
                
#         else:
#             assessment = "low"
#             comment = f"Target return of {annual_target:.1f}% annually is unrealistic for retail trading. Expected volatility is {volatility:.1%}. Recommend focusing on risk-adjusted returns."
#             recommended_target = self._generate_recommended_target(conservative_annual, user_request.target_return.period)
        
#         return assessment, comment, recommended_target
    
#     def _generate_recommended_target(self, annual_return: float, original_period: str) -> UserTarget:
#         """Generate recommended target based on annual return"""
#         period_divisors = {
#             "day": 365,
#             "week": 52,
#             "month": 12,
#             "year": 1
#         }
        
#         divisor = period_divisors.get(original_period, 12)
        
#         if original_period == "day":
#             # Convert from annual to daily with compounding
#             daily_rate = (1 + annual_return / 100) ** (1/365) - 1
#             target_value = daily_rate * 100
#         else:
#             target_value = annual_return / divisor
        
#         return UserTarget(
#             value=round(target_value, 1),
#             period=original_period,
#             type="gross"
#         )


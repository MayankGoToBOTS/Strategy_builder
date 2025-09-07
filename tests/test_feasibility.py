# backend/tests/test_feasibility.py
import pytest
from app.core.feasibility import FeasibilityAnalyzer
from app.core.schemas.strategy_spec import UserRequest, UserTarget, RiskTolerance, Constraints, TimeHorizon

class TestFeasibilityAnalyzer:
    """Test feasibility analysis functionality"""
    
    @pytest.fixture
    def analyzer(self):
        return FeasibilityAnalyzer()
    
    @pytest.fixture
    def conservative_request(self):
        return UserRequest(
            raw="Conservative 5% monthly strategy",
            target_return=UserTarget(value=5.0, period="month", type="gross"),
            risk_tolerance=RiskTolerance(level="low"),
            constraints=Constraints(
                leverage_max=1.5,
                max_drawdown_pct=10.0,
                allowed_exchanges=["binance"],
                blacklist_symbols=[]
            ),
            time_horizon=TimeHorizon(value=6, unit="month"),
            capital_usd=10000
        )
    
    @pytest.fixture
    def aggressive_request(self):
        return UserRequest(
            raw="Aggressive 50% monthly strategy",
            target_return=UserTarget(value=50.0, period="month", type="gross"),
            risk_tolerance=RiskTolerance(level="high"),
            constraints=Constraints(
                leverage_max=5.0,
                max_drawdown_pct=25.0,
                allowed_exchanges=["binance"],
                blacklist_symbols=[]
            ),
            time_horizon=TimeHorizon(value=3, unit="month"),
            capital_usd=25000
        )
    
    def test_convert_to_annual_monthly(self, analyzer):
        """Test conversion of monthly target to annual"""
        monthly_target = UserTarget(value=10.0, period="month", type="gross")
        annual = analyzer._convert_to_annual(monthly_target)
        assert annual == 120.0  # 10% * 12 months
    
    def test_convert_to_annual_daily(self, analyzer):
        """Test conversion of daily target to annual with compounding"""
        daily_target = UserTarget(value=1.0, period="day", type="gross")
        annual = analyzer._convert_to_annual(daily_target)
        # Should be compound: (1.01)^365 - 1 ≈ 3678%
        assert annual > 3000  # Should be very high due to compounding
    
    def test_convert_to_annual_yearly(self, analyzer):
        """Test conversion of yearly target"""
        yearly_target = UserTarget(value=20.0, period="year", type="gross")
        annual = analyzer._convert_to_annual(yearly_target)
        assert annual == 20.0  # No conversion needed
    
    @pytest.mark.asyncio
    async def test_conservative_feasibility(self, analyzer, conservative_request):
        """Test feasibility analysis for conservative request"""
        feasibility = await analyzer.analyze_request(conservative_request)
        
        assert feasibility.assessment in ["viable", "stretch"]  # Should be reasonable
        assert feasibility.comment is not None
        assert len(feasibility.comment) > 0
    
    @pytest.mark.asyncio
    async def test_aggressive_feasibility(self, analyzer, aggressive_request):
        """Test feasibility analysis for aggressive request"""
        feasibility = await analyzer.analyze_request(aggressive_request)
        
        assert feasibility.assessment in ["stretch", "low"]  # Should be challenging
        assert feasibility.recommended_target is not None  # Should suggest alternative
        assert feasibility.recommended_target.value < aggressive_request.target_return.value
    
    def test_implied_sharpe_calculation(self, analyzer):
        """Test implied Sharpe ratio calculation"""
        # 50% annual return with 30% volatility
        sharpe = analyzer._calculate_implied_sharpe(50.0, 0.30)
        
        # (50% - 5% risk-free) / 30% volatility = 1.5
        assert abs(sharpe - 1.5) < 0.1
    
    def test_generate_recommended_target_monthly(self, analyzer):
        """Test recommended target generation for monthly period"""
        recommended = analyzer._generate_recommended_target(60.0, "month")  # 60% annual
        
        assert recommended.period == "month"
        assert recommended.value == 5.0  # 60% / 12 months
        assert recommended.type == "gross"
    
    def test_generate_recommended_target_daily(self, analyzer):
        """Test recommended target generation for daily period"""
        recommended = analyzer._generate_recommended_target(36.5, "day")  # 36.5% annual
        
        assert recommended.period == "day"
        # Should use compound formula: (1 + 0.365)^(1/365) - 1 ≈ 0.1%
        assert 0.05 < recommended.value < 0.15


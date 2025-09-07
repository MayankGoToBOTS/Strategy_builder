# backend/tests/test_risk_management.py
import pytest
from app.core.risk import RiskManager, RiskMetrics, RiskLimits

class TestRiskManager:
    """Test risk management functionality"""
    
    @pytest.fixture
    def risk_manager(self):
        return RiskManager(capital_usd=10000)
    
    def test_position_sizing_basic(self, risk_manager):
        """Test basic position sizing calculation"""
        metrics = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            entry_price=50000,
            stop_loss_price=49000,  # 2% stop
            risk_pct=1.0  # 1% risk
        )
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.position_size_usd > 0
        assert metrics.max_loss_usd == 100  # 1% of $10,000
        assert metrics.risk_percentage <= 1.0
    
    def test_position_sizing_with_atr(self, risk_manager):
        """Test position sizing with ATR-based stop"""
        metrics = risk_manager.calculate_position_size(
            symbol="BTCUSDT",
            entry_price=50000,
            stop_loss_price=0,  # No explicit stop
            risk_pct=2.0,
            atr=1000  # $1000 ATR
        )
        
        assert metrics.position_size_usd > 0
        assert metrics.max_loss_usd == 200  # 2% of $10,000
    
    def test_trade_validation_success(self, risk_manager):
        """Test successful trade validation"""
        is_valid, message = risk_manager.validate_trade(
            symbol="BTCUSDT",
            position_size_usd=500,  # 5% of capital
            side="long",
            leverage=1.5
        )
        
        assert is_valid is True
        assert "approved" in message.lower()
    
    def test_trade_validation_position_too_large(self, risk_manager):
        """Test trade validation with oversized position"""
        is_valid, message = risk_manager.validate_trade(
            symbol="BTCUSDT",
            position_size_usd=1500,  # 15% of capital (too large)
            side="long",
            leverage=1.0
        )
        
        assert is_valid is False
        assert "position size" in message.lower()
    
    def test_trade_validation_leverage_too_high(self, risk_manager):
        """Test trade validation with excessive leverage"""
        is_valid, message = risk_manager.validate_trade(
            symbol="BTCUSDT",
            position_size_usd=500,
            side="long",
            leverage=10.0  # Exceeds max leverage
        )
        
        assert is_valid is False
        assert "leverage" in message.lower()
    
    def test_position_update(self, risk_manager):
        """Test position tracking updates"""
        # Initial position
        risk_manager.update_position(
            symbol="BTCUSDT",
            position_size_usd=1000,
            entry_price=50000,
            side="long",
            leverage=1.0
        )
        
        assert "BTCUSDT" in risk_manager.positions
        position = risk_manager.positions["BTCUSDT"]
        assert position["size_usd"] == 1000
        assert position["side"] == "long"
    
    def test_unrealized_pnl_calculation(self, risk_manager):
        """Test unrealized PnL calculation"""
        # Set up position
        risk_manager.update_position(
            symbol="BTCUSDT",
            position_size_usd=1000,
            entry_price=50000,
            side="long",
            leverage=1.0
        )
        
        # Calculate PnL with price increase
        pnl = risk_manager.calculate_unrealized_pnl("BTCUSDT", 52000)  # +4% price move
        
        assert pnl > 0  # Should be profitable
        assert abs(pnl - 40) < 5  # Should be approximately $40 profit
    
    def test_kill_switch_daily_loss(self, risk_manager):
        """Test kill switch triggers on daily loss"""
        # Simulate large daily loss
        risk_manager.daily_pnl = -600  # 6% daily loss
        
        warnings = risk_manager.check_kill_switches()
        
        assert len(warnings) > 0
        assert any("KILL SWITCH" in warning for warning in warnings)
    
    def test_kill_switch_drawdown(self, risk_manager):
        """Test kill switch triggers on drawdown"""
        # Simulate large drawdown
        risk_manager.peak_capital = 12000
        risk_manager.current_drawdown = 20.0  # 20% drawdown
        
        warnings = risk_manager.check_kill_switches()
        
        assert len(warnings) > 0
        assert any("drawdown" in warning.lower() for warning in warnings)
    
    def test_portfolio_summary(self, risk_manager):
        """Test portfolio summary generation"""
        # Add some positions
        risk_manager.update_position("BTCUSDT", 1000, 50000, "long")
        risk_manager.daily_pnl = 50
        
        summary = risk_manager.get_portfolio_summary()
        
        assert "capital_usd" in summary
        assert "daily_pnl" in summary
        assert "total_exposure_usd" in summary
        assert "positions_count" in summary
        assert summary["capital_usd"] == 10000
        assert summary["daily_pnl"] == 50


# backend/app/core/indicators/__init__.py
"""
Technical Indicators Library for GoToBots Strategy Builder

This library ensures truth parity between offline backtesting and online execution
by providing consistent indicator calculations across all components.
"""

from .atr import calculate_atr, ATRIndicator
from .rsi import calculate_rsi, RSIIndicator
from .adx import calculate_adx, ADXIndicator
from .vwap import calculate_vwap, VWAPIndicator
from .volatility import calculate_realized_volatility, VolatilityIndicator
from .regime import RegimeDetector

__all__ = [
    'calculate_atr', 'ATRIndicator',
    'calculate_rsi', 'RSIIndicator', 
    'calculate_adx', 'ADXIndicator',
    'calculate_vwap', 'VWAPIndicator',
    'calculate_realized_volatility', 'VolatilityIndicator',
    'RegimeDetector'
]

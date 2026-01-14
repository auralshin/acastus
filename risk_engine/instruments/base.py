"""Base instrument interface and types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class MarketData:
    """Market data snapshot at a point in time."""
    
    timestamp: datetime
    spot_prices: Dict[str, float]  # symbol -> price
    perp_prices: Dict[str, float]  # symbol -> mark price
    funding_rates: Dict[str, float]  # symbol -> annualized funding rate
    volatilities: Optional[Dict[str, float]] = None  # symbol -> implied vol
    liquidity: Optional[Dict[str, float]] = None  # symbol -> liquidity multiplier
    
    def get_spot_price(self, symbol: str) -> float:
        """Get spot price for a symbol."""
        if symbol not in self.spot_prices:
            raise ValueError(f"No spot price for {symbol}")
        return self.spot_prices[symbol]
    
    def get_perp_price(self, symbol: str) -> float:
        """Get perp price for a symbol."""
        if symbol not in self.perp_prices:
            raise ValueError(f"No perp price for {symbol}")
        return self.perp_prices[symbol]
    
    def get_funding_rate(self, symbol: str) -> float:
        """Get funding rate for a symbol."""
        return self.funding_rates.get(symbol, 0.0)
    
    def get_basis(self, symbol: str) -> float:
        """Calculate basis (perp - spot) for a symbol."""
        return self.get_perp_price(symbol) - self.get_spot_price(symbol)
    
    def get_basis_pct(self, symbol: str) -> float:
        """Calculate basis as percentage of spot."""
        spot = self.get_spot_price(symbol)
        return (self.get_perp_price(symbol) - spot) / spot if spot != 0 else 0.0

    def get_liquidity_multiplier(self, symbol: str) -> float:
        """Get liquidity multiplier for a symbol."""
        if not self.liquidity:
            return 1.0
        return self.liquidity.get(symbol, 1.0)


@dataclass
class RiskExposure:
    """Risk exposures for an instrument."""
    
    delta: float = 0.0  # Price sensitivity (∂Value/∂Price)
    basis_delta: float = 0.0  # Basis sensitivity
    gamma: float = 0.0  # Convexity (liquidation-driven)
    vega: float = 0.0  # Vol sensitivity
    theta: float = 0.0  # Time decay
    rho: float = 0.0  # Funding/rate sensitivity
    
    def total_directional_exposure(self) -> float:
        """Total exposure to directional price moves."""
        return self.delta


class Instrument(ABC):
    """Base interface for all tradeable instruments."""
    
    def __init__(self, symbol: str):
        """
        Initialize instrument.
        
        Args:
            symbol: Instrument identifier (e.g., 'ETH', 'BTC-PERP')
        """
        self.symbol = symbol
    
    @abstractmethod
    def mark_to_market(self, market: MarketData, quantity: float) -> float:
        """
        Calculate mark-to-market value of a position.
        
        Args:
            market: Current market data
            quantity: Position quantity (positive = long, negative = short)
        
        Returns:
            Current market value
        """
        pass
    
    @abstractmethod
    def calculate_pnl(
        self,
        market_t0: MarketData,
        market_t1: MarketData,
        quantity: float,
        entry_price: float
    ) -> Dict[str, float]:
        """
        Calculate P&L between two time points.
        
        Args:
            market_t0: Market data at start
            market_t1: Market data at end
            quantity: Position quantity
            entry_price: Entry price for the position
        
        Returns:
            Dictionary with P&L components
        """
        pass
    
    @abstractmethod
    def risk_exposures(
        self,
        market: MarketData,
        quantity: float,
        entry_price: float
    ) -> RiskExposure:
        """
        Calculate risk exposures (greeks).
        
        Args:
            market: Current market data
            quantity: Position quantity
            entry_price: Entry price
        
        Returns:
            Risk exposures
        """
        pass
    
    def get_current_price(self, market: MarketData) -> float:
        """Get current market price for this instrument."""
        raise NotImplementedError(f"get_current_price not implemented for {self.__class__.__name__}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(symbol='{self.symbol}')"

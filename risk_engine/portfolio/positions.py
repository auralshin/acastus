from dataclasses import dataclass, field
from typing import List, Dict, Optional
from risk_engine.instruments.base import Instrument, MarketData, RiskExposure


@dataclass
class Position:
    """
    A position in a specific instrument.
    
    Represents ownership of an instrument at a specific venue.
    """
    
    instrument: Instrument
    quantity: float  # Positive = long, negative = short
    entry_price: float
    venue: str  # e.g., 'binance', 'dydx', 'hyperliquid'
    entry_timestamp: Optional[str] = None
    
    def notional_value(self, market: MarketData) -> float:
        """Calculate notional value of position."""
        current_price = self.instrument.get_current_price(market)
        return abs(self.quantity * current_price)
    
    def market_value(self, market: MarketData) -> float:
        """Calculate current market value."""
        return self.instrument.mark_to_market(market, self.quantity)
    
    def unrealized_pnl(self, market: MarketData) -> float:
        """Calculate unrealized P&L vs entry."""
        current_price = self.instrument.get_current_price(market)
        return self.quantity * (current_price - self.entry_price)
    
    def risk_exposures(self, market: MarketData) -> RiskExposure:
        """Get risk exposures for this position."""
        return self.instrument.risk_exposures(market, self.quantity, self.entry_price)
    
    def __repr__(self) -> str:
        direction = "LONG" if self.quantity > 0 else "SHORT"
        return (f"Position({direction} {abs(self.quantity):.4f} {self.instrument.symbol} "
                f"@ {self.entry_price:.2f} on {self.venue})")


@dataclass
class Portfolio:
    """
    A portfolio containing multiple positions across instruments and venues.
    
    Tracks positions, cash, and provides aggregated valuation and risk metrics.
    """
    
    positions: List[Position] = field(default_factory=list)
    cash: float = 0.0
    
    def add_position(self, position: Position) -> None:
        """Add a position to the portfolio."""
        self.positions.append(position)
    
    def remove_position(self, position: Position) -> None:
        """Remove a position from the portfolio."""
        self.positions.remove(position)
    
    def total_market_value(self, market: MarketData) -> float:
        """Calculate total market value of all positions."""
        return sum(pos.market_value(market) for pos in self.positions)
    
    def total_notional_exposure(self, market: MarketData) -> float:
        """Calculate total notional exposure (sum of abs values)."""
        return sum(pos.notional_value(market) for pos in self.positions)
    
    def net_asset_value(self, market: MarketData) -> float:
        """
        Calculate NAV (equity).
        
        NAV = Cash + Total Market Value of Positions
        """
        return self.cash + self.total_market_value(market)
    
    def unrealized_pnl(self, market: MarketData) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl(market) for pos in self.positions)
    
    def net_delta(self, market: MarketData) -> Dict[str, float]:
        """
        Calculate net delta exposure by symbol.
        
        Returns:
            Dictionary mapping symbol -> net delta
        """
        delta_by_symbol: Dict[str, float] = {}
        
        for pos in self.positions:
            exposures = pos.risk_exposures(market)
            symbol = pos.instrument.symbol.replace("-PERP", "").replace("PERP", "")
            
            if symbol not in delta_by_symbol:
                delta_by_symbol[symbol] = 0.0
            delta_by_symbol[symbol] += exposures.delta
        
        return delta_by_symbol
    
    def aggregate_risk_exposures(self, market: MarketData) -> RiskExposure:
        """
        Calculate aggregate risk exposures across all positions.
        
        Returns:
            Aggregated RiskExposure
        """
        total = RiskExposure()
        
        for pos in self.positions:
            exposures = pos.risk_exposures(market)
            total.delta += exposures.delta
            total.basis_delta += exposures.basis_delta
            total.gamma += exposures.gamma
            total.vega += exposures.vega
            total.theta += exposures.theta
            total.rho += exposures.rho
        
        return total
    
    def positions_by_venue(self) -> Dict[str, List[Position]]:
        """Group positions by venue."""
        by_venue: Dict[str, List[Position]] = {}
        
        for pos in self.positions:
            if pos.venue not in by_venue:
                by_venue[pos.venue] = []
            by_venue[pos.venue].append(pos)
        
        return by_venue
    
    def positions_by_instrument_type(self) -> Dict[str, List[Position]]:
        """Group positions by instrument type."""
        by_type: Dict[str, List[Position]] = {}
        
        for pos in self.positions:
            inst_type = pos.instrument.__class__.__name__
            if inst_type not in by_type:
                by_type[inst_type] = []
            by_type[inst_type].append(pos)
        
        return by_type
    
    def is_delta_neutral(
        self,
        market: MarketData,
        threshold_pct: float = 0.05
    ) -> bool:
        """
        Check if portfolio is approximately delta neutral.
        
        Args:
            market: Current market data
            threshold_pct: Max allowed net delta as % of total notional
        
        Returns:
            True if delta neutral within threshold
        """
        net_deltas = self.net_delta(market)
        total_notional = self.total_notional_exposure(market)
        
        if total_notional == 0:
            return True
        
        for symbol, delta in net_deltas.items():
            # Get price to convert delta to notional
            try:
                price = market.get_spot_price(symbol)
            except ValueError:
                continue
            
            delta_notional = abs(delta * price)
            delta_pct = delta_notional / total_notional
            
            if delta_pct > threshold_pct:
                return False
        
        return True
    
    def summary(self, market: MarketData) -> Dict[str, any]:
        """
        Get portfolio summary statistics.
        
        Returns:
            Dictionary with key metrics
        """
        nav = self.net_asset_value(market)
        notional = self.total_notional_exposure(market)
        net_deltas = self.net_delta(market)
        exposures = self.aggregate_risk_exposures(market)
        
        return {
            "nav": nav,
            "cash": self.cash,
            "num_positions": len(self.positions),
            "total_notional": notional,
            "leverage": notional / nav if nav != 0 else 0,
            "unrealized_pnl": self.unrealized_pnl(market),
            "net_deltas": net_deltas,
            "total_delta": exposures.delta,
            "total_basis_delta": exposures.basis_delta,
            "total_rho": exposures.rho,
        }
    
    def __repr__(self) -> str:
        return f"Portfolio(positions={len(self.positions)}, cash={self.cash:.2f})"

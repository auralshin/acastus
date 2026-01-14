"""Portfolio valuation and P&L tracking."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime
from risk_engine.instruments.base import MarketData
from risk_engine.portfolio.positions import Portfolio, Position


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    
    timestamp: datetime
    nav: float
    cash: float
    positions: List[Position]
    market: MarketData
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "nav": self.nav,
            "cash": self.cash,
            "num_positions": len(self.positions),
        }


@dataclass
class PnLAttribution:
    """
    P&L attribution broken down by source.
    
    Attributes:
        spot_pnl: P&L from spot price moves
        mark_pnl: P&L from perp mark price moves  
        funding_pnl: P&L from funding payments
        fees_pnl: P&L from trading fees and slippage
        other_pnl: Other P&L sources
        total_pnl: Total P&L
    """
    
    spot_pnl: float = 0.0
    mark_pnl: float = 0.0
    funding_pnl: float = 0.0
    fees_pnl: float = 0.0
    basis_pnl: float = 0.0  # Basis convergence/divergence
    other_pnl: float = 0.0
    total_pnl: float = 0.0
    
    def add(self, other: "PnLAttribution") -> "PnLAttribution":
        """Add another attribution to this one."""
        return PnLAttribution(
            spot_pnl=self.spot_pnl + other.spot_pnl,
            mark_pnl=self.mark_pnl + other.mark_pnl,
            funding_pnl=self.funding_pnl + other.funding_pnl,
            fees_pnl=self.fees_pnl + other.fees_pnl,
            basis_pnl=self.basis_pnl + other.basis_pnl,
            other_pnl=self.other_pnl + other.other_pnl,
            total_pnl=self.total_pnl + other.total_pnl,
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "spot_pnl": self.spot_pnl,
            "mark_pnl": self.mark_pnl,
            "funding_pnl": self.funding_pnl,
            "fees_pnl": self.fees_pnl,
            "basis_pnl": self.basis_pnl,
            "other_pnl": self.other_pnl,
            "total_pnl": self.total_pnl,
        }


class PortfolioValuation:
    """
    Portfolio valuation and P&L calculation engine.
    
    Tracks portfolio evolution over time and attributes P&L to sources.
    """
    
    def __init__(self, portfolio: Portfolio):
        """
        Initialize valuation engine.
        
        Args:
            portfolio: Portfolio to track
        """
        self.portfolio = portfolio
        self.snapshots: List[PortfolioSnapshot] = []
    
    def take_snapshot(self, market: MarketData) -> PortfolioSnapshot:
        """
        Take a snapshot of current portfolio state.
        
        Args:
            market: Current market data
        
        Returns:
            Portfolio snapshot
        """
        nav = self.portfolio.net_asset_value(market)
        
        snapshot = PortfolioSnapshot(
            timestamp=market.timestamp,
            nav=nav,
            cash=self.portfolio.cash,
            positions=self.portfolio.positions.copy(),
            market=market,
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def calculate_period_pnl(
        self,
        market_t0: MarketData,
        market_t1: MarketData,
        include_fees: float = 0.0
    ) -> PnLAttribution:
        """
        Calculate P&L between two time points with attribution.
        
        Args:
            market_t0: Market data at start
            market_t1: Market data at end
            include_fees: Additional fees/costs to include
        
        Returns:
            P&L attribution
        """
        attribution = PnLAttribution()
        
        # Calculate P&L for each position
        for pos in self.portfolio.positions:
            pos_pnl = pos.instrument.calculate_pnl(
                market_t0, market_t1, pos.quantity, pos.entry_price
            )
            
            # Attribute P&L based on instrument type and components
            if "spot_pnl" in pos_pnl:
                attribution.spot_pnl += pos_pnl["spot_pnl"]
            
            if "mark_pnl" in pos_pnl:
                attribution.mark_pnl += pos_pnl["mark_pnl"]
            
            if "funding_pnl" in pos_pnl:
                attribution.funding_pnl += pos_pnl["funding_pnl"]
            
            attribution.total_pnl += pos_pnl.get("total_pnl", 0.0)
        
        # Add fees
        attribution.fees_pnl = -abs(include_fees)
        attribution.total_pnl += attribution.fees_pnl
        
        return attribution
    
    def calculate_returns(self, market: MarketData) -> List[float]:
        """
        Calculate returns from snapshots.
        
        Args:
            market: Current market (for final snapshot if needed)
        
        Returns:
            List of period returns
        """
        if len(self.snapshots) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.snapshots)):
            prev_nav = self.snapshots[i - 1].nav
            curr_nav = self.snapshots[i].nav
            
            if prev_nav > 0:
                ret = (curr_nav - prev_nav) / prev_nav
                returns.append(ret)
        
        return returns
    
    def equity_curve(self) -> Tuple[List[datetime], List[float]]:
        """
        Get equity curve from snapshots.
        
        Returns:
            Tuple of (timestamps, NAV values)
        """
        timestamps = [snap.timestamp for snap in self.snapshots]
        navs = [snap.nav for snap in self.snapshots]
        return timestamps, navs
    
    def reset_snapshots(self) -> None:
        """Clear snapshot history."""
        self.snapshots = []

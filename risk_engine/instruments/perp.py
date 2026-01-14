"""Perpetual swap instrument."""

from typing import Dict
from risk_engine.instruments.base import Instrument, MarketData, RiskExposure


class PerpetualSwap(Instrument):
    """
    Perpetual swap contract.
    
    Features:
    - Mark-to-market P&L vs entry price
    - Funding payments (positive = longs pay shorts, negative = shorts pay longs)
    - Basis exposure (perp vs spot)
    """
    
    def __init__(self, symbol: str, funding_period_hours: float = 8.0):
        """
        Initialize perpetual swap.
        
        Args:
            symbol: Perp symbol (e.g., 'ETH-PERP', 'BTC-PERP')
            funding_period_hours: Hours between funding payments (typically 8 hours)
        """
        super().__init__(symbol)
        self.funding_period_hours = funding_period_hours
        # Extract base symbol (e.g., 'ETH' from 'ETH-PERP')
        self.base_symbol = symbol.replace("-PERP", "").replace("PERP", "")
    
    def mark_to_market(self, market: MarketData, quantity: float) -> float:
        """
        MTM value for perp = quantity * mark_price.
        
        Note: For perps, notional value doesn't change with price like spot.
        The actual "value" is the unrealized P&L vs entry, but for consistency
        we return the notional.
        
        Args:
            market: Current market data
            quantity: Position size
        
        Returns:
            Current notional value
        """
        mark_price = market.get_perp_price(self.symbol)
        return quantity * mark_price
    
    def calculate_pnl(
        self,
        market_t0: MarketData,
        market_t1: MarketData,
        quantity: float,
        entry_price: float
    ) -> Dict[str, float]:
        """
        Calculate perp P&L including funding.
        
        P&L components:
        1. Mark-to-market: quantity * (price_t1 - price_t0)
        2. Funding: -quantity * funding_rate * notional * time_fraction
           (negative sign because longs pay when funding is positive)
        
        Args:
            market_t0: Market at start
            market_t1: Market at end
            quantity: Position size (+ = long, - = short)
            entry_price: Entry price (used for notional calculation)
        
        Returns:
            Dictionary with 'mark_pnl', 'funding_pnl', 'total_pnl'
        """
        price_t0 = market_t0.get_perp_price(self.symbol)
        price_t1 = market_t1.get_perp_price(self.symbol)
        
        # Mark-to-market P&L
        mark_pnl = quantity * (price_t1 - price_t0)
        
        # Funding P&L
        # Funding rate is typically annualized; need to pro-rate for time period
        funding_rate = market_t1.get_funding_rate(self.symbol)
        time_delta_hours = (market_t1.timestamp - market_t0.timestamp).total_seconds() / 3600
        time_fraction = time_delta_hours / self.funding_period_hours
        
        # Notional = quantity * mark price (use avg price for accuracy)
        avg_price = (price_t0 + price_t1) / 2
        notional = abs(quantity * avg_price)
        
        # Longs pay when funding > 0, shorts pay when funding < 0
        # funding_pnl is negative when you're paying, positive when receiving
        funding_pnl = -quantity * (funding_rate / (365.25 * 24 / self.funding_period_hours)) * notional * time_fraction
        
        return {
            "mark_pnl": mark_pnl,
            "funding_pnl": funding_pnl,
            "total_pnl": mark_pnl + funding_pnl
        }
    
    def risk_exposures(
        self,
        market: MarketData,
        quantity: float,
        entry_price: float
    ) -> RiskExposure:
        """
        Perp exposures.
        
        - Delta: directional price exposure (same as spot)
        - Basis delta: exposure to perp-spot basis
        - Rho: exposure to funding rate changes
        
        Returns:
            RiskExposure
        """
        mark_price = market.get_perp_price(self.symbol)
        funding_rate = market.get_funding_rate(self.symbol)
        
        # Delta: same as spot (1:1 with perp price)
        delta = quantity
        
        # Basis delta: exposure to (perp - spot) spread
        basis_delta = quantity
        
        # Rho: sensitivity to funding rate - key risk for delta-neutral strategies
        notional = abs(quantity * mark_price)
        rho = -quantity * notional * 8760  # Annual funding exposure (key risk!)
        
        return RiskExposure(
            delta=delta,
            basis_delta=basis_delta,  # Basis risk exposure
            gamma=0.0,   # Linear payoff
            vega=0.0,    # No direct vol sensitivity
            theta=0.0,   # No time decay (perpetual)
            rho=rho      # Funding rate risk - critical for delta-neutral
        )
    
    def get_current_price(self, market: MarketData) -> float:
        """Get current perp mark price."""
        return market.get_perp_price(self.symbol)
    
    def calculate_funding_payment(
        self,
        market: MarketData,
        quantity: float,
        hours: float | None = None
    ) -> float:
        """
        Calculate funding payment over a period.
        
        Args:
            market: Current market data
            quantity: Position size
            hours: Time period in hours (default: one funding period)
        
        Returns:
            Funding payment (negative = you pay, positive = you receive)
        """
        if hours is None:
            hours = self.funding_period_hours
        
        funding_rate = market.get_funding_rate(self.symbol)
        mark_price = market.get_perp_price(self.symbol)
        notional = abs(quantity * mark_price)
        
        time_fraction = hours / self.funding_period_hours
        annual_periods = 365.25 * 24 / self.funding_period_hours
        
        return -quantity * (funding_rate / annual_periods) * notional * time_fraction

"""Margin models and requirements."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from risk_engine.portfolio.positions import Portfolio, Position
from risk_engine.instruments.base import MarketData


@dataclass
class MarginRequirements:
    """Margin requirements for a position or portfolio."""
    
    initial_margin: float  # Required to open position
    maintenance_margin: float  # Required to keep position open
    available_margin: float  # Excess margin above requirement
    margin_ratio: float  # Equity / Maintenance Margin
    
    def is_healthy(self) -> bool:
        """Check if margin is above maintenance."""
        return self.margin_ratio > 1.0
    
    def distance_to_liquidation_pct(self) -> float:
        """Calculate % distance to liquidation."""
        if self.margin_ratio <= 0:
            return 0.0
        return (self.margin_ratio - 1.0) * 100


@dataclass
class VenueMarginConfig:
    """Margin configuration for a specific venue."""
    
    venue_name: str
    initial_margin_rate: float = 0.10  # 10% initial margin
    maintenance_margin_rate: float = 0.05  # 5% maintenance
    liquidation_fee: float = 0.005  # 0.5% liquidation fee
    
    # Position-specific multipliers
    size_multipliers: Optional[Dict[str, float]] = None  # Larger positions need more margin
    
    def calculate_im(self, notional: float) -> float:
        """Calculate initial margin requirement."""
        base_im = notional * self.initial_margin_rate
        # Could add size-based adjustments here
        return base_im
    
    def calculate_mm(self, notional: float) -> float:
        """Calculate maintenance margin requirement."""
        base_mm = notional * self.maintenance_margin_rate
        return base_mm


class MarginModel:
    """
    Margin model for calculating requirements and liquidation risk.
    
    Supports venue-specific margin rules.
    """
    
    def __init__(self, venue_configs: Optional[Dict[str, VenueMarginConfig]] = None):
        """
        Initialize margin model.
        
        Args:
            venue_configs: Dictionary of venue name -> margin config
        """
        if venue_configs is None:
            # Default config
            self.venue_configs = {
                "default": VenueMarginConfig("default")
            }
        else:
            self.venue_configs = venue_configs
    
    def get_venue_config(self, venue: str) -> VenueMarginConfig:
        """Get margin config for a venue."""
        return self.venue_configs.get(venue, self.venue_configs.get("default"))
    
    def calculate_position_margin(
        self,
        position: Position,
        market: MarketData,
        equity: float
    ) -> MarginRequirements:
        """
        Calculate margin requirements for a position.
        
        Args:
            position: Position to calculate margin for
            market: Current market data
            equity: Available equity
        
        Returns:
            MarginRequirements
        """
        config = self.get_venue_config(position.venue)
        notional = position.notional_value(market)
        
        im = config.calculate_im(notional)
        mm = config.calculate_mm(notional)
        
        margin_ratio = equity / mm if mm > 0 else float('inf')
        available = equity - mm
        
        return MarginRequirements(
            initial_margin=im,
            maintenance_margin=mm,
            available_margin=available,
            margin_ratio=margin_ratio
        )
    
    def calculate_portfolio_margin(
        self,
        portfolio: Portfolio,
        market: MarketData,
        by_venue: bool = True
    ) -> Dict[str, MarginRequirements]:
        """
        Calculate margin requirements for entire portfolio.
        
        Args:
            portfolio: Portfolio
            market: Market data
            by_venue: If True, calculate per venue; else aggregate
        
        Returns:
            Dictionary of venue -> MarginRequirements
        """
        if by_venue:
            return self._calculate_by_venue(portfolio, market)
        else:
            return {"total": self._calculate_aggregate(portfolio, market)}
    
    def _calculate_by_venue(
        self,
        portfolio: Portfolio,
        market: MarketData
    ) -> Dict[str, MarginRequirements]:
        """Calculate margin requirements per venue."""
        by_venue = portfolio.positions_by_venue()
        equity = portfolio.net_asset_value(market)
        
        results = {}
        
        for venue, positions in by_venue.items():
            config = self.get_venue_config(venue)
            
            total_notional = sum(pos.notional_value(market) for pos in positions)
            im = config.calculate_im(total_notional)
            mm = config.calculate_mm(total_notional)
            
            # Allocate equity proportionally
            venue_equity = equity * (total_notional / portfolio.total_notional_exposure(market))
            
            margin_ratio = venue_equity / mm if mm > 0 else float('inf')
            available = venue_equity - mm
            
            results[venue] = MarginRequirements(
                initial_margin=im,
                maintenance_margin=mm,
                available_margin=available,
                margin_ratio=margin_ratio
            )
        
        return results
    
    def _calculate_aggregate(
        self,
        portfolio: Portfolio,
        market: MarketData
    ) -> MarginRequirements:
        """Calculate aggregate margin requirements."""
        equity = portfolio.net_asset_value(market)
        total_notional = portfolio.total_notional_exposure(market)
        
        # Use weighted average config
        config = self.venue_configs.get("default", VenueMarginConfig("default"))
        
        im = config.calculate_im(total_notional)
        mm = config.calculate_mm(total_notional)
        
        margin_ratio = equity / mm if mm > 0 else float('inf')
        available = equity - mm
        
        return MarginRequirements(
            initial_margin=im,
            maintenance_margin=mm,
            available_margin=available,
            margin_ratio=margin_ratio
        )
    
    def check_liquidation_risk(
        self,
        portfolio: Portfolio,
        market: MarketData,
        by_venue: bool = True
    ) -> Dict[str, bool]:
        """
        Check if any venues are at risk of liquidation.
        
        Args:
            portfolio: Portfolio
            market: Market data
            by_venue: Check per venue vs aggregate
        
        Returns:
            Dictionary of venue -> is_at_risk
        """
        margin_reqs = self.calculate_portfolio_margin(portfolio, market, by_venue)
        
        return {
            venue: not reqs.is_healthy()
            for venue, reqs in margin_reqs.items()
        }
    
    def liquidation_price(
        self,
        position: Position,
        market: MarketData,
        equity: float
    ) -> Optional[float]:
        """
        Calculate liquidation price for a position.
        
        The price at which maintenance margin would be breached.
        
        Args:
            position: Position
            market: Current market
            equity: Available equity
        
        Returns:
            Liquidation price, or None if position has no liquidation risk
        """
        config = self.get_venue_config(position.venue)
        current_price = position.instrument.get_current_price(market)
        
        if position.quantity == 0:
            return None
        
        # Linear approximation for liquidation threshold.
        # equity + quantity * (liq_price - current_price) = MM
        # MM = |quantity * liq_price| * mm_rate
        
        mm_rate = config.maintenance_margin_rate
        
        if position.quantity > 0:  # Long position
            # Liquidate if price drops too much
            # equity = quantity * liq_price * mm_rate - quantity * (current_price - liq_price)
            # Solving: liq_price = (equity + quantity * current_price) / (quantity * (1 + mm_rate))
            liq_price = (equity + position.quantity * current_price) / (position.quantity * (1 + mm_rate))
        else:  # Short position
            # Liquidate if price rises too much
            # equity = |quantity| * liq_price * mm_rate + quantity * (current_price - liq_price)
            # Note: quantity is negative
            qty_abs = abs(position.quantity)
            liq_price = (equity - position.quantity * current_price) / (qty_abs * (1 - mm_rate))
        
        return liq_price if liq_price > 0 else None

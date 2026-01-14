"""Rebalancing logic and schedules."""

from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from risk_engine.portfolio.positions import Portfolio
from risk_engine.instruments.base import MarketData
from risk_engine.hedging.delta import DeltaHedger, HedgeRecommendation


@dataclass
class RebalanceConfig:
    """Configuration for rebalancing strategy."""
    
    # Time-based rebalancing
    periodic_rebalance: bool = True
    rebalance_period_hours: int = 1  # Rebalance every hour
    
    # Threshold-based rebalancing
    threshold_rebalance: bool = True
    delta_threshold_pct: float = 0.05  # Rebalance if delta > 5% of notional
    
    # Execution parameters
    execution_delay_seconds: float = 0.0  # Simulated execution delay
    min_trade_size: float = 100.0  # Minimum trade size in USD
    
    # Cost controls
    max_rebalance_cost_pct: float = 0.01  # Max 1% of NAV per rebalance


@dataclass
class RebalanceEvent:
    """Record of a rebalancing event."""
    
    timestamp: datetime
    trigger: str  # 'periodic', 'threshold', 'manual'
    hedges_executed: dict  # symbol -> HedgeRecommendation
    total_cost: float
    delta_before: dict
    delta_after: dict


class RebalanceScheduler:
    """
    Scheduler for portfolio rebalancing.
    
    Determines when to rebalance based on time and/or thresholds.
    """
    
    def __init__(
        self,
        config: RebalanceConfig,
        hedger: DeltaHedger
    ):
        """
        Initialize rebalance scheduler.
        
        Args:
            config: Rebalancing configuration
            hedger: Delta hedger to use
        """
        self.config = config
        self.hedger = hedger
        self.last_rebalance: Optional[datetime] = None
        self.rebalance_history: list[RebalanceEvent] = []
    
    def should_rebalance(
        self,
        portfolio: Portfolio,
        market: MarketData
    ) -> tuple[bool, str]:
        """
        Check if rebalancing should occur.
        
        Args:
            portfolio: Current portfolio
            market: Current market data
        
        Returns:
            Tuple of (should_rebalance, reason)
        """
        current_time = market.timestamp
        
        # Check periodic rebalance
        if self.config.periodic_rebalance:
            if self.last_rebalance is None:
                return True, "initial_rebalance"
            
            hours_since_last = (current_time - self.last_rebalance).total_seconds() / 3600
            if hours_since_last >= self.config.rebalance_period_hours:
                return True, "periodic"
        
        # Check threshold rebalance
        if self.config.threshold_rebalance:
            net_deltas = portfolio.net_delta(market)
            total_notional = portfolio.total_notional_exposure(market)
            
            if total_notional > 0:
                for symbol, delta in net_deltas.items():
                    try:
                        price = market.get_spot_price(symbol)
                        delta_notional = abs(delta * price)
                        delta_pct = delta_notional / total_notional
                        
                        if delta_pct > self.config.delta_threshold_pct:
                            return True, f"threshold_breach_{symbol}"
                    except ValueError:
                        continue
        
        return False, "no_trigger"
    
    def execute_rebalance(
        self,
        portfolio: Portfolio,
        market: MarketData,
        venue: str = "default",
        trigger: str = "manual"
    ) -> RebalanceEvent:
        """
        Execute a rebalancing event.
        
        Args:
            portfolio: Portfolio to rebalance
            market: Current market
            venue: Venue for trades
            trigger: Trigger reason
        
        Returns:
            RebalanceEvent record
        """
        # Record state before
        delta_before = portfolio.net_delta(market)
        
        # Calculate and execute hedges
        recommendations = self.hedger.calculate_hedge_need(portfolio, market)
        
        # Filter by minimum trade size
        filtered_recs = {
            symbol: rec for symbol, rec in recommendations.items()
            if abs(rec.quantity * market.get_spot_price(symbol)) >= self.config.min_trade_size
        }
        
        # Execute hedges
        total_cost = 0.0
        executed_hedges = {}
        
        for symbol, rec in filtered_recs.items():
            # Check cost limit
            nav = portfolio.net_asset_value(market)
            if total_cost + rec.expected_cost <= self.config.max_rebalance_cost_pct * nav:
                self.hedger.execute_hedge(portfolio, market, rec, venue)
                executed_hedges[symbol] = rec
                total_cost += rec.expected_cost
        
        # Record state after
        delta_after = portfolio.net_delta(market)
        
        # Create event record
        event = RebalanceEvent(
            timestamp=market.timestamp,
            trigger=trigger,
            hedges_executed=executed_hedges,
            total_cost=total_cost,
            delta_before=delta_before,
            delta_after=delta_after
        )
        
        self.rebalance_history.append(event)
        self.last_rebalance = market.timestamp
        
        return event
    
    def check_and_rebalance(
        self,
        portfolio: Portfolio,
        market: MarketData,
        venue: str = "default"
    ) -> Optional[RebalanceEvent]:
        """
        Check if rebalance needed and execute if so.
        
        Args:
            portfolio: Portfolio
            market: Market data
            venue: Venue for trades
        
        Returns:
            RebalanceEvent if rebalance occurred, None otherwise
        """
        should_rebal, trigger = self.should_rebalance(portfolio, market)
        
        if should_rebal:
            return self.execute_rebalance(portfolio, market, venue, trigger)
        
        return None
    
    def get_rebalance_summary(self) -> dict:
        """Get summary statistics of rebalancing activity."""
        if not self.rebalance_history:
            return {
                "total_rebalances": 0,
                "total_cost": 0.0,
                "avg_cost_per_rebalance": 0.0
            }
        
        total_cost = sum(event.total_cost for event in self.rebalance_history)
        
        return {
            "total_rebalances": len(self.rebalance_history),
            "total_cost": total_cost,
            "avg_cost_per_rebalance": total_cost / len(self.rebalance_history),
            "triggers": {
                event.trigger: sum(1 for e in self.rebalance_history if e.trigger == event.trigger)
                for event in self.rebalance_history
            }
        }

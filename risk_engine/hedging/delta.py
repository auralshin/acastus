"""Delta hedging logic."""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from risk_engine.portfolio.positions import Portfolio, Position
from risk_engine.instruments.base import MarketData, Instrument
from risk_engine.instruments.spot import SpotAsset
from risk_engine.instruments.perp import PerpetualSwap


@dataclass
class HedgeTarget:
    """Target delta exposure for hedging."""
    
    symbol: str
    target_delta: float = 0.0  # Target delta (typically 0 for delta neutral)
    tolerance_pct: float = 0.05  # Acceptable deviation (5% of notional)
    
    def is_within_tolerance(self, current_delta: float, portfolio_notional: float) -> bool:
        """Check if current delta is within tolerance."""
        if portfolio_notional == 0:
            return True
        
        delta_deviation = abs(current_delta - self.target_delta)
        deviation_pct = delta_deviation / portfolio_notional
        
        return deviation_pct <= self.tolerance_pct


@dataclass
class HedgeRecommendation:
    """Recommendation for a hedge trade."""
    
    symbol: str
    instrument_type: str  # 'spot' or 'perp'
    quantity: float  # Positive = buy, negative = sell
    current_delta: float
    target_delta: float
    expected_cost: float
    reason: str


class DeltaHedger:
    """
    Delta hedging engine.
    
    Calculates and executes trades to maintain delta neutrality.
    """
    
    def __init__(
        self,
        hedge_targets: Optional[Dict[str, HedgeTarget]] = None,
        hedge_instrument: str = "perp"  # 'spot' or 'perp'
    ):
        """
        Initialize delta hedger.
        
        Args:
            hedge_targets: Dictionary of symbol -> HedgeTarget
            hedge_instrument: Instrument type to use for hedging
        """
        self.hedge_targets = hedge_targets or {}
        self.hedge_instrument = hedge_instrument
    
    def set_hedge_target(self, target: HedgeTarget) -> None:
        """Set or update hedge target for a symbol."""
        self.hedge_targets[target.symbol] = target
    
    def calculate_hedge_need(
        self,
        portfolio: Portfolio,
        market: MarketData
    ) -> Dict[str, HedgeRecommendation]:
        """
        Calculate required hedge trades.
        
        Args:
            portfolio: Current portfolio
            market: Market data
        
        Returns:
            Dictionary of symbol -> HedgeRecommendation
        """
        recommendations = {}
        
        # Get current net deltas
        net_deltas = portfolio.net_delta(market)
        total_notional = portfolio.total_notional_exposure(market)
        
        for symbol, current_delta in net_deltas.items():
            # Get target for this symbol
            target = self.hedge_targets.get(
                symbol,
                HedgeTarget(symbol, target_delta=0.0, tolerance_pct=0.05)
            )
            
            # Check if hedge needed
            if target.is_within_tolerance(current_delta, total_notional):
                continue
            
            # Calculate hedge quantity
            delta_to_hedge = target.target_delta - current_delta
            
            # Determine instrument and price
            if self.hedge_instrument == "perp":
                inst_symbol = f"{symbol}-PERP"
                price = market.get_perp_price(inst_symbol)
            else:
                inst_symbol = symbol
                price = market.get_spot_price(symbol)
            
            # Expected cost (notional * slippage estimate)
            hedge_notional = abs(delta_to_hedge * price)
            expected_cost = hedge_notional * 0.001  # Rough 10 bps estimate
            
            recommendations[symbol] = HedgeRecommendation(
                symbol=symbol,
                instrument_type=self.hedge_instrument,
                quantity=delta_to_hedge,
                current_delta=current_delta,
                target_delta=target.target_delta,
                expected_cost=expected_cost,
                reason=f"Delta {current_delta:.2f} outside tolerance for {symbol}"
            )
        
        return recommendations
    
    def execute_hedge(
        self,
        portfolio: Portfolio,
        market: MarketData,
        recommendation: HedgeRecommendation,
        venue: str = "default"
    ) -> Position:
        """
        Execute a hedge trade.
        
        Args:
            portfolio: Portfolio to add hedge to
            market: Current market
            recommendation: Hedge recommendation
            venue: Venue to execute on
        
        Returns:
            New hedge position
        """
        # Create instrument
        if recommendation.instrument_type == "perp":
            instrument = PerpetualSwap(f"{recommendation.symbol}-PERP")
            entry_price = market.get_perp_price(instrument.symbol)
        else:
            instrument = SpotAsset(recommendation.symbol)
            entry_price = market.get_spot_price(recommendation.symbol)
        
        # Create position
        position = Position(
            instrument=instrument,
            quantity=recommendation.quantity,
            entry_price=entry_price,
            venue=venue,
            entry_timestamp=market.timestamp.isoformat()
        )
        
        # Add to portfolio
        portfolio.add_position(position)
        
        # Deduct costs from cash
        portfolio.cash -= recommendation.expected_cost
        
        return position
    
    def auto_hedge(
        self,
        portfolio: Portfolio,
        market: MarketData,
        venue: str = "default"
    ) -> Dict[str, Position]:
        """
        Automatically calculate and execute all needed hedges.
        
        Args:
            portfolio: Portfolio
            market: Market data
            venue: Venue for hedges
        
        Returns:
            Dictionary of symbol -> new hedge position
        """
        recommendations = self.calculate_hedge_need(portfolio, market)
        
        executed_hedges = {}
        for symbol, rec in recommendations.items():
            hedge_position = self.execute_hedge(portfolio, market, rec, venue)
            executed_hedges[symbol] = hedge_position
        
        return executed_hedges

"""Liquidation mechanics and modeling."""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from risk_engine.portfolio.positions import Portfolio, Position
from risk_engine.instruments.base import MarketData
from risk_engine.margin.models import MarginModel
from risk_engine.risk_factors.liquidity import LiquidityModel


@dataclass
class LiquidationEvent:
    """Record of a liquidation event."""
    
    venue: str
    positions_closed: List[Position]
    market_at_liquidation: MarketData
    equity_before: float
    equity_after: float
    liquidation_cost: float
    pnl_loss: float
    
    def total_loss(self) -> float:
        """Total loss from liquidation."""
        return self.liquidation_cost + self.pnl_loss


@dataclass
class LiquidationRisk:
    """Liquidation risk metrics."""
    
    probability: float  # Probability of liquidation
    expected_loss: float  # Expected loss given liquidation
    expected_shortfall: float  # Tail loss
    distance_to_liquidation: float  # Price move to trigger liquidation
    liquidation_prices: dict  # Asset -> liquidation price


class LiquidationModel:
    """
    Model for liquidation mechanics and risk.
    
    Handles:
    - Liquidation triggers (margin breach)
    - Forced close execution
    - Liquidation costs (fees + slippage)
    - Liquidation probability estimation
    """
    
    def __init__(
        self,
        margin_model: MarginModel,
        liquidity_model: LiquidityModel
    ):
        """
        Initialize liquidation model.
        
        Args:
            margin_model: Margin model for requirements
            liquidity_model: Liquidity model for execution costs
        """
        self.margin_model = margin_model
        self.liquidity_model = liquidity_model
    
    def check_liquidation_trigger(
        self,
        portfolio: Portfolio,
        market: MarketData,
        venue: Optional[str] = None
    ) -> bool:
        """
        Check if liquidation should be triggered.
        
        Args:
            portfolio: Portfolio
            market: Market data
            venue: Specific venue to check (None = check all)
        
        Returns:
            True if liquidation triggered
        """
        at_risk = self.margin_model.check_liquidation_risk(
            portfolio, market, by_venue=(venue is None)
        )
        
        if venue:
            return at_risk.get(venue, False)
        else:
            return any(at_risk.values())
    
    def simulate_liquidation(
        self,
        portfolio: Portfolio,
        market: MarketData,
        venue: str,
        is_stress: bool = True
    ) -> LiquidationEvent:
        """
        Simulate a liquidation event.
        
        Args:
            portfolio: Portfolio
            market: Market at liquidation
            venue: Venue being liquidated
            is_stress: Whether liquidation occurs during stress
        
        Returns:
            LiquidationEvent with costs and outcomes
        """
        config = self.margin_model.get_venue_config(venue)
        equity_before = portfolio.net_asset_value(market)
        
        # Get positions to close
        by_venue = portfolio.positions_by_venue()
        positions_to_close = by_venue.get(venue, [])
        
        # Calculate liquidation costs
        total_notional = sum(pos.notional_value(market) for pos in positions_to_close)
        
        # Liquidation fee
        liquidation_fee = total_notional * config.liquidation_fee
        
        # Market impact / slippage (worse during stress)
        slippage_cost = self.liquidity_model.calculate_slippage(
            total_notional,
            current_volatility=1.5 if is_stress else 0.75,
            base_volatility=0.75,
            is_stress=is_stress
        ) * total_notional
        
        total_cost = liquidation_fee + slippage_cost
        
        # Calculate P&L loss from closing positions
        pnl_loss = sum(pos.unrealized_pnl(market) for pos in positions_to_close)
        pnl_loss = min(pnl_loss, 0)  # Only count losses
        
        # Update equity
        equity_after = equity_before - total_cost + pnl_loss
        
        return LiquidationEvent(
            venue=venue,
            positions_closed=positions_to_close,
            market_at_liquidation=market,
            equity_before=equity_before,
            equity_after=equity_after,
            liquidation_cost=total_cost,
            pnl_loss=abs(pnl_loss)
        )
    
    def calculate_liquidation_risk(
        self,
        portfolio: Portfolio,
        market: MarketData,
        price_scenarios: np.ndarray,
        symbol: str = "ETH"
    ) -> LiquidationRisk:
        """
        Calculate liquidation risk metrics using price scenarios.
        
        Args:
            portfolio: Portfolio
            market: Current market
            price_scenarios: Array of potential future prices
            symbol: Symbol to stress (e.g., 'ETH')
        
        Returns:
            LiquidationRisk metrics
        """
        current_price = market.get_spot_price(symbol)
        liquidations = []
        losses = []
        
        # Test each price scenario
        for scenario_price in price_scenarios:
            # Create stressed market
            stressed_market = MarketData(
                timestamp=market.timestamp,
                spot_prices={symbol: scenario_price},
                perp_prices={f"{symbol}-PERP": scenario_price},
                funding_rates=market.funding_rates.copy()
            )
            
            # Check for liquidation
            if self.check_liquidation_trigger(portfolio, stressed_market):
                liquidations.append(True)
                
                # Estimate loss
                equity_loss = portfolio.net_asset_value(market) - portfolio.net_asset_value(stressed_market)
                
                # Add liquidation costs using an assumed cost rate
                total_notional = portfolio.total_notional_exposure(stressed_market)
                liquidation_cost_rate = 0.02
                liq_costs = total_notional * liquidation_cost_rate
                
                total_loss = equity_loss + liq_costs
                losses.append(max(0, total_loss))
            else:
                liquidations.append(False)
                losses.append(0)
        
        # Calculate metrics
        prob_liquidation = np.mean(liquidations)
        
        if prob_liquidation > 0:
            losses_given_liq = [l for l, liq in zip(losses, liquidations) if liq]
            expected_loss = np.mean(losses_given_liq)
            expected_shortfall = np.percentile(losses_given_liq, 95)
        else:
            expected_loss = 0
            expected_shortfall = 0
        
        # Distance to liquidation (price move)
        liquidation_prices_down = []
        liquidation_prices_up = []
        
        for pos in portfolio.positions:
            if pos.instrument.symbol == symbol or pos.instrument.symbol == f"{symbol}-PERP":
                liq_price = self.margin_model.liquidation_price(
                    pos, market, portfolio.net_asset_value(market)
                )
                if liq_price:
                    if pos.quantity > 0:
                        liquidation_prices_down.append(liq_price)
                    else:
                        liquidation_prices_up.append(liq_price)
        
        # Distance is the closest liquidation price
        distance_down = min(liquidation_prices_down) if liquidation_prices_down else 0
        distance_up = min(liquidation_prices_up) if liquidation_prices_up else float('inf')
        
        distance_to_liq_pct = min(
            abs(distance_down - current_price) / current_price if distance_down > 0 else float('inf'),
            abs(distance_up - current_price) / current_price
        )
        
        return LiquidationRisk(
            probability=prob_liquidation,
            expected_loss=expected_loss,
            expected_shortfall=expected_shortfall,
            distance_to_liquidation=distance_to_liq_pct,
            liquidation_prices={
                f"{symbol}_down": distance_down,
                f"{symbol}_up": distance_up
            }
        )
    
    def calculate_effective_gamma(
        self,
        portfolio: Portfolio,
        market: MarketData,
        price_bump_pct: float = 0.01
    ) -> float:
        """
        Calculate "effective gamma" from liquidation convexity.
        
        Measures how P&L changes non-linearly near liquidation prices
        due to forced unwinds.
        
        Args:
            portfolio: Portfolio
            market: Current market
            price_bump_pct: Price bump size (1% default)
        
        Returns:
            Effective gamma (second derivative of equity w.r.t. price)
        """
        # Finite-difference approximation near liquidation thresholds.
        
        # Get current NAV and margin status
        nav_0 = portfolio.net_asset_value(market)
        # Bump price down
        symbol = "ETH"  # Assume ETH is primary risk
        price_0 = market.get_spot_price(symbol)
        price_down = price_0 * (1 - price_bump_pct)
        
        market_down = MarketData(
            timestamp=market.timestamp,
            spot_prices={symbol: price_down},
            perp_prices={f"{symbol}-PERP": price_down},
            funding_rates=market.funding_rates.copy()
        )
        
        nav_down = portfolio.net_asset_value(market_down)
        delta_nav_down = nav_down - nav_0
        
        # Bump price up
        price_up = price_0 * (1 + price_bump_pct)
        market_up = MarketData(
            timestamp=market.timestamp,
            spot_prices={symbol: price_up},
            perp_prices={f"{symbol}-PERP": price_up},
            funding_rates=market.funding_rates.copy()
        )
        
        nav_up = portfolio.net_asset_value(market_up)
        delta_nav_up = nav_up - nav_0
        
        # Second derivative
        gamma = (delta_nav_up - 2 * nav_0 + delta_nav_down) / (price_bump_pct * price_0)**2
        
        return gamma

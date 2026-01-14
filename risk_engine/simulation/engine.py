"""Portfolio simulation engine."""

import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from risk_engine.portfolio.positions import Portfolio, Position
from risk_engine.portfolio.valuation import PortfolioValuation, PnLAttribution
from risk_engine.instruments.base import MarketData
from risk_engine.instruments.perp import PerpetualSwap
from risk_engine.instruments.spot import SpotAsset
from risk_engine.simulation.paths import ScenarioPath
from risk_engine.margin.models import MarginModel, MarginRequirements
from risk_engine.hedging.delta import DeltaHedger, HedgeRecommendation
from risk_engine.risk_factors.counterparty import CounterpartyRiskModel, CounterpartyScenario, VenueStatus


@dataclass
class SimulationState:
    """State of portfolio at a simulation step."""
    
    timestamp: datetime
    portfolio: Portfolio
    market: MarketData
    nav: float
    margin_requirements: Optional[Dict[str, MarginRequirements]] = None
    is_liquidated: bool = False
    counterparty_events: List = field(default_factory=list)  # Venue failures/halts
    venue_haircuts: Dict[str, float] = field(default_factory=dict)
    hedge_trades: List[HedgeRecommendation] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    
    scenario_id: int
    states: List[SimulationState]
    pnl_attributions: List[PnLAttribution]
    final_nav: float
    max_drawdown: float
    was_liquidated: bool
    liquidation_time: Optional[datetime] = None
    
    def nav_path(self) -> np.ndarray:
        """Get NAV path as numpy array."""
        return np.array([state.nav for state in self.states])
    
    def total_return(self) -> float:
        """Calculate total return."""
        if len(self.states) == 0:
            return 0.0
        initial_nav = self.states[0].nav
        return (self.final_nav - initial_nav) / initial_nav if initial_nav > 0 else 0.0


class SimulationEngine:
    """
    Portfolio simulation engine.
    
    Evolves portfolio through scenario paths with:
    - Dynamic hedging
    - Margin management
    - Liquidation mechanics
    - P&L attribution
    """
    
    def __init__(
        self,
        initial_portfolio: Portfolio,
        margin_model: Optional[MarginModel] = None,
        hedger: Optional['DeltaHedger'] = None,
        slippage_model: Optional[Callable] = None,
        counterparty_model: Optional[CounterpartyRiskModel] = None,
        counterparty_scenario: Optional[CounterpartyScenario] = None
    ):
        """
        Initialize simulation engine.
        
        Args:
            initial_portfolio: Starting portfolio
            margin_model: Margin calculation model
            hedger: Delta hedging strategy
            slippage_model: Function(quantity, volatility) -> slippage_cost
            counterparty_model: Counterparty risk model
            counterparty_scenario: Optional counterparty scenario to apply
        """
        self.initial_portfolio = initial_portfolio
        self.margin_model = margin_model or MarginModel()
        self.hedger = hedger
        self.slippage_model = slippage_model or self._default_slippage
        self.counterparty_model = counterparty_model
        self.counterparty_scenario = counterparty_scenario
        self.hedger = hedger
        self.slippage_model = slippage_model or self._default_slippage
    
    def _default_slippage(self, quantity: float, notional: float, volatility: float = 0.5) -> float:
        """
        Default slippage model.
        
        Args:
            quantity: Trade quantity
            notional: Trade notional value
            volatility: Market volatility regime
        
        Returns:
            Slippage cost (positive = cost)
        """
        base_slippage_bps = 5.0
        vol_multiplier = 1.0 + volatility
        
        # Impact scales with square root of size
        size_factor = np.sqrt(abs(notional) / 100000)  # Normalized to $100k
        
        slippage_bps = base_slippage_bps * vol_multiplier * size_factor
        return abs(notional) * slippage_bps / 10000
    
    def _create_market_data(self, scenario: ScenarioPath, step: int) -> MarketData:
        """Create MarketData object from scenario path at given step."""
        spot_prices = {
            symbol: float(path[step])
            for symbol, path in scenario.spot_prices.items()
        }

        # For perps, apply basis if available
        perp_prices = {}
        for symbol, path in scenario.spot_prices.items():
            spot_price = float(path[step])
            if scenario.basis_spreads and symbol in scenario.basis_spreads:
                basis = float(scenario.basis_spreads[symbol][step])
                perp_prices[f"{symbol}-PERP"] = spot_price + basis
            else:
                perp_prices[f"{symbol}-PERP"] = spot_price
        
        funding_rates = {
            f"{symbol}-PERP": float(path[step])
            for symbol, path in scenario.funding_rates.items()
        }

        volatilities = None
        if scenario.volatilities:
            volatilities = {
                symbol: float(path[step])
                for symbol, path in scenario.volatilities.items()
            }

        liquidity = None
        if scenario.liquidity_multipliers:
            liquidity = {
                symbol: float(path[step])
                for symbol, path in scenario.liquidity_multipliers.items()
            }
        
        return MarketData(
            timestamp=scenario.timestamps[step],
            spot_prices=spot_prices,
            perp_prices=perp_prices,
            funding_rates=funding_rates,
            volatilities=volatilities,
            liquidity=liquidity
        )
    
    def _apply_hedge_trades(
        self,
        portfolio: Portfolio,
        recommendations: Dict[str, HedgeRecommendation],
        market: MarketData
    ) -> float:
        """
        Apply hedge trades to portfolio.
        
        Args:
            portfolio: Portfolio to modify
            recommendations: Hedge recommendations
            market: Current market data
        
        Returns:
            Total trading costs (fees + slippage)
        """
        total_cost = 0.0
        
        for rec in recommendations.values():
            if abs(rec.quantity) < 1e-6:
                continue

            if rec.instrument_type == "perp":
                instrument_symbol = f"{rec.symbol}-PERP"
                price = market.get_perp_price(instrument_symbol)
                instrument = PerpetualSwap(instrument_symbol)
            else:
                instrument_symbol = rec.symbol
                price = market.get_spot_price(instrument_symbol)
                instrument = SpotAsset(instrument_symbol)
                portfolio.cash -= rec.quantity * price

            notional = abs(rec.quantity * price)
            volatility = 0.5
            if market.volatilities and rec.symbol in market.volatilities:
                volatility = market.volatilities[rec.symbol]
            liquidity_multiplier = market.get_liquidity_multiplier(rec.symbol)
            slippage = self.slippage_model(
                rec.quantity,
                notional,
                volatility * liquidity_multiplier
            )
            total_cost += slippage
            portfolio.cash -= slippage

            portfolio.add_position(
                Position(
                    instrument=instrument,
                    quantity=rec.quantity,
                    entry_price=price,
                    venue="default"
                )
            )
        
        return total_cost
    
    def _check_liquidation(
        self,
        portfolio: Portfolio,
        market: MarketData,
        margin_reqs: Dict
    ) -> bool:
        """
        Check if portfolio should be liquidated.
        
        Args:
            portfolio: Current portfolio
            market: Market data
            margin_reqs: Dictionary of margin requirements
        
        Returns:
            True if liquidated
        """
        # Check if any venue has insufficient margin
        for venue_reqs in margin_reqs.values():
            if venue_reqs.margin_ratio < 1.0:
                return True
        return False
    
    def run_scenario(
        self,
        scenario: ScenarioPath,
        scenario_id: int = 0,
        rebalance_frequency: int = 4  # Rebalance every N steps
    ) -> SimulationResult:
        """
        Run simulation through a scenario path.
        
        Args:
            scenario: Scenario path to simulate
            scenario_id: Identifier for this scenario
            rebalance_frequency: Steps between rebalances
        
        Returns:
            SimulationResult
        """
        # Deep copy initial portfolio
        portfolio = Portfolio(
            positions=self.initial_portfolio.positions.copy(),
            cash=self.initial_portfolio.cash
        )
        
        states = []
        pnl_attributions = []
        was_liquidated = False
        liquidation_time = None
        
        valuation = PortfolioValuation(portfolio)
        
        for step in range(len(scenario)):
            market = self._create_market_data(scenario, step)
            
            # Calculate NAV
            nav = portfolio.net_asset_value(market)
            
            # Calculate margin requirements
            margin_reqs = self.margin_model.calculate_portfolio_margin(
                portfolio, market, by_venue=False
            )
            
            # Apply counterparty scenario if configured
            venue_haircuts = {}
            counterparty_events_step = []
            
            if self.counterparty_scenario and step == len(scenario) // 2:  # Apply at midpoint
                # Apply venue haircuts
                venue_exposures = {}
                for pos in portfolio.positions:
                    venue = pos.venue if hasattr(pos, 'venue') else 'default'
                    if venue not in venue_exposures:
                        venue_exposures[venue] = 0.0
                    # Get price based on instrument type
                    if hasattr(pos, 'instrument') and hasattr(pos.instrument, 'symbol'):
                        symbol = pos.instrument.symbol
                        price = market.get_spot_price(symbol) if symbol in market.spot_prices else market.get_perp_price(symbol)
                    else:
                        price = 1.0  # Fallback for unknown instruments
                    venue_exposures[venue] += abs(pos.quantity * price)
                
                for event in self.counterparty_scenario.events:
                    if event.venue in venue_exposures or event.venue == "default":
                        target_venues = [event.venue] if event.venue != "default" else list(venue_exposures.keys())
                        
                        for venue in target_venues:
                            if venue in venue_exposures:
                                loss = venue_exposures[venue] * event.haircut
                                portfolio.cash -= loss
                                venue_haircuts[venue] = event.haircut
                                counterparty_events_step.append(event)
            
            # Check for liquidation
            if self._check_liquidation(portfolio, market, margin_reqs):
                was_liquidated = True
                liquidation_time = market.timestamp
                nav = 0.0  # Assume total loss on liquidation
                
                state = SimulationState(
                    timestamp=market.timestamp,
                    portfolio=portfolio,
                    market=market,
                    nav=nav,
                    margin_requirements=margin_reqs,
                    is_liquidated=True,
                    counterparty_events=counterparty_events_step,
                    venue_haircuts=venue_haircuts
                )
                states.append(state)
                break
            
            # Apply hedging logic
            hedge_recs = {}
            if self.hedger and step % rebalance_frequency == 0:
                hedge_recs = self.hedger.calculate_hedge_need(portfolio, market)
                if hedge_recs:
                    cost = self._apply_hedge_trades(portfolio, hedge_recs, market)
                    nav -= cost
            
            # Record state
            state = SimulationState(
                timestamp=market.timestamp,
                portfolio=portfolio,
                market=market,
                nav=nav,
                margin_requirements=margin_reqs,
                counterparty_events=counterparty_events_step,
                venue_haircuts=venue_haircuts,
                hedge_trades=list(hedge_recs.values())
            )
            states.append(state)
            
            # Calculate P&L attribution if not first step
            if step > 0:
                prev_market = self._create_market_data(scenario, step - 1)
                pnl = valuation.calculate_period_pnl(prev_market, market)
                pnl_attributions.append(pnl)
        
        # Calculate drawdown
        nav_path = np.array([s.nav for s in states])
        running_max = np.maximum.accumulate(nav_path)
        drawdowns = (nav_path - running_max) / running_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
        
        return SimulationResult(
            scenario_id=scenario_id,
            states=states,
            pnl_attributions=pnl_attributions,
            final_nav=states[-1].nav if states else 0.0,
            max_drawdown=max_drawdown,
            was_liquidated=was_liquidated,
            liquidation_time=liquidation_time
        )
    
    def run_multiple_scenarios(
        self,
        scenarios: List[ScenarioPath],
        parallel: bool = True,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[SimulationResult]:
        """
        Run simulation across multiple scenarios.
        
        Args:
            scenarios: List of scenario paths
            parallel: Whether to run in parallel (default: True for speedup)
            max_workers: Number of parallel workers (default: CPU count)
            progress_callback: Optional callback(completed, total) for progress tracking
        
        Returns:
            List of SimulationResult objects
        """
        if parallel and len(scenarios) > 10:
            # Use parallel processing for significant speedup
            results = []
            completed = 0
            total = len(scenarios)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all scenarios
                future_to_idx = {
                    executor.submit(self._run_scenario_worker, scenario, i): i 
                    for i, scenario in enumerate(scenarios)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_idx):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total)
            
            # Sort by scenario_id to maintain order
            results.sort(key=lambda r: r.scenario_id)
            return results
        else:
            # Sequential processing for small batches or when parallel=False
            results = []
            for i, scenario in enumerate(scenarios):
                result = self.run_scenario(scenario, scenario_id=i)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(scenarios))
            
            return results
    
    def _run_scenario_worker(self, scenario: ScenarioPath, scenario_id: int) -> SimulationResult:
        """Worker function for parallel scenario execution."""
        return self.run_scenario(scenario, scenario_id=scenario_id)

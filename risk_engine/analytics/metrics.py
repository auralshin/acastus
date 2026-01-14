"""Additional risk metrics and analytics."""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from risk_engine.simulation.engine import SimulationResult

# Try to import JIT functions for performance
try:
    from risk_engine.utils.jit_functions import calculate_var_from_returns
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a portfolio."""
    
    # Return metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration_days: float
    
    # Tail risk
    var_95: float
    var_99: float
    es_975: float
    
    # Liquidation risk
    liquidation_probability: float
    expected_time_to_liquidation_days: Optional[float]
    
    # Other
    num_scenarios: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "es_975": self.es_975,
            "liquidation_probability": self.liquidation_probability,
            "expected_time_to_liquidation_days": self.expected_time_to_liquidation_days,
            "num_scenarios": self.num_scenarios,
        }


@dataclass
class StressTestResult:
    """Result from a stress test scenario."""
    
    scenario_name: str
    description: str
    parameters: Dict
    final_nav: float
    nav_change_pct: float
    max_drawdown: float
    liquidated: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "scenario": self.scenario_name,
            "description": self.description,
            "parameters": self.parameters,
            "final_nav": self.final_nav,
            "nav_change_pct": self.nav_change_pct,
            "max_drawdown": self.max_drawdown,
            "liquidated": self.liquidated,
        }


class RiskMetricsCalculator:
    """Calculate comprehensive risk metrics from simulation results."""
    
    @staticmethod
    def calculate_from_simulations(
        results: List[SimulationResult],
        horizon_days: float = 30,
        risk_free_rate: float = 0.0
    ) -> RiskMetrics:
        """
        Calculate risk metrics from Monte Carlo simulation results.
        
        Args:
            results: List of simulation results
            horizon_days: Simulation horizon in days
            risk_free_rate: Risk-free rate for Sharpe ratio
        
        Returns:
            RiskMetrics
        """
        if len(results) == 0:
            return RiskMetrics(
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_duration_days=0.0,
                var_95=0.0,
                var_99=0.0,
                es_975=0.0,
                liquidation_probability=0.0,
                expected_time_to_liquidation_days=None,
                num_scenarios=0
            )
        
        # Extract returns
        returns = np.array([r.total_return() for r in results])
        
        # Return metrics
        mean_return = np.mean(returns)
        annualized_return = mean_return * (365 / horizon_days)
        volatility = np.std(returns) * np.sqrt(365 / horizon_days)
        
        sharpe = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # Drawdown metrics
        drawdowns = np.array([r.max_drawdown for r in results])
        max_dd = np.max(drawdowns)

        def max_drawdown_duration_days(result: SimulationResult) -> float:
            if len(result.states) < 2:
                return 0.0

            running_max = -np.inf
            peak_time = result.states[0].timestamp
            max_duration = 0.0

            for state in result.states:
                if state.nav >= running_max:
                    running_max = state.nav
                    peak_time = state.timestamp
                else:
                    duration = (state.timestamp - peak_time).total_seconds() / 86400
                    if duration > max_duration:
                        max_duration = duration

            return max_duration

        max_dd_duration = max(
            (max_drawdown_duration_days(result) for result in results),
            default=0.0
        )
        
        # VaR and ES
        if JIT_AVAILABLE:
            # Use JIT-compiled VaR calculation
            confidence_levels = np.array([0.95, 0.99])
            var_values = calculate_var_from_returns(returns, confidence_levels)
            var_95, var_99 = var_values[0], var_values[1]
            
            # ES: average of returns below 2.5th percentile
            tail_threshold = np.quantile(returns, 0.025)
            tail_returns = returns[returns <= tail_threshold]
            es_975 = -np.mean(tail_returns) if len(tail_returns) > 0 else var_99
        else:
            # Fallback to Python implementation
            var_95 = -np.quantile(returns, 0.05)
            var_99 = -np.quantile(returns, 0.01)
            
            # ES: average of returns below 2.5th percentile
            tail_threshold = np.quantile(returns, 0.025)
            tail_returns = returns[returns <= tail_threshold]
            es_975 = -np.mean(tail_returns) if len(tail_returns) > 0 else var_99
        
        # Liquidation probability
        n_liquidated = sum(1 for r in results if r.was_liquidated)
        liquidation_prob = n_liquidated / len(results)
        
        # Expected time to liquidation (for liquidated scenarios)
        liquidation_times = []
        for r in results:
            if r.was_liquidated and r.liquidation_time and len(r.states) > 0:
                initial_time = r.states[0].timestamp
                hours = (r.liquidation_time - initial_time).total_seconds() / 3600
                liquidation_times.append(hours / 24)
        
        expected_liq_time = np.mean(liquidation_times) if liquidation_times else None
        
        return RiskMetrics(
            total_return=mean_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration,
            var_95=var_95,
            var_99=var_99,
            es_975=es_975,
            liquidation_probability=liquidation_prob,
            expected_time_to_liquidation_days=expected_liq_time,
            num_scenarios=len(results)
        )
    
    @staticmethod
    def calculate_breach_probability(
        results: List[SimulationResult],
        threshold_nav: float
    ) -> float:
        """
        Calculate probability of NAV breaching a threshold.
        
        Args:
            results: Simulation results
            threshold_nav: NAV threshold
        
        Returns:
            Probability of breach
        """
        n_breached = 0
        
        for result in results:
            min_nav = min(state.nav for state in result.states)
            if min_nav < threshold_nav:
                n_breached += 1
        
        return n_breached / len(results) if results else 0.0
    
    @staticmethod
    def calculate_liquidity_at_risk(
        results: List[SimulationResult],
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Liquidity at Risk: cost to unwind at worst case.
        
        Args:
            results: Simulation results
            confidence_level: Confidence level
        
        Returns:
            Liquidity at risk amount
        """
        # Extract total slippage/costs from each scenario
        costs = []
        
        for result in results:
            total_cost = 0.0
            for state in result.states:
                # Proxy costs using hedge trade count
                estimated_trade_cost = 100.0
                total_cost += len(state.hedge_trades) * estimated_trade_cost
            costs.append(total_cost)
        
        costs = np.array(costs)
        lar = np.quantile(costs, confidence_level)
        
        return lar


class StressTestEngine:
    """Engine for running deterministic stress tests."""
    
    def __init__(self):
        """Initialize stress test engine."""
        self.predefined_scenarios = {
            "moderate_crash": {
                "description": "20% price drop, funding spike to 50%",
                "params": {"price_shock": -0.20, "funding_shock": 0.50}
            },
            "severe_crash": {
                "description": "40% price drop, funding spike to 100%",
                "params": {"price_shock": -0.40, "funding_shock": 1.00}
            },
            "flash_crash": {
                "description": "30% instant drop with recovery, 5x volatility",
                "params": {"price_shock": -0.30, "vol_multiplier": 5.0, "recovery": True}
            },
            "funding_crisis": {
                "description": "Funding rate crisis: 200% annual, basis blowout",
                "params": {"funding_shock": 2.00, "basis_shock": 0.10}
            },
            "venue_failure": {
                "description": "Major venue failure with haircuts",
                "params": {"venue": "binance", "haircut": 0.30}
            },
        }
    
    def run_stress_scenario(
        self,
        scenario_name: str,
        initial_nav: float,
        scenario_params: Optional[Dict] = None
    ) -> StressTestResult:
        """
        Run a single stress test scenario.
        
        Args:
            scenario_name: Name of scenario
            initial_nav: Initial NAV
            scenario_params: Override parameters
        
        Returns:
            StressTestResult
        """
        if scenario_params is None:
            if scenario_name in self.predefined_scenarios:
                scenario_def = self.predefined_scenarios[scenario_name]
                description = scenario_def["description"]
                params = scenario_def["params"]
            else:
                raise ValueError(f"Unknown scenario: {scenario_name}")
        else:
            description = "Custom stress scenario"
            params = scenario_params
        
        # Heuristic stress calculation using parameter shocks.
        # For full fidelity, run shocked scenarios through the simulation engine.
        price_shock = params.get("price_shock", 0.0)
        funding_shock = params.get("funding_shock", 0.0)
        
        # Heuristic P&L estimate
        # Assumes delta-neutral, so price shock has limited impact
        # Main impact from funding and basis
        nav_impact = initial_nav * (price_shock * 0.1 + funding_shock * -0.05)
        final_nav = initial_nav + nav_impact
        
        nav_change_pct = nav_impact / initial_nav if initial_nav > 0 else 0.0
        max_dd = abs(min(0, nav_change_pct))
        liquidated = final_nav < initial_nav * 0.5  # Heuristic liquidation threshold
        
        return StressTestResult(
            scenario_name=scenario_name,
            description=description,
            parameters=params,
            final_nav=final_nav,
            nav_change_pct=nav_change_pct,
            max_drawdown=max_dd,
            liquidated=liquidated
        )
    
    def run_all_scenarios(self, initial_nav: float) -> List[StressTestResult]:
        """
        Run all predefined stress scenarios.
        
        Args:
            initial_nav: Initial NAV
        
        Returns:
            List of StressTestResult
        """
        results = []
        
        for scenario_name in self.predefined_scenarios.keys():
            result = self.run_stress_scenario(scenario_name, initial_nav)
            results.append(result)
        
        return results

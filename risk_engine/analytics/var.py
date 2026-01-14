"""VaR (Value at Risk) calculations."""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from risk_engine.simulation.engine import SimulationResult


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    
    confidence_level: float
    var: float  # VaR amount (positive = loss)
    var_pct: float  # VaR as % of NAV
    method: str  # 'historical', 'parametric', 'monte_carlo'


@dataclass
class ExpectedShortfallResult:
    """Expected Shortfall (CVaR) result."""
    
    confidence_level: float
    es: float  # Expected shortfall (positive = loss)
    es_pct: float  # ES as % of NAV
    method: str


class VaRCalculator:
    """
    Value at Risk calculator.
    
    Supports multiple methods:
    - Historical simulation
    - Parametric (variance-covariance)
    - Monte Carlo simulation
    """
    
    @staticmethod
    def historical_var(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        initial_value: float = 1.0
    ) -> VaRResult:
        """
        Calculate VaR using historical simulation.
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            initial_value: Initial portfolio value
        
        Returns:
            VaRResult
        """
        if len(returns) == 0:
            return VaRResult(confidence_level, 0.0, 0.0, "historical")
        
        # VaR is the negative of the quantile (since we want loss magnitude)
        var_return = np.quantile(returns, 1 - confidence_level)
        var_amount = -var_return * initial_value  # Positive for loss
        var_pct = -var_return
        
        return VaRResult(
            confidence_level=confidence_level,
            var=float(var_amount),
            var_pct=float(var_pct),
            method="historical"
        )
    
    @staticmethod
    def parametric_var(
        mean_return: float,
        std_return: float,
        confidence_level: float = 0.95,
        initial_value: float = 1.0
    ) -> VaRResult:
        """
        Calculate VaR using parametric method (assumes normal distribution).
        
        Args:
            mean_return: Expected return
            std_return: Return standard deviation
            confidence_level: Confidence level
            initial_value: Initial portfolio value
        
        Returns:
            VaRResult
        """
        from scipy import stats
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR = -(mean + z * std) * value
        var_return = mean_return + z_score * std_return
        var_amount = -var_return * initial_value
        var_pct = -var_return
        
        return VaRResult(
            confidence_level=confidence_level,
            var=float(var_amount),
            var_pct=float(var_pct),
            method="parametric"
        )
    
    @staticmethod
    def monte_carlo_var(
        simulation_results: List[SimulationResult],
        confidence_level: float = 0.95
    ) -> VaRResult:
        """
        Calculate VaR from Monte Carlo simulation results.
        
        Args:
            simulation_results: List of simulation results
            confidence_level: Confidence level
        
        Returns:
            VaRResult
        """
        if len(simulation_results) == 0:
            return VaRResult(confidence_level, 0.0, 0.0, "monte_carlo")
        
        # Extract returns from simulations
        returns = []
        initial_navs = []
        
        for result in simulation_results:
            if len(result.states) >= 2:
                initial_nav = result.states[0].nav
                final_nav = result.final_nav
                ret = (final_nav - initial_nav) / initial_nav if initial_nav > 0 else 0.0
                returns.append(ret)
                initial_navs.append(initial_nav)
        
        returns = np.array(returns)
        avg_initial_nav = np.mean(initial_navs) if initial_navs else 1.0
        
        return VaRCalculator.historical_var(returns, confidence_level, float(avg_initial_nav))


class ExpectedShortfallCalculator:
    """
    Expected Shortfall (ES) / Conditional Value at Risk (CVaR) calculator.
    
    ES measures the expected loss given that loss exceeds VaR.
    """
    
    @staticmethod
    def calculate_es(
        returns: np.ndarray,
        confidence_level: float = 0.975,
        initial_value: float = 1.0
    ) -> ExpectedShortfallResult:
        """
        Calculate Expected Shortfall.
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (typically higher than VaR, e.g., 0.975)
            initial_value: Initial portfolio value
        
        Returns:
            ExpectedShortfallResult
        """
        if len(returns) == 0:
            return ExpectedShortfallResult(confidence_level, 0.0, 0.0, "historical")
        
        # Find VaR threshold
        var_threshold = np.quantile(returns, 1 - confidence_level)
        
        # ES is the average of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            es_return = var_threshold
        else:
            es_return = np.mean(tail_returns)
        
        es_amount = -es_return * initial_value
        es_pct = -es_return
        
        return ExpectedShortfallResult(
            confidence_level=confidence_level,
            es=float(es_amount),
            es_pct=float(es_pct),
            method="historical"
        )
    
    @staticmethod
    def calculate_es_from_simulations(
        simulation_results: List[SimulationResult],
        confidence_level: float = 0.975
    ) -> ExpectedShortfallResult:
        """
        Calculate ES from Monte Carlo simulations.
        
        Args:
            simulation_results: List of simulation results
            confidence_level: Confidence level
        
        Returns:
            ExpectedShortfallResult
        """
        if len(simulation_results) == 0:
            return ExpectedShortfallResult(confidence_level, 0.0, 0.0, "monte_carlo")
        
        returns = []
        initial_navs = []
        
        for result in simulation_results:
            if len(result.states) >= 2:
                initial_nav = result.states[0].nav
                final_nav = result.final_nav
                ret = (final_nav - initial_nav) / initial_nav if initial_nav > 0 else -1.0
                returns.append(ret)
                initial_navs.append(initial_nav)
        
        returns = np.array(returns)
        avg_initial_nav = np.mean(initial_navs) if initial_navs else 1.0
        
        return ExpectedShortfallCalculator.calculate_es(
            returns, confidence_level, float(avg_initial_nav)
        )

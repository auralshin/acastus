from typing import Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class LiquidityParams:
    """Parameters for liquidity/slippage modeling."""
    
    base_slippage_bps: float = 5.0  # Base slippage in basis points
    size_impact_coef: float = 0.01  # Impact per $1M notional
    volatility_multiplier: float = 2.0  # Slippage increase per vol regime
    stress_multiplier: float = 5.0  # Additional multiplier during stress


class LiquidityModel:
    """
    Model for execution costs and market impact.
    
    Slippage depends on:
    - Trade size (larger trades have more impact)
    - Market volatility (higher vol = wider spreads)
    - Stress regime (liquidity dries up)
    """
    
    def __init__(
        self,
        params: Optional[LiquidityParams] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize liquidity model.
        
        Args:
            params: Liquidity parameters
            random_seed: Random seed
        """
        self.params = params or LiquidityParams()
        self.rng = np.random.default_rng(random_seed)
    
    def calculate_slippage(
        self,
        notional: float,
        current_volatility: float = 0.75,  # Annualized vol
        base_volatility: float = 0.75,
        is_stress: bool = False
    ) -> float:
        """
        Calculate slippage for a trade.
        
        Args:
            notional: Trade notional in USD
            current_volatility: Current annualized volatility
            base_volatility: Normal volatility level
            is_stress: Whether market is in stress
        
        Returns:
            Slippage cost as fraction of notional
        """
        # Base slippage
        slippage = self.params.base_slippage_bps / 10000
        
        # Size impact (square root of notional)
        notional_millions = abs(notional) / 1_000_000
        size_impact = self.params.size_impact_coef * np.sqrt(notional_millions)
        slippage += size_impact
        
        # Volatility impact
        vol_ratio = current_volatility / base_volatility
        if vol_ratio > 1.0:
            slippage *= (1 + (vol_ratio - 1) * self.params.volatility_multiplier)
        
        # Stress impact
        if is_stress:
            slippage *= self.params.stress_multiplier
        
        return slippage
    
    def calculate_execution_cost(
        self,
        notional: float,
        current_price: float,
        current_volatility: float = 0.75,
        base_volatility: float = 0.75,
        is_stress: bool = False
    ) -> dict:
        """
        Calculate total execution cost.
        
        Args:
            notional: Trade notional
            current_price: Current market price
            current_volatility: Current volatility
            base_volatility: Normal volatility
            is_stress: Stress flag
        
        Returns:
            Dictionary with cost breakdown
        """
        slippage_pct = self.calculate_slippage(
            notional, current_volatility, base_volatility, is_stress
        )
        
        slippage_cost = abs(notional) * slippage_pct
        
        return {
            "slippage_pct": slippage_pct,
            "slippage_cost": slippage_cost,
            "total_cost": slippage_cost
        }
    
    def liquidity_at_risk(
        self,
        portfolio_notional: float,
        unwind_horizon_hours: float = 24,
        confidence: float = 0.95,
        stress_scenario: bool = True
    ) -> float:
        """
        Calculate Liquidity-at-Risk.
        
        Estimates the cost to unwind the portfolio under stress.
        
        Args:
            portfolio_notional: Total portfolio notional
            unwind_horizon_hours: Time to fully unwind (hours)
            confidence: Confidence level
            stress_scenario: Whether to use stress assumptions
        
        Returns:
            Estimated unwind cost
        """
        # Assume need to trade in smaller chunks
        n_chunks = int(unwind_horizon_hours)
        notional_per_chunk = portfolio_notional / n_chunks
        
        total_cost = 0.0
        current_vol = 1.5 if stress_scenario else 0.75  # Higher vol in stress
        
        for _ in range(n_chunks):
            chunk_cost = self.calculate_slippage(
                notional_per_chunk,
                current_vol,
                base_volatility=0.75,
                is_stress=stress_scenario
            ) * notional_per_chunk
            
            total_cost += chunk_cost
        
        # Add uncertainty factor
        if confidence == 0.95:
            total_cost *= 1.5  # 95th percentile roughly 1.5x expected
        elif confidence == 0.99:
            total_cost *= 2.0  # 99th percentile roughly 2x expected
        
        return total_cost

"""Basis risk modeling (spot-perp, LST-spot spreads)."""

from typing import Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class BasisScenario:
    """Basis spread scenario/path."""
    
    timestamps: np.ndarray
    basis: np.ndarray  # Basis in absolute terms
    basis_pct: Optional[np.ndarray] = None  # Basis as % of spot


class BasisModel:
    """Base class for basis spread models."""
    
    def __init__(self, initial_basis: float = 0.0, random_seed: Optional[int] = None):
        """
        Initialize basis model.
        
        Args:
            initial_basis: Initial basis (perp - spot or LST - ETH)
            random_seed: Random seed
        """
        self.initial_basis = initial_basis
        self.rng = np.random.default_rng(random_seed)
    
    def generate_path(self, n_steps: int, dt: float = 1/24) -> BasisScenario:
        """Generate basis path."""
        raise NotImplementedError


class MeanRevertingBasisModel(BasisModel):
    """
    Mean-reverting basis model with jumps.
    
    Basis typically mean-reverts to zero (arbitrage), but can experience
    sudden blowouts during stress or illiquidity.
    
    db = κ(θ - b)dt + σ dW + J dN
    """
    
    def __init__(
        self,
        initial_basis: float = 0.0,
        mean_basis: float = 0.0,  # Long-term mean basis
        reversion_speed: float = 5.0,  # Fast mean reversion
        volatility: float = 0.10,  # Basis volatility
        jump_intensity: float = 0.02,  # Expected jumps per year
        jump_mean: float = 50.0,  # Mean jump size (absolute)
        jump_std: float = 30.0,  # Jump size volatility
        random_seed: Optional[int] = None
    ):
        """
        Initialize mean-reverting basis model.
        
        Args:
            initial_basis: Starting basis
            mean_basis: Long-term equilibrium basis
            reversion_speed: Mean reversion speed (κ)
            volatility: Continuous volatility (σ)
            jump_intensity: Expected jumps per year (λ)
            jump_mean: Mean jump size
            jump_std: Jump size std
            random_seed: Random seed
        """
        super().__init__(initial_basis, random_seed)
        self.mean_basis = mean_basis
        self.reversion_speed = reversion_speed
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
    
    def generate_path(self, n_steps: int, dt: float = 1/24) -> BasisScenario:
        """Generate mean-reverting basis path with jumps."""
        timestamps = np.arange(n_steps + 1) * dt
        basis = np.zeros(n_steps + 1)
        basis[0] = self.initial_basis
        
        for i in range(n_steps):
            # Mean reversion component
            dW = self.rng.normal(0, np.sqrt(dt))
            db = (self.reversion_speed * (self.mean_basis - basis[i]) * dt +
                  self.volatility * dW)
            
            # Jump component (basis blowouts)
            jump_intensity_per_step = self.jump_intensity * dt
            if self.rng.random() < jump_intensity_per_step:
                jump = self.rng.normal(self.jump_mean, self.jump_std)
                db += jump
            
            basis[i + 1] = basis[i] + db
        
        return BasisScenario(
            timestamps=timestamps,
            basis=basis
        )


class StressCorrelatedBasisModel(BasisModel):
    """
    Basis model that widens during price stress.
    
    When price drops sharply, basis tends to blow out as liquidity dries up
    and arbitrageurs pull back.
    """
    
    def __init__(
        self,
        initial_basis: float = 0.0,
        mean_basis: float = 0.0,
        base_volatility: float = 0.05,
        stress_multiplier: float = 5.0,  # Vol increase during stress
        stress_threshold: float = -0.10,  # Price drop that triggers stress
        random_seed: Optional[int] = None
    ):
        """
        Initialize stress-correlated basis model.
        
        Args:
            initial_basis: Starting basis
            mean_basis: Long-term mean
            base_volatility: Normal volatility
            stress_multiplier: Volatility multiplier during stress
            stress_threshold: Price return threshold for stress
            random_seed: Random seed
        """
        super().__init__(initial_basis, random_seed)
        self.mean_basis = mean_basis
        self.base_volatility = base_volatility
        self.stress_multiplier = stress_multiplier
        self.stress_threshold = stress_threshold
    
    def generate_path(
        self,
        n_steps: int,
        dt: float = 1/24,
        price_returns: Optional[np.ndarray] = None
    ) -> BasisScenario:
        """
        Generate basis path correlated with price stress.
        
        Args:
            n_steps: Number of steps
            dt: Time step
            price_returns: Optional price returns to correlate with
        
        Returns:
            BasisScenario
        """
        timestamps = np.arange(n_steps + 1) * dt
        basis = np.zeros(n_steps + 1)
        basis[0] = self.initial_basis
        
        for i in range(n_steps):
            # Check if in stress regime
            is_stress = False
            if price_returns is not None and i < len(price_returns):
                is_stress = price_returns[i] < self.stress_threshold
            
            # Adjust volatility based on stress
            vol = (self.base_volatility * self.stress_multiplier if is_stress 
                   else self.base_volatility)
            
            # Mean reversion
            reversion_speed = 2.0 if not is_stress else 0.5  # Slower in stress
            dW = self.rng.normal(0, np.sqrt(dt))
            db = (reversion_speed * (self.mean_basis - basis[i]) * dt + vol * dW)
            
            # Additional positive drift in stress (basis widens)
            if is_stress:
                db += 0.5 * dt  # Drift towards wider basis
            
            basis[i + 1] = basis[i] + db
        
        return BasisScenario(
            timestamps=timestamps,
            basis=basis
        )

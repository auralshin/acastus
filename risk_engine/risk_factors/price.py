"""Price risk factor modeling."""

from typing import Optional, List
import numpy as np
from dataclasses import dataclass


@dataclass
class PriceScenario:
    """A single price scenario/path."""
    
    timestamps: np.ndarray  # Time points
    prices: np.ndarray  # Price values
    returns: Optional[np.ndarray] = None  # Log returns
    
    def __post_init__(self) -> None:
        if self.returns is None and len(self.prices) > 1:
            self.returns = np.diff(np.log(self.prices))


class PriceModel:
    """Base class for price evolution models."""
    
    def __init__(self, initial_price: float, random_seed: Optional[int] = None):
        """
        Initialize price model.
        
        Args:
            initial_price: Starting price
            random_seed: Random seed for reproducibility
        """
        self.initial_price = initial_price
        self.rng = np.random.default_rng(random_seed)
    
    def generate_path(
        self,
        n_steps: int,
        dt: float = 1/24  # Default: hourly (1/24 day)
    ) -> PriceScenario:
        """
        Generate a single price path.
        
        Args:
            n_steps: Number of time steps
            dt: Time step size in days
        
        Returns:
            PriceScenario
        """
        raise NotImplementedError
    
    def generate_scenarios(
        self,
        n_scenarios: int,
        n_steps: int,
        dt: float = 1/24
    ) -> List[PriceScenario]:
        """
        Generate multiple price scenarios.
        
        Args:
            n_scenarios: Number of scenarios
            n_steps: Steps per scenario
            dt: Time step size
        
        Returns:
            List of scenarios
        """
        return [self.generate_path(n_steps, dt) for _ in range(n_scenarios)]


class GBMPriceModel(PriceModel):
    """
    Geometric Brownian Motion price model.
    
    dS/S = μ dt + σ dW
    
    Classic model for asset prices with constant drift and volatility.
    """
    
    def __init__(
        self,
        initial_price: float,
        drift: float = 0.0,  # Annual drift
        volatility: float = 0.75,  # Annual volatility
        random_seed: Optional[int] = None
    ):
        """
        Initialize GBM model.
        
        Args:
            initial_price: Starting price
            drift: Annual drift (μ)
            volatility: Annual volatility (σ)
            random_seed: Random seed
        """
        super().__init__(initial_price, random_seed)
        self.drift = drift
        self.volatility = volatility
    
    def generate_path(self, n_steps: int, dt: float = 1/24) -> PriceScenario:
        """Generate GBM price path."""
        timestamps = np.arange(n_steps + 1) * dt
        
        # Generate random shocks
        dW = self.rng.normal(0, np.sqrt(dt), n_steps)
        
        # Calculate log returns
        log_returns = (self.drift - 0.5 * self.volatility**2) * dt + self.volatility * dW
        
        # Calculate prices
        log_prices = np.concatenate([[np.log(self.initial_price)], 
                                     np.log(self.initial_price) + np.cumsum(log_returns)])
        prices = np.exp(log_prices)
        
        return PriceScenario(
            timestamps=timestamps,
            prices=prices,
            returns=log_returns
        )


class HistoricalBootstrapModel(PriceModel):
    """
    Historical bootstrap model.
    
    Resamples returns from historical data to generate scenarios.
    Preserves empirical distribution including fat tails.
    """
    
    def __init__(
        self,
        initial_price: float,
        historical_returns: np.ndarray,
        random_seed: Optional[int] = None
    ):
        """
        Initialize bootstrap model.
        
        Args:
            initial_price: Starting price
            historical_returns: Historical return series
            random_seed: Random seed
        """
        super().__init__(initial_price, random_seed)
        self.historical_returns = np.asarray(historical_returns)
    
    def generate_path(self, n_steps: int, dt: float = 1/24) -> PriceScenario:
        """Generate path by bootstrapping historical returns."""
        timestamps = np.arange(n_steps + 1) * dt
        
        # Bootstrap returns from historical data
        indices = self.rng.integers(0, len(self.historical_returns), size=n_steps)
        resampled_returns = self.historical_returns[indices]
        
        # Calculate prices
        log_prices = np.concatenate([[np.log(self.initial_price)],
                                     np.log(self.initial_price) + np.cumsum(resampled_returns)])
        prices = np.exp(log_prices)
        
        return PriceScenario(
            timestamps=timestamps,
            prices=prices,
            returns=resampled_returns
        )


class JumpDiffusionModel(PriceModel):
    """
    Jump-diffusion price model (Merton).
    
    Combines continuous GBM with discrete jumps.
    Good for modeling crash risk and tail events.
    
    dS/S = μ dt + σ dW + J dN
    
    where dN is a Poisson process and J is jump size.
    """
    
    def __init__(
        self,
        initial_price: float,
        drift: float = 0.0,
        volatility: float = 0.75,
        jump_intensity: float = 0.1,  # Expected jumps per year
        jump_mean: float = -0.1,  # Mean jump size (log)
        jump_std: float = 0.2,  # Jump size volatility
        random_seed: Optional[int] = None
    ):
        """
        Initialize jump-diffusion model.
        
        Args:
            initial_price: Starting price
            drift: Continuous drift
            volatility: Continuous volatility
            jump_intensity: Expected number of jumps per year (λ)
            jump_mean: Mean of jump size (log scale)
            jump_std: Std of jump size
            random_seed: Random seed
        """
        super().__init__(initial_price, random_seed)
        self.drift = drift
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
    
    def generate_path(self, n_steps: int, dt: float = 1/24) -> PriceScenario:
        """Generate jump-diffusion price path."""
        timestamps = np.arange(n_steps + 1) * dt
        
        # Continuous component (GBM)
        dW = self.rng.normal(0, np.sqrt(dt), n_steps)
        continuous_returns = (self.drift - 0.5 * self.volatility**2) * dt + self.volatility * dW
        
        # Jump component
        jump_intensity_per_step = self.jump_intensity * dt
        n_jumps = self.rng.poisson(jump_intensity_per_step, n_steps)
        
        jump_returns = np.zeros(n_steps)
        for i in range(n_steps):
            if n_jumps[i] > 0:
                # Sum of multiple jumps if they occur
                jumps = self.rng.normal(self.jump_mean, self.jump_std, n_jumps[i])
                jump_returns[i] = np.sum(jumps)
        
        # Combined returns
        log_returns = continuous_returns + jump_returns
        
        # Calculate prices
        log_prices = np.concatenate([[np.log(self.initial_price)],
                                     np.log(self.initial_price) + np.cumsum(log_returns)])
        prices = np.exp(log_prices)
        
        return PriceScenario(
            timestamps=timestamps,
            prices=prices,
            returns=log_returns
        )

from typing import Optional, List
import numpy as np
from dataclasses import dataclass
from enum import Enum


class FundingRegime(Enum):
    """Funding rate regime."""
    POSITIVE = "positive"  # Longs pay shorts (contango)
    NEGATIVE = "negative"  # Shorts pay longs (backwardation)
    NEUTRAL = "neutral"  # Near zero


@dataclass
class FundingScenario:
    """Funding rate scenario/path."""
    
    timestamps: np.ndarray
    rates: np.ndarray  # Annualized funding rates
    regime: Optional[List[FundingRegime]] = None


class FundingModel:
    """Base class for funding rate models."""
    
    def __init__(self, initial_rate: float = 0.0, random_seed: Optional[int] = None):
        """
        Initialize funding model.
        
        Args:
            initial_rate: Initial annualized funding rate
            random_seed: Random seed
        """
        self.initial_rate = initial_rate
        self.rng = np.random.default_rng(random_seed)
    
    def generate_path(self, n_steps: int, dt: float = 1/24) -> FundingScenario:
        """Generate funding rate path."""
        raise NotImplementedError


class ConstantFundingModel(FundingModel):
    """Simple model with constant funding rate."""
    
    def generate_path(self, n_steps: int, dt: float = 1/24) -> FundingScenario:
        """Generate constant funding path."""
        timestamps = np.arange(n_steps + 1) * dt
        rates = np.full(n_steps + 1, self.initial_rate)
        
        return FundingScenario(
            timestamps=timestamps,
            rates=rates
        )


class MeanRevertingFundingModel(FundingModel):
    """
    Mean-reverting funding rate model (Ornstein-Uhlenbeck).
    
    dr = κ(θ - r)dt + σ dW
    
    Funding tends to revert to a long-term mean, with volatility.
    """
    
    def __init__(
        self,
        initial_rate: float = 0.0,
        mean_rate: float = 0.05,  # Long-term mean (annualized)
        reversion_speed: float = 2.0,  # Speed of mean reversion (κ)
        volatility: float = 0.30,  # Funding volatility (σ)
        random_seed: Optional[int] = None
    ):
        """
        Initialize mean-reverting funding model.
        
        Args:
            initial_rate: Starting funding rate
            mean_rate: Long-term mean rate
            reversion_speed: Mean reversion speed (κ)
            volatility: Funding rate volatility (σ)
            random_seed: Random seed
        """
        super().__init__(initial_rate, random_seed)
        self.mean_rate = mean_rate
        self.reversion_speed = reversion_speed
        self.volatility = volatility
    
    def generate_path(self, n_steps: int, dt: float = 1/24) -> FundingScenario:
        """Generate mean-reverting funding path."""
        timestamps = np.arange(n_steps + 1) * dt
        rates = np.zeros(n_steps + 1)
        rates[0] = self.initial_rate
        
        for i in range(n_steps):
            dW = self.rng.normal(0, np.sqrt(dt))
            dr = (self.reversion_speed * (self.mean_rate - rates[i]) * dt +
                  self.volatility * dW)
            rates[i + 1] = rates[i] + dr
        
        return FundingScenario(
            timestamps=timestamps,
            rates=rates
        )


class RegimeSwitchingFundingModel(FundingModel):
    """
    Regime-switching funding model.
    
    Switches between positive (contango), negative (backwardation), and neutral regimes.
    Each regime has different mean and volatility.
    """
    
    def __init__(
        self,
        initial_rate: float = 0.0,
        regime_params: Optional[dict] = None,
        transition_prob: float = 0.05,  # Daily probability of regime change
        random_seed: Optional[int] = None
    ):
        """
        Initialize regime-switching model.
        
        Args:
            initial_rate: Starting funding rate
            regime_params: Dict of regime -> (mean, vol) parameters
            transition_prob: Probability of switching regimes per day
            random_seed: Random seed
        """
        super().__init__(initial_rate, random_seed)
        
        if regime_params is None:
            # Default regime parameters (mean, volatility)
            self.regime_params = {
                FundingRegime.POSITIVE: (0.15, 0.10),  # High positive funding
                FundingRegime.NEUTRAL: (0.03, 0.05),  # Near zero
                FundingRegime.NEGATIVE: (-0.10, 0.15),  # Negative funding
            }
        else:
            self.regime_params = regime_params
        
        self.transition_prob = transition_prob
        
        # Determine initial regime
        if initial_rate > 0.08:
            self.current_regime = FundingRegime.POSITIVE
        elif initial_rate < -0.05:
            self.current_regime = FundingRegime.NEGATIVE
        else:
            self.current_regime = FundingRegime.NEUTRAL
    
    def generate_path(self, n_steps: int, dt: float = 1/24) -> FundingScenario:
        """Generate regime-switching funding path."""
        timestamps = np.arange(n_steps + 1) * dt
        rates = np.zeros(n_steps + 1)
        regimes = []
        
        rates[0] = self.initial_rate
        current_regime = self.current_regime
        regimes.append(current_regime)
        
        for i in range(n_steps):
            # Check for regime switch
            if self.rng.random() < self.transition_prob * dt:
                # Switch to a different regime
                other_regimes = [r for r in FundingRegime if r != current_regime]
                current_regime = self.rng.choice(np.array(other_regimes))
            
            regimes.append(current_regime)
            
            # Get regime parameters
            mean_rate, vol = self.regime_params[current_regime]
            
            # Mean-reverting dynamics within regime
            reversion_speed = 3.0
            dW = self.rng.normal(0, np.sqrt(dt))
            dr = (reversion_speed * (mean_rate - rates[i]) * dt + vol * dW)
            rates[i + 1] = rates[i] + dr
        
        return FundingScenario(
            timestamps=timestamps,
            rates=rates,
            regime=regimes
        )


class HistoricalFundingModel(FundingModel):
    """Bootstrap funding rates from historical data."""
    
    def __init__(
        self,
        initial_rate: float,
        historical_rates: np.ndarray,
        random_seed: Optional[int] = None
    ):
        """
        Initialize historical funding model.
        
        Args:
            initial_rate: Starting rate
            historical_rates: Historical funding rate series
            random_seed: Random seed
        """
        super().__init__(initial_rate, random_seed)
        self.historical_rates = np.asarray(historical_rates)
    
    def generate_path(self, n_steps: int, dt: float = 1/24) -> FundingScenario:
        """Generate funding path by bootstrapping."""
        timestamps = np.arange(n_steps + 1) * dt
        
        # Bootstrap rates
        indices = self.rng.integers(0, len(self.historical_rates), size=n_steps)
        rates = np.concatenate([[self.initial_rate], self.historical_rates[indices]])
        
        return FundingScenario(
            timestamps=timestamps,
            rates=rates
        )

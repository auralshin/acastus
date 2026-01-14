"""Scenario path generation for Monte Carlo simulation."""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from risk_engine.risk_factors.basis import (
    MeanRevertingBasisModel,
    StressCorrelatedBasisModel,
)

try:
    from risk_engine.utils.jit_functions import calculate_gbm_paths, calculate_ou_paths
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False
    # Define fallback functions
    calculate_gbm_paths = None
    calculate_ou_paths = None


@dataclass
class ScenarioPath:
    """A single scenario path for simulation."""

    timestamps: List[datetime]
    spot_prices: Dict[str, np.ndarray]  # symbol -> price path
    funding_rates: Dict[str, np.ndarray]  # symbol -> funding path
    volatilities: Optional[Dict[str, np.ndarray]] = None
    basis_spreads: Optional[Dict[str, np.ndarray]] = None  # symbol -> perp-spot basis
    liquidity_multipliers: Optional[Dict[str, np.ndarray]] = None  # symbol -> liquidity factor

    def __len__(self) -> int:
        """Number of time steps."""
        return len(self.timestamps)


class PathGenerator:
    """
    Monte Carlo path generator for risk factors.

    Generates correlated paths for:
    - Spot prices (GBM with jumps)
    - Funding rates (mean-reverting with regime switches)
    - Basis (mean-reverting)
    - Volatility (GARCH-like)
    - Liquidity (stress-dependent multiplier)
    """

    def __init__(
        self,
        symbols: List[str],
        seed: Optional[int] = None,
        correlation_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize path generator.

        Args:
            symbols: List of symbols to generate paths for
            seed: Random seed for reproducibility
            correlation_matrix: Correlation between symbols
        """
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.rng = np.random.default_rng(seed)

        # Default: moderate positive correlation
        if correlation_matrix is None:
            self.correlation_matrix = np.eye(self.n_assets) * 0.7 + 0.3
            np.fill_diagonal(self.correlation_matrix, 1.0)
        else:
            self.correlation_matrix = correlation_matrix

        # Cholesky decomposition for correlated normals
        self.cholesky = np.linalg.cholesky(self.correlation_matrix)

    def generate_price_paths(
        self,
        initial_prices: Dict[str, float],
        volatilities: Dict[str, float],
        drift: Dict[str, float],
        n_steps: int,
        dt: float,
        jump_intensity: float = 0.01,
        jump_mean: float = -0.02,
        jump_std: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Generate price paths using GBM with jumps.

        Uses JIT-compiled functions when available for 10-50x speedup.

        Args:
            initial_prices: Starting prices for each symbol
            volatilities: Annualized volatilities
            drift: Expected drift (typically 0 for risk-neutral)
            n_steps: Number of time steps
            dt: Time step in years (e.g., 1/365/24 for hourly)
            jump_intensity: Poisson jump intensity
            jump_mean: Mean jump size (log-normal)
            jump_std: Jump size std dev

        Returns:
            Dictionary of symbol -> price path array
        """
        paths = {}

        for i, symbol in enumerate(self.symbols):
            S0 = initial_prices[symbol]
            vol = volatilities[symbol]
            mu = drift[symbol]

            if JIT_AVAILABLE and calculate_gbm_paths is not None and jump_intensity == 0:
                # Use JIT-compiled GBM for maximum speed (no jumps)
                random_normals = self.rng.standard_normal(n_steps)
                path = calculate_gbm_paths(
                    n_paths=1,
                    n_steps=n_steps,
                    S0=S0,
                    mu=mu,
                    sigma=vol,
                    dt=dt,
                    random_normals=random_normals.reshape(1, -1)
                )[0]  # Extract single path
                paths[symbol] = path
            else:
                # Fallback to Python implementation or when jumps are needed
                # Generate correlated normal shocks
                Z = self.rng.standard_normal((n_steps, self.n_assets))
                correlated_Z = Z @ self.cholesky.T

                # GBM component
                returns = (mu - 0.5 * vol**2) * dt + vol * \
                    np.sqrt(dt) * correlated_Z[:, i]

                # Add jumps (Merton jump diffusion)
                if jump_intensity > 0:
                    n_jumps = self.rng.poisson(jump_intensity * dt * n_steps)
                    if n_jumps > 0:
                        jump_times = self.rng.choice(
                            n_steps, size=n_jumps, replace=False)
                        jump_sizes = self.rng.normal(
                            jump_mean, jump_std, size=n_jumps)
                        returns[jump_times] += jump_sizes

                # Cumulative returns to prices
                price_path = S0 * np.exp(np.cumsum(returns))
                price_path = np.concatenate(
                    [[S0], price_path])  # Include initial price

                paths[symbol] = price_path

        return paths

    def generate_funding_paths(
        self,
        initial_rates: Dict[str, float],
        mean_rates: Dict[str, float],
        rate_volatilities: Dict[str, float],
        mean_reversion_speed: float,
        n_steps: int,
        dt: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate funding rate paths using mean-reverting process.

        Uses Ornstein-Uhlenbeck process:
        dr = kappa * (theta - r) * dt + sigma * dW

        Args:
            initial_rates: Starting funding rates (annualized)
            mean_rates: Long-term mean rates
            rate_volatilities: Rate volatilities
            mean_reversion_speed: Speed of mean reversion (kappa)
            n_steps: Number of time steps
            dt: Time step in years

        Returns:
            Dictionary of symbol -> funding rate path
        """
        paths = {}

        for symbol in self.symbols:
            r0 = initial_rates.get(symbol, 0.0)
            theta = mean_rates.get(symbol, 0.1)  # Default 10% annual
            sigma = rate_volatilities.get(symbol, 0.5)
            kappa = mean_reversion_speed

            # OU process
            rates = np.zeros(n_steps + 1)
            rates[0] = r0

            for t in range(n_steps):
                dW = self.rng.standard_normal()
                dr = kappa * (theta - rates[t]) * dt + sigma * np.sqrt(dt) * dW
                rates[t + 1] = rates[t] + dr

                # Cap funding rates at reasonable bounds
                # -200% to 300% annual
                rates[t + 1] = np.clip(rates[t + 1], -2.0, 3.0)

            paths[symbol] = rates

        return paths

    def generate_volatility_paths(
        self,
        base_volatilities: Dict[str, float],
        price_paths: Dict[str, np.ndarray],
        n_steps: int,
        dt: float,
        params: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate volatility paths with mean reversion and stress spikes.

        Args:
            base_volatilities: Baseline annualized volatility per symbol
            price_paths: Spot price paths (used for stress detection)
            n_steps: Number of steps
            dt: Time step in years
            params: Optional parameters for vol dynamics

        Returns:
            Dictionary of symbol -> volatility path
        """
        params = params or {}
        kappa = params.get("mean_reversion_speed", 2.0)
        vol_of_vol = params.get("vol_of_vol", 0.0)
        stress_threshold = params.get("stress_threshold", -0.03)
        stress_multiplier = params.get("stress_multiplier", 1.0)
        min_vol = params.get("min_vol", 0.05)
        max_vol = params.get("max_vol", 3.0)

        vol_paths = {}

        for symbol in self.symbols:
            base_vol = max(base_volatilities.get(symbol, 0.5), 1e-6)
            prices = price_paths[symbol]
            returns = np.diff(np.log(prices))

            vols = np.zeros(n_steps + 1)
            vols[0] = base_vol

            for t in range(n_steps):
                dW = self.rng.standard_normal()
                vol_next = (
                    vols[t]
                    + kappa * (base_vol - vols[t]) * dt
                    + vol_of_vol * np.sqrt(dt) * dW
                )
                vol_next = abs(vol_next)

                if returns[t] < stress_threshold:
                    vol_next *= stress_multiplier

                vols[t + 1] = float(np.clip(vol_next, min_vol, max_vol))

            vol_paths[symbol] = vols

        return vol_paths

    def generate_basis_paths(
        self,
        price_paths: Dict[str, np.ndarray],
        n_steps: int,
        dt: float,
        basis_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate basis (perp - spot) paths per symbol.

        Args:
            price_paths: Spot price paths
            n_steps: Number of steps
            dt: Time step in years
            basis_params: Optional parameters for basis dynamics

        Returns:
            Dictionary of symbol -> basis spread path (absolute)
        """
        params = basis_params or {}
        use_pct = params.get("basis_is_pct", True)

        mean_basis = params.get("mean_basis", 0.0)
        reversion_speed = params.get("reversion_speed", 5.0)
        jump_intensity = params.get("jump_intensity", 0.0)
        jump_mean = params.get("jump_mean", 0.0)
        jump_std = params.get("jump_std", 0.0)

        volatility = params.get("volatility", None)
        volatility_pct = params.get("volatility_pct", None)
        if use_pct:
            vol = volatility_pct if volatility_pct is not None else (volatility or 0.0)
        else:
            vol = volatility if volatility is not None else 0.0

        correlation_breakdown = params.get("correlation_breakdown", 0.0)
        stress_threshold = params.get("stress_threshold", -0.03)
        stress_multiplier = params.get("stress_multiplier", 1.0 + correlation_breakdown * 5.0)
        correlated_with_price = params.get(
            "correlated_with_price",
            correlation_breakdown > 0.0
        )

        basis_paths = {}

        for symbol in self.symbols:
            seed = int(self.rng.integers(0, 2**32 - 1))
            prices = price_paths[symbol]
            returns = np.diff(np.log(prices))

            if correlated_with_price:
                model = StressCorrelatedBasisModel(
                    initial_basis=0.0,
                    mean_basis=mean_basis,
                    base_volatility=vol,
                    stress_multiplier=stress_multiplier,
                    stress_threshold=stress_threshold,
                    random_seed=seed
                )
                basis_path = model.generate_path(
                    n_steps=n_steps,
                    dt=dt,
                    price_returns=returns
                ).basis
            else:
                model = MeanRevertingBasisModel(
                    initial_basis=0.0,
                    mean_basis=mean_basis,
                    reversion_speed=reversion_speed,
                    volatility=vol,
                    jump_intensity=jump_intensity,
                    jump_mean=jump_mean,
                    jump_std=jump_std,
                    random_seed=seed
                )
                basis_path = model.generate_path(n_steps=n_steps, dt=dt).basis

            if use_pct:
                basis_paths[symbol] = basis_path * prices
            else:
                basis_paths[symbol] = basis_path

        return basis_paths

    def generate_liquidity_paths(
        self,
        price_paths: Dict[str, np.ndarray],
        vol_paths: Dict[str, np.ndarray],
        base_volatilities: Dict[str, float],
        n_steps: int,
        params: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate liquidity multipliers driven by volatility and stress.

        Args:
            price_paths: Spot price paths
            vol_paths: Volatility paths
            base_volatilities: Baseline volatilities per symbol
            n_steps: Number of steps
            params: Optional liquidity parameters

        Returns:
            Dictionary of symbol -> liquidity multiplier path
        """
        params = params or {}
        base_multiplier = params.get("base_multiplier", 1.0)
        vol_sensitivity = params.get("vol_sensitivity", 0.0)
        stress_threshold = params.get("stress_threshold", -0.03)
        stress_multiplier = params.get("stress_multiplier", 1.0)

        liquidity_paths = {}

        for symbol in self.symbols:
            base_vol = max(base_volatilities.get(symbol, 0.5), 1e-6)
            prices = price_paths[symbol]
            returns = np.diff(np.log(prices))
            vols = vol_paths[symbol]

            multipliers = np.zeros(n_steps + 1)
            multipliers[0] = base_multiplier

            for t in range(n_steps):
                vol_ratio = vols[t] / base_vol
                multiplier = base_multiplier + max(0.0, vol_ratio - 1.0) * vol_sensitivity

                if returns[t] < stress_threshold:
                    multiplier *= stress_multiplier

                multipliers[t + 1] = multiplier

            liquidity_paths[symbol] = multipliers

        return liquidity_paths

    def generate_scenario(
        self,
        start_time: datetime,
        n_steps: int,
        time_step_hours: float,
        initial_prices: Dict[str, float],
        initial_rates: Dict[str, float],
        price_volatilities: Dict[str, float],
        rate_params: Optional[Dict[str, any]] = None,
        price_params: Optional[Dict[str, any]] = None,
        basis_params: Optional[Dict[str, float]] = None,
        volatility_params: Optional[Dict[str, float]] = None,
        liquidity_params: Optional[Dict[str, float]] = None
    ) -> ScenarioPath:
        """
        Generate a complete scenario path.

        Args:
            start_time: Starting timestamp
            n_steps: Number of steps
            time_step_hours: Hours per step
            initial_prices: Initial spot prices
            initial_rates: Initial funding rates
            price_volatilities: Price volatilities
            rate_params: Parameters for rate generation
            price_params: Parameters for price path generation
            basis_params: Parameters for basis dynamics
            volatility_params: Parameters for volatility dynamics
            liquidity_params: Parameters for liquidity dynamics

        Returns:
            ScenarioPath
        """
        dt = time_step_hours / (365.25 * 24)  # Convert to years

        # Generate timestamps
        timestamps = [
            start_time + timedelta(hours=i * time_step_hours)
            for i in range(n_steps + 1)
        ]

        # Generate price paths
        price_params = price_params or {}
        drift_value = price_params.get("drift", 0.0)
        if isinstance(drift_value, dict):
            drift = drift_value
        else:
            drift = {s: drift_value for s in self.symbols}

        price_kwargs = {}
        for key in ("jump_intensity", "jump_mean", "jump_std"):
            if key in price_params:
                price_kwargs[key] = price_params[key]

        price_paths = self.generate_price_paths(
            initial_prices,
            price_volatilities,
            drift,
            n_steps,
            dt,
            **price_kwargs
        )

        # Generate funding rate paths
        if rate_params is None:
            rate_params = {
                "mean_rates": {s: 0.1 for s in self.symbols},
                "rate_volatilities": {s: 0.5 for s in self.symbols},
                "mean_reversion_speed": 2.0
            }

        funding_paths = self.generate_funding_paths(
            initial_rates,
            rate_params["mean_rates"],
            rate_params["rate_volatilities"],
            rate_params["mean_reversion_speed"],
            n_steps,
            dt
        )

        # Generate volatility and liquidity paths
        vol_paths = self.generate_volatility_paths(
            base_volatilities=price_volatilities,
            price_paths=price_paths,
            n_steps=n_steps,
            dt=dt,
            params=volatility_params
        )

        liquidity_paths = self.generate_liquidity_paths(
            price_paths=price_paths,
            vol_paths=vol_paths,
            base_volatilities=price_volatilities,
            n_steps=n_steps,
            params=liquidity_params
        )

        # Generate basis paths
        basis_paths = self.generate_basis_paths(
            price_paths=price_paths,
            n_steps=n_steps,
            dt=dt,
            basis_params=basis_params
        )

        return ScenarioPath(
            timestamps=timestamps,
            spot_prices=price_paths,
            funding_rates=funding_paths,
            volatilities=vol_paths,
            basis_spreads=basis_paths,
            liquidity_multipliers=liquidity_paths
        )

    def generate_multiple_scenarios(
        self,
        n_scenarios: int,
        **kwargs
    ) -> List[ScenarioPath]:
        """
        Generate multiple scenario paths.

        Args:
            n_scenarios: Number of scenarios
            **kwargs: Arguments for generate_scenario

        Returns:
            List of ScenarioPath objects
        """
        scenarios = []

        for _ in range(n_scenarios):
            scenario = self.generate_scenario(**kwargs)
            scenarios.append(scenario)

        return scenarios

"""Optimized numeric computations with (Numba) JIT"""

import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def calculate_gbm_paths(
    n_paths: int,
    n_steps: int,
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    random_normals: np.ndarray
) -> np.ndarray:
    """
    JIT-compiled GBM path generation.
    
    Args:
        n_paths: Number of paths
        n_steps: Number of time steps
        S0: Initial price
        mu: Drift
        sigma: Volatility
        dt: Time step
        random_normals: Pre-generated random normals (n_paths, n_steps)
    
    Returns:
        Price paths array (n_paths, n_steps + 1)
    """
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    for i in prange(n_paths):
        for t in range(n_steps):
            paths[i, t + 1] = paths[i, t] * np.exp(
                drift + diffusion * random_normals[i, t]
            )
    
    return paths


@jit(nopython=True, parallel=True)
def calculate_ou_paths(
    n_paths: int,
    n_steps: int,
    r0: float,
    theta: float,
    kappa: float,
    sigma: float,
    dt: float,
    random_normals: np.ndarray
) -> np.ndarray:
    """
    JIT-compiled Ornstein-Uhlenbeck process paths.
    
    Args:
        n_paths: Number of paths
        n_steps: Number of time steps
        r0: Initial rate
        theta: Long-term mean
        kappa: Mean reversion speed
        sigma: Volatility
        dt: Time step
        random_normals: Pre-generated random normals
    
    Returns:
        Rate paths array (n_paths, n_steps + 1)
    """
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0
    
    for i in prange(n_paths):
        for t in range(n_steps):
            dr = kappa * (theta - paths[i, t]) * dt + sigma * np.sqrt(dt) * random_normals[i, t]
            paths[i, t + 1] = paths[i, t] + dr
            # Clip to reasonable bounds
            paths[i, t + 1] = max(min(paths[i, t + 1], 3.0), -2.0)
    
    return paths


@jit(nopython=True)
def calculate_portfolio_value(
    spot_quantities: np.ndarray,
    perp_quantities: np.ndarray,
    spot_prices: np.ndarray,
    perp_prices: np.ndarray,
    cash: float
) -> float:
    """
    JIT-compiled portfolio valuation.
    
    Args:
        spot_quantities: Array of spot quantities
        perp_quantities: Array of perp quantities
        spot_prices: Array of spot prices
        perp_prices: Array of perp prices
        cash: Cash amount
    
    Returns:
        Total portfolio value
    """
    spot_value = np.sum(spot_quantities * spot_prices)
    perp_value = np.sum(perp_quantities * perp_prices)
    return cash + spot_value + perp_value


@jit(nopython=True)
def calculate_funding_pnl(
    perp_quantities: np.ndarray,
    perp_prices: np.ndarray,
    funding_rates: np.ndarray,
    time_fraction: float
) -> float:
    """
    JIT-compiled funding P&L calculation.
    
    Args:
        perp_quantities: Perp position sizes
        perp_prices: Perp prices
        funding_rates: Funding rates (annualized)
        time_fraction: Time as fraction of funding period
    
    Returns:
        Total funding P&L
    """
    notionals = np.abs(perp_quantities * perp_prices)
    # Longs pay when funding positive, shorts receive
    funding_pnl = -np.sum(perp_quantities * funding_rates * notionals * time_fraction)
    return funding_pnl


@jit(nopython=True, parallel=True)
def calculate_var_from_returns(
    returns: np.ndarray,
    confidence_levels: np.ndarray
) -> np.ndarray:
    """
    JIT-compiled VaR calculation from returns.
    
    Args:
        returns: Array of returns
        confidence_levels: Array of confidence levels (e.g., [0.95, 0.99])
    
    Returns:
        VaR values for each confidence level
    """
    n_levels = len(confidence_levels)
    var_values = np.zeros(n_levels)
    
    for i in range(n_levels):
        quantile = 1 - confidence_levels[i]
        var_values[i] = -np.quantile(returns, quantile)
    
    return var_values


@jit(nopython=True)
def calculate_drawdowns(nav_path: np.ndarray) -> tuple:
    """
    JIT-compiled drawdown calculation.
    
    Args:
        nav_path: NAV time series
    
    Returns:
        Tuple of (drawdowns, max_drawdown, max_drawdown_index)
    """
    n = len(nav_path)
    drawdowns = np.zeros(n)
    running_max = nav_path[0]
    max_dd = 0.0
    max_dd_idx = 0
    
    for i in range(n):
        if nav_path[i] > running_max:
            running_max = nav_path[i]
        
        dd = (nav_path[i] - running_max) / running_max if running_max > 0 else 0.0
        drawdowns[i] = dd
        
        if dd < max_dd:
            max_dd = dd
            max_dd_idx = i
    
    return drawdowns, max_dd, max_dd_idx


@jit(nopython=True, parallel=True)
def monte_carlo_portfolio_evolution(
    n_scenarios: int,
    n_steps: int,
    initial_nav: float,
    spot_quantities: np.ndarray,
    perp_quantities: np.ndarray,
    spot_price_paths: np.ndarray,  # (n_scenarios, n_assets, n_steps+1)
    funding_rate_paths: np.ndarray,  # (n_scenarios, n_assets, n_steps+1)
    funding_period_hours: float,
    time_step_hours: float
) -> tuple:
    """
    JIT-compiled Monte Carlo portfolio evolution.
    
    Returns:
        Tuple of (nav_paths, final_navs, max_drawdowns)
    """
    n_assets = len(spot_quantities)
    nav_paths = np.zeros((n_scenarios, n_steps + 1))
    final_navs = np.zeros(n_scenarios)
    max_drawdowns = np.zeros(n_scenarios)
    
    time_fraction = time_step_hours / funding_period_hours
    
    for scenario in prange(n_scenarios):
        nav_paths[scenario, 0] = initial_nav
        cash = initial_nav - np.sum(spot_quantities * spot_price_paths[scenario, :, 0])
        
        for step in range(n_steps):
            # Calculate spot value
            spot_value = np.sum(spot_quantities * spot_price_paths[scenario, :, step])
            
            # Calculate perp value (unrealized PnL)
            perp_value = np.sum(perp_quantities * spot_price_paths[scenario, :, step])
            
            # Calculate funding P&L
            funding_pnl = 0.0
            for asset in range(n_assets):
                if perp_quantities[asset] != 0:
                    notional = abs(perp_quantities[asset] * spot_price_paths[scenario, asset, step])
                    rate = funding_rate_paths[scenario, asset, step]
                    funding_pnl -= perp_quantities[asset] * rate * notional * time_fraction
            
            cash += funding_pnl
            nav_paths[scenario, step + 1] = cash + spot_value + perp_value
        
        final_navs[scenario] = nav_paths[scenario, -1]
        
        # Calculate drawdown
        running_max = nav_paths[scenario, 0]
        max_dd = 0.0
        for step in range(n_steps + 1):
            if nav_paths[scenario, step] > running_max:
                running_max = nav_paths[scenario, step]
            dd = (nav_paths[scenario, step] - running_max) / running_max if running_max > 0 else 0.0
            if dd < max_dd:
                max_dd = dd
        max_drawdowns[scenario] = abs(max_dd)
    
    return nav_paths, final_navs, max_drawdowns


@jit(nopython=True)
def calculate_margin_ratio(
    equity: float,
    notional: float,
    maintenance_margin_rate: float
) -> float:
    """
    JIT-compiled margin ratio calculation.
    
    Args:
        equity: Portfolio equity
        notional: Total notional exposure
        maintenance_margin_rate: Maintenance margin rate
    
    Returns:
        Margin ratio (equity / required_margin)
    """
    required_margin = notional * maintenance_margin_rate
    if required_margin == 0:
        return np.inf
    return equity / required_margin


@jit(nopython=True)
def check_liquidation_vectorized(
    nav_paths: np.ndarray,
    notionals: np.ndarray,
    maintenance_margin_rate: float
) -> tuple:
    """
    Vectorized liquidation check across scenarios.
    
    Args:
        nav_paths: NAV paths (n_scenarios, n_steps)
        notionals: Notional exposures (n_scenarios, n_steps)
        maintenance_margin_rate: MM rate
    
    Returns:
        Tuple of (is_liquidated, liquidation_step)
    """
    n_scenarios, n_steps = nav_paths.shape
    is_liquidated = np.zeros(n_scenarios, dtype=np.bool_)
    liquidation_step = np.full(n_scenarios, -1, dtype=np.int32)
    
    for i in range(n_scenarios):
        for t in range(n_steps):
            required_margin = notionals[i, t] * maintenance_margin_rate
            if nav_paths[i, t] < required_margin:
                is_liquidated[i] = True
                liquidation_step[i] = t
                break
    
    return is_liquidated, liquidation_step

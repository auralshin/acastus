"""Mathematical utilities for the risk engine."""

from typing import Union, Optional
import numpy as np
from scipy import stats


def calculate_returns(prices: np.ndarray, log_returns: bool = True) -> np.ndarray:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price array
        log_returns: If True, calculate log returns; otherwise simple returns
    
    Returns:
        Return array (length = len(prices) - 1)
    """
    prices = np.asarray(prices)
    if log_returns:
        return np.diff(np.log(prices))
    else:
        return np.diff(prices) / prices[:-1]


def calculate_volatility(
    returns: np.ndarray,
    annualize: bool = True,
    periods_per_year: int = 365 * 24
) -> float:
    """
    Calculate volatility from returns.
    
    Args:
        returns: Return array
        annualize: Whether to annualize the volatility
        periods_per_year: Number of periods per year for annualization
    
    Returns:
        Volatility (standard deviation of returns)
    """
    vol = np.std(returns, ddof=1)
    if annualize:
        vol *= np.sqrt(periods_per_year)
    return vol


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365 * 24
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Return array
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sharpe ratio
    """
    mean_return = np.mean(returns) * periods_per_year
    vol = calculate_volatility(returns, annualize=True, periods_per_year=periods_per_year)
    
    if vol == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / vol


def calculate_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Calculate Value at Risk.
    
    Args:
        returns: Return array
        confidence: Confidence level (e.g., 0.95 for 95% VaR)
        method: 'historical' or 'parametric'
    
    Returns:
        VaR (positive number representing potential loss)
    """
    if method == "historical":
        return -np.percentile(returns, (1 - confidence) * 100)
    elif method == "parametric":
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        return -(mean + stats.norm.ppf(1 - confidence) * std)
    else:
        raise ValueError(f"Unknown VaR method: {method}")


def calculate_expected_shortfall(
    returns: np.ndarray,
    confidence: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    Args:
        returns: Return array
        confidence: Confidence level
    
    Returns:
        Expected Shortfall (positive number representing expected loss in tail)
    """
    var_threshold = -calculate_var(returns, confidence, method="historical")
    tail_losses = returns[returns <= var_threshold]
    
    if len(tail_losses) == 0:
        return calculate_var(returns, confidence, method="historical")
    
    return -np.mean(tail_losses)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Array of equity values over time
    
    Returns:
        Maximum drawdown as a percentage
    """
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    return abs(np.min(drawdown))


def basis_points_to_decimal(bps: float) -> float:
    """Convert basis points to decimal (e.g., 5 bps -> 0.0005)."""
    return bps / 10000.0


def decimal_to_basis_points(decimal: float) -> float:
    """Convert decimal to basis points (e.g., 0.0005 -> 5 bps)."""
    return decimal * 10000.0

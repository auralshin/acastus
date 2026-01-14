"""Time utilities for the risk engine."""

from datetime import datetime, timedelta
from typing import Union
import numpy as np


def timestamp_to_datetime(ts: Union[int, float]) -> datetime:
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(ts)


def datetime_to_timestamp(dt: datetime) -> float:
    """Convert datetime to Unix timestamp."""
    return dt.timestamp()


def generate_date_range(
    start: datetime,
    end: datetime,
    freq: str = "1H"
) -> np.ndarray:
    """
    Generate a date range.
    
    Args:
        start: Start datetime
        end: End datetime
        freq: Frequency ('1H' for hourly, '1D' for daily)
    
    Returns:
        Array of timestamps
    """
    import pandas as pd
    return pd.date_range(start, end, freq=freq).to_numpy()


def hours_between(dt1: datetime, dt2: datetime) -> float:
    """Calculate hours between two datetimes."""
    return abs((dt2 - dt1).total_seconds()) / 3600


def days_between(dt1: datetime, dt2: datetime) -> float:
    """Calculate days between two datetimes."""
    return abs((dt2 - dt1).total_seconds()) / 86400


def annualize_factor(hours: float) -> float:
    """
    Get annualization factor for given time period.
    
    Args:
        hours: Time period in hours
    
    Returns:
        Factor to annualize returns
    """
    hours_per_year = 24 * 365.25
    return np.sqrt(hours_per_year / hours)

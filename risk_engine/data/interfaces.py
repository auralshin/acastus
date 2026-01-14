"""Market data provider interfaces."""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from risk_engine.instruments.base import MarketData


class MarketDataProvider(ABC):
    """Protocol for market data providers."""
    
    @abstractmethod
    def get_market_data(self, timestamp: datetime, symbols: List[str]) -> MarketData:
        """
        Get market data for given timestamp and symbols.
        
        Args:
            timestamp: Time point to get data for
            symbols: List of symbols to get data for
        
        Returns:
            MarketData snapshot
        """
        pass
    
    @abstractmethod
    def get_historical_data(
        self,
        start: datetime,
        end: datetime,
        symbols: List[str],
        frequency: str = "1H"
    ) -> List[MarketData]:
        """
        Get historical market data.
        
        Args:
            start: Start timestamp
            end: End timestamp
            symbols: Symbols to get data for
            frequency: Data frequency (e.g., '1H', '1D')
        
        Returns:
            List of MarketData snapshots
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass


class HistoricalDataProvider(MarketDataProvider):
    """Simple historical data provider using pandas DataFrames."""
    
    def __init__(self):
        """Initialize provider."""
        self.data_cache = {}
    
    def get_market_data(self, timestamp: datetime, symbols: List[str]) -> MarketData:
        """Get market data for specific timestamp."""
        raise NotImplementedError("Load data first using load_from_csv/parquet")
    
    def get_historical_data(
        self,
        start: datetime,
        end: datetime,
        symbols: List[str],
        frequency: str = "1H"
    ) -> List[MarketData]:
        """Get historical market data series."""
        raise NotImplementedError("Load data first using load_from_csv/parquet")
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols."""
        return list(self.data_cache.keys())

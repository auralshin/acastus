from typing import Dict
from risk_engine.instruments.base import Instrument, MarketData, RiskExposure


class SpotAsset(Instrument):
    """
    Spot asset (e.g., ETH, BTC).
    
    Simple instrument with no funding or time decay.
    """
    
    def __init__(self, symbol: str):
        """
        Initialize spot asset.
        
        Args:
            symbol: Asset symbol (e.g., 'ETH', 'BTC')
        """
        super().__init__(symbol)
    
    def mark_to_market(self, market: MarketData, quantity: float) -> float:
        """
        MTM value = quantity * spot_price.
        
        Args:
            market: Current market data
            quantity: Position size
        
        Returns:
            Current value
        """
        price = market.get_spot_price(self.symbol)
        return quantity * price
    
    def calculate_pnl(
        self,
        market_t0: MarketData,
        market_t1: MarketData,
        quantity: float,
        entry_price: float
    ) -> Dict[str, float]:
        """
        Calculate spot P&L.
        
        Returns:
            Dictionary with 'spot_pnl' component
        """
        price_t0 = market_t0.get_spot_price(self.symbol)
        price_t1 = market_t1.get_spot_price(self.symbol)
        
        pnl = quantity * (price_t1 - price_t0)
        
        return {
            "spot_pnl": pnl,
            "total_pnl": pnl
        }
    
    def risk_exposures(
        self,
        market: MarketData,
        quantity: float,
        entry_price: float
    ) -> RiskExposure:
        """
        Spot asset risk exposures for delta-neutral strategy analysis.
        
        Returns:
            RiskExposure focused on delta-neutral strategy risks
        """
        return RiskExposure(
            delta=quantity,  # 1:1 price sensitivity
            basis_delta=0.0,  # No basis risk for spot
            gamma=0.0,  # Linear payoff
            vega=0.0,   # No direct vol sensitivity
            theta=0.0,  # No time decay
            rho=0.0     # No funding sensitivity
        )
    
    def get_current_price(self, market: MarketData) -> float:
        """Get current spot price."""
        return market.get_spot_price(self.symbol)

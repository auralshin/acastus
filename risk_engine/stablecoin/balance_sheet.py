"""Stablecoin balance sheet and solvency modeling."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

from risk_engine.portfolio.positions import Portfolio
from risk_engine.instruments.base import MarketData


@dataclass
class BalanceSheet:
    """
    Stablecoin balance sheet.
    
    Assets = Liabilities + Equity
    """
    
    # Assets
    collateral_portfolio: Portfolio
    reserve_cash: float = 0.0
    insurance_fund: float = 0.0
    
    # Liabilities
    stablecoin_supply: float = 0.0  # Outstanding stablecoin units
    stablecoin_peg: float = 1.0  # Target peg value (USD)
    
    # Equity (protocol/treasury)
    protocol_equity: float = 0.0
    
    def total_assets(self, market: MarketData) -> float:
        """Calculate total asset value."""
        collateral_value = self.collateral_portfolio.net_asset_value(market)
        return collateral_value + self.reserve_cash + self.insurance_fund
    
    def total_liabilities(self) -> float:
        """Calculate total liability value."""
        return self.stablecoin_supply * self.stablecoin_peg
    
    def equity(self, market: MarketData) -> float:
        """Calculate equity (buffer)."""
        return self.total_assets(market) - self.total_liabilities()
    
    def collateral_ratio(self, market: MarketData) -> float:
        """
        Calculate collateralization ratio.
        
        CR = Assets / Liabilities
        """
        liabilities = self.total_liabilities()
        if liabilities == 0:
            return float('inf')
        return self.total_assets(market) / liabilities
    
    def coverage_ratio(self, market: MarketData) -> float:
        """
        Calculate liquid coverage ratio.
        
        Liquid assets / Redeemable liabilities
        """
        liquid_assets = self.reserve_cash + self.insurance_fund
        return liquid_assets / self.total_liabilities() if self.total_liabilities() > 0 else 0.0
    
    def is_solvent(self, market: MarketData, min_ratio: float = 1.0) -> bool:
        """Check if system is solvent."""
        return self.collateral_ratio(market) >= min_ratio


@dataclass
class SolvencyMetrics:
    """Solvency metrics for stablecoin system."""
    
    timestamp: datetime
    collateral_ratio: float
    coverage_ratio: float
    equity_buffer: float
    equity_buffer_pct: float
    
    # Risk metrics
    distance_to_insolvency_pct: float
    breach_probability: float
    
    # Composition
    collateral_value: float
    reserve_cash: float
    insurance_fund: float
    total_assets: float
    total_liabilities: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "collateral_ratio": self.collateral_ratio,
            "coverage_ratio": self.coverage_ratio,
            "equity_buffer": self.equity_buffer,
            "equity_buffer_pct": self.equity_buffer_pct,
            "distance_to_insolvency_pct": self.distance_to_insolvency_pct,
            "breach_probability": self.breach_probability,
            "collateral_value": self.collateral_value,
            "reserve_cash": self.reserve_cash,
            "insurance_fund": self.insurance_fund,
            "total_assets": self.total_assets,
            "total_liabilities": self.total_liabilities,
        }


@dataclass
class StablecoinInvariants:
    """
    Invariants that must hold for stablecoin system.
    
    These are solvency and operational constraints.
    """
    
    min_collateral_ratio: float = 1.05  # 105% minimum
    target_collateral_ratio: float = 1.20  # 120% target
    min_coverage_ratio: float = 0.03  # 3% liquid reserves
    max_leverage: float = 10.0  # Max leverage on collateral
    
    def check_invariants(
        self,
        balance_sheet: BalanceSheet,
        market: MarketData
    ) -> Dict[str, bool]:
        """
        Check if invariants hold.
        
        Args:
            balance_sheet: Current balance sheet
            market: Market data
        
        Returns:
            Dictionary of invariant -> bool (True = holding)
        """
        cr = balance_sheet.collateral_ratio(market)
        cov_ratio = balance_sheet.coverage_ratio(market)
        
        # Calculate leverage
        collateral_value = balance_sheet.collateral_portfolio.net_asset_value(market)
        notional = balance_sheet.collateral_portfolio.total_notional_exposure(market)
        leverage = notional / collateral_value if collateral_value > 0 else 0.0
        
        return {
            "min_collateral_ratio": cr >= self.min_collateral_ratio,
            "target_collateral_ratio": cr >= self.target_collateral_ratio,
            "min_coverage_ratio": cov_ratio >= self.min_coverage_ratio,
            "max_leverage": leverage <= self.max_leverage,
            "positive_equity": balance_sheet.equity(market) > 0,
        }
    
    def all_invariants_hold(
        self,
        balance_sheet: BalanceSheet,
        market: MarketData
    ) -> bool:
        """Check if all critical invariants hold."""
        checks = self.check_invariants(balance_sheet, market)
        
        # Critical invariants
        critical = [
            "min_collateral_ratio",
            "positive_equity",
        ]
        
        return all(checks[inv] for inv in critical)


class StablecoinRiskModel:
    """
    Risk model specifically for delta-neutral stablecoins.
    
    Extends generic risk engine with stablecoin-specific metrics:
    - Collateral ratio monitoring
    - Run risk (redemption cascades)
    - Basis risk (LST vs ETH, perp vs spot)
    - Venue concentration risk
    """
    
    def __init__(
        self,
        balance_sheet: BalanceSheet,
        invariants: Optional[StablecoinInvariants] = None
    ):
        """
        Initialize stablecoin risk model.
        
        Args:
            balance_sheet: Initial balance sheet
            invariants: System invariants
        """
        self.balance_sheet = balance_sheet
        self.invariants = invariants or StablecoinInvariants()
        self.history: List[SolvencyMetrics] = []
    
    def calculate_solvency_metrics(self, market: MarketData) -> SolvencyMetrics:
        """
        Calculate comprehensive solvency metrics.
        
        Args:
            market: Current market data
        
        Returns:
            SolvencyMetrics
        """
        cr = self.balance_sheet.collateral_ratio(market)
        cov_ratio = self.balance_sheet.coverage_ratio(market)
        equity = self.balance_sheet.equity(market)
        total_assets = self.balance_sheet.total_assets(market)
        total_liabilities = self.balance_sheet.total_liabilities()
        
        equity_pct = equity / total_liabilities if total_liabilities > 0 else 0.0
        
        # Distance to insolvency (when CR = 1.0)
        distance_pct = (cr - 1.0) * 100 if cr >= 1.0 else 0.0
        
        # Heuristic breach probability; replace with simulation-based estimate.
        breach_prob = max(0.0, 1.0 - cr) if cr < 1.2 else 0.0
        
        metrics = SolvencyMetrics(
            timestamp=market.timestamp,
            collateral_ratio=cr,
            coverage_ratio=cov_ratio,
            equity_buffer=equity,
            equity_buffer_pct=equity_pct,
            distance_to_insolvency_pct=distance_pct,
            breach_probability=breach_prob,
            collateral_value=self.balance_sheet.collateral_portfolio.net_asset_value(market),
            reserve_cash=self.balance_sheet.reserve_cash,
            insurance_fund=self.balance_sheet.insurance_fund,
            total_assets=total_assets,
            total_liabilities=total_liabilities
        )
        
        self.history.append(metrics)
        return metrics
    
    def check_health(self, market: MarketData) -> Dict[str, any]:
        """
        Comprehensive health check.
        
        Args:
            market: Current market data
        
        Returns:
            Dictionary with health status
        """
        metrics = self.calculate_solvency_metrics(market)
        invariants_check = self.invariants.check_invariants(self.balance_sheet, market)
        
        return {
            "healthy": all(invariants_check.values()),
            "metrics": metrics.to_dict(),
            "invariants": invariants_check,
            "warnings": self._generate_warnings(metrics, invariants_check),
        }
    
    def _generate_warnings(
        self,
        metrics: SolvencyMetrics,
        invariants: Dict[str, bool]
    ) -> List[str]:
        """Generate warning messages."""
        warnings = []
        
        if not invariants["min_collateral_ratio"]:
            warnings.append(f"⚠️ Collateral ratio below minimum: {metrics.collateral_ratio:.2%}")
        
        if not invariants["target_collateral_ratio"]:
            warnings.append(f"⚠️ Collateral ratio below target: {metrics.collateral_ratio:.2%}")
        
        if not invariants["min_coverage_ratio"]:
            warnings.append(f"⚠️ Coverage ratio below minimum: {metrics.coverage_ratio:.2%}")
        
        if not invariants["positive_equity"]:
            warnings.append("🚨 CRITICAL: Negative equity - system insolvent!")
        
        if metrics.breach_probability > 0.05:
            warnings.append(f"⚠️ High breach probability: {metrics.breach_probability:.2%}")
        
        return warnings
    
    def simulate_redemption_stress(
        self,
        market: MarketData,
        redemption_pct: float = 0.20,
        slippage_impact: float = 0.02
    ) -> SolvencyMetrics:
        """
        Simulate a redemption stress scenario.
        
        Args:
            market: Current market
            redemption_pct: Percentage of supply redeemed (e.g., 0.20 = 20% run)
            slippage_impact: Price impact from forced selling
        
        Returns:
            SolvencyMetrics after stress
        """
        # Apply redemption impact to assets and liabilities.
        redemption_amount = self.balance_sheet.stablecoin_supply * redemption_pct
        
        # Estimate unwind cost
        collateral_to_sell = redemption_amount / self.balance_sheet.collateral_ratio(market)
        slippage_cost = collateral_to_sell * slippage_impact
        
        # Stressed state after redemption and slippage.
        stressed_supply = self.balance_sheet.stablecoin_supply - redemption_amount
        stressed_collateral_value = (
            self.balance_sheet.collateral_portfolio.net_asset_value(market)
            - redemption_amount
            - slippage_cost
        )
        
        # Calculate stressed metrics.
        stressed_assets = stressed_collateral_value + self.balance_sheet.reserve_cash
        stressed_liabilities = stressed_supply
        stressed_cr = stressed_assets / stressed_liabilities if stressed_liabilities > 0 else 0.0
        
        return SolvencyMetrics(
            timestamp=market.timestamp,
            collateral_ratio=stressed_cr,
            coverage_ratio=self.balance_sheet.coverage_ratio(market),
            equity_buffer=stressed_assets - stressed_liabilities,
            equity_buffer_pct=(stressed_assets - stressed_liabilities) / stressed_liabilities,
            distance_to_insolvency_pct=(stressed_cr - 1.0) * 100,
            breach_probability=0.0,
            collateral_value=stressed_collateral_value,
            reserve_cash=self.balance_sheet.reserve_cash,
            insurance_fund=self.balance_sheet.insurance_fund,
            total_assets=stressed_assets,
            total_liabilities=stressed_liabilities
        )

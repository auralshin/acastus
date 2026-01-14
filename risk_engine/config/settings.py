"""Configuration management using Pydantic settings."""

from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # General
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Simulation
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    n_scenarios: int = Field(default=10000, description="Number of Monte Carlo scenarios")
    simulation_days: int = Field(default=30, description="Simulation horizon in days")
    
    # Risk parameters
    var_confidence: float = Field(default=0.95, description="VaR confidence level")
    es_confidence: float = Field(default=0.95, description="Expected Shortfall confidence level")
    
    # Hedging defaults
    rebalance_threshold_pct: float = Field(default=0.05, description="Rebalance if delta > 5% notional")
    rebalance_period_hours: int = Field(default=1, description="Minimum hours between rebalances")
    
    # Slippage model
    base_slippage_bps: float = Field(default=5.0, description="Base slippage in basis points")
    volatility_slippage_multiplier: float = Field(default=2.0, description="Slippage increase per vol regime")
    
    # Margin defaults
    initial_margin_pct: float = Field(default=0.10, description="Initial margin requirement")
    maintenance_margin_pct: float = Field(default=0.05, description="Maintenance margin requirement")
    liquidation_fee_pct: float = Field(default=0.005, description="Liquidation fee")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

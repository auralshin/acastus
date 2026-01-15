# Acastus: Risk Engine for Delta-Neutral Strategies

Risk modeling framework for delta-neutral strategies and stablecoin economics.

## Theory

### Delta-Neutral Portfolio Construction

A delta-neutral portfolio maintains zero first-order sensitivity to price movements:
- Long spot: Δ_spot = +Q
- Short derivative: Δ_derivative = -Q  
- Net delta: Δ_portfolio = 0

**Key Properties**: Price independence for small moves, carry exposure via funding/basis, liquidation convexity introduces hidden gamma.

### Greeks for Delta-Neutral Strategies

#### Delta (Δ): `∂V/∂S`
- **Target**: Δ_portfolio ≈ 0 via continuous rebalancing
- **Example**: Long 10 ETH spot (Δ=+10) + Short 10 ETH perp (Δ=-10) = Net Δ=0

#### Gamma (Γ): `∂²V/∂S²`
- Linear instruments have Γ ≈ 0, but liquidation mechanics create effective gamma near margin thresholds

#### Vega (ν): `∂V/∂σ`
- Direct vega ≈ 0 (no options), but volatility affects funding rates, basis, and liquidity costs

#### Theta (Θ): `∂V/∂t`
```
Total_Θ = Funding_Θ + Basis_Θ - Transaction_Cost_Θ
```
Time decay from funding payments and rebalancing costs.

#### Rho (ρ): `∂V/∂r`
Funding rate sensitivity dominates delta-neutral P&L:
```
Funding_ρ = Position_Size × Mark_Price × Time_to_Funding
```

#### Basis Delta: `∂V/∂(Perp - Spot)`
Primary risk when price delta is neutralized; basis changes create P&L even with perfect hedge.

#### Lambda (λ): `∂V/∂(Spread)`
Liquidity sensitivity from rebalancing costs; stressed markets widen spreads when rehedging needed most.

### Key Insights

1. **Risk Shifts**: From directional to basis, funding, and liquidity risks
2. **Hidden Gamma**: Liquidation mechanics create convexity with linear instruments
3. **Funding Dominance**: Rho often drives P&L in delta-neutral strategies
4. **Correlation Risk**: Spot-perp correlation breakdown breaks neutrality
5. **Regime Sensitivity**: Greeks change dramatically across market regimes

### Perpetual Swap Mechanics

**Funding Rate**: Periodic payment anchoring perpetual to spot
```
Funding_Payment = Position_Size × Mark_Price × Rate × dt
```
- Positive: Longs pay shorts (perp at premium)
- Negative: Shorts pay longs (perp at discount)

**Basis Risk**: `Basis = Perp_Price - Spot_Price`  
Driven by funding expectations, liquidity differences, and exchange-specific factors.

### Risk Metrics

**VaR**: `VaR_α = -Quantile(Returns, α)` - Maximum expected loss at confidence level  
**ES**: `ES_α = -E[Returns | Returns < -VaR]` - Average loss beyond VaR threshold  
**Max Drawdown**: Largest peak-to-trough decline  
**Sharpe**: `(Return - RFR) / Volatility` - Risk-adjusted returns (>1 good, >2 excellent)

### Stablecoin Modeling

**Balance Sheet**:
- Assets: Cash reserves, spot collateral, derivative hedges
- Liabilities: Stablecoin supply
- Equity: Insurance fund, protocol reserves

**Key Ratios**:
```
Collateral_Ratio = Total_Assets / Stablecoin_Supply  (>100% = overcollateralized)
Coverage_Ratio = (Cash + Insurance) / Supply  (immediate redemption capacity)
```

**Delta-Neutral Stablecoin**: Hold volatile collateral (ETH/BTC), hedge with short perps (Δ≈0), generate yield from funding arbitrage.

**Trade-offs**: Capital efficient vs. cash reserves, funding income potential vs. liquidation/basis/counterparty risks.

## Installation

```bash
pip install -e ".[dev]"
```

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard provides an interactive interface to:
- Configure delta-neutral trading strategies or stablecoin portfolios
- Run Monte Carlo simulations
- Visualize risk metrics, equity curves, and scenarios
- Analyze VaR, drawdowns, and liquidation risk

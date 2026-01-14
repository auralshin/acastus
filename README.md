# Acastus: Risk Engine for Delta-Neutral Strategies

A comprehensive risk modeling framework for delta-neutral strategies and stablecoin economics.

## Theory

### Delta-Neutral Portfolio Construction

A delta-neutral portfolio maintains zero first-order sensitivity to price movements of the underlying asset. This is achieved by holding offsetting positions in correlated instruments:

**Construction:**
- Long position in spot asset: Δ_spot = +Q (Q = quantity)
- Short position in derivative: Δ_derivative = -Q
- Net delta: Δ_portfolio = Δ_spot + Δ_derivative = 0

**Key Properties:**
1. **Price Independence**: Portfolio value remains approximately constant for small price moves
2. **Carry Exposure**: Funding and basis drive returns when delta is neutral
3. **Liquidity & Margin Risk**: Execution costs and margin breaches dominate in stress
4. **Non-linear Losses**: Liquidation mechanics introduce convexity near thresholds

### Risk Factor Sensitivities

#### Delta (Δ)
First derivative of portfolio value with respect to underlying price (directional exposure).

```
Δ = ∂V/∂S
```

- **Spot Asset**: Δ = +1 per unit
- **Perpetual Swap**: Δ ≈ +1 per contract (long), -1 (short)

**Delta Hedging**: Rebalance positions to maintain Δ_portfolio ≈ 0.

#### Basis Delta
Sensitivity of equity to the perp-spot (or LST-spot) basis.

```
Basis = Perpetual_Price - Spot_Price
```

Basis exposure drives P&L even when delta is neutral.

#### Funding Sensitivity (Rho-like)
Sensitivity to funding rate changes (annualized).

```
Funding_PnL ≈ Notional × Funding_Rate × Time_Fraction
```

Funding is a primary carry driver and a key source of drawdowns when regimes flip.

#### Effective Gamma (Liquidation Convexity)
Non-linear losses near margin thresholds from liquidation mechanics and forced unwinds.

### Perpetual Swap Mechanics

#### Funding Rate
Periodic payment between long and short positions to anchor perpetual price to spot:

```
Funding_Payment = Position_Size × Mark_Price × Funding_Rate(annualized) × Time_Fraction
```

- **Positive Funding**: Longs pay shorts (perpetual trading at premium)
- **Negative Funding**: Shorts pay longs (perpetual trading at discount)
- **Typical Range**: Varies by venue; modeled as an annualized rate in this engine

**Impact on Delta-Neutral Strategy:**
- Short perpetual position receives funding when positive
- Creates positive carry if funding remains consistently positive
- Risk: Funding can reverse, creating negative carry

#### Basis Risk
Price divergence between spot and perpetual contracts:

```
Basis = Perpetual_Price - Spot_Price
```

**Sources:**
1. Funding rate expectations
2. Liquidity differences
3. Market microstructure effects
4. Exchange-specific factors

**Management**: Monitor basis mean reversion and adjust hedging as needed.

### Risk Metrics

#### Value at Risk (VaR)
Maximum expected loss at a given confidence level over a time horizon:

```
VaR_α = -Quantile(Returns, α)
```

Example: VaR(95%, 1-day) = $10,000 means 5% probability of losing more than $10,000 in one day.

#### Expected Shortfall (ES) / Conditional VaR
Average loss in worst-case scenarios beyond VaR threshold:

```
ES_α = -E[Returns | Returns < -VaR_α]
```

More conservative than VaR, captures tail risk severity.

#### Maximum Drawdown
Largest peak-to-trough decline in portfolio value:

```
MaxDD = max{(Peak_Value - Trough_Value) / Peak_Value}
```

Critical for understanding worst-case loss sequences and capital requirements.

#### Sharpe Ratio
Risk-adjusted return metric:

```
Sharpe = (Return - Risk_Free_Rate) / Volatility
```

- **> 1.0**: Good risk-adjusted returns
- **> 2.0**: Excellent risk-adjusted returns
- **< 0**: Strategy underperforming risk-free rate

### Stablecoin Modeling

#### Balance Sheet Structure
```
Assets:
  - Cash Reserves
  - Spot Collateral (e.g., ETH, BTC)
  - Derivative Hedge Positions

Liabilities:
  - Stablecoin Supply (circulating tokens)
  
Equity:
  - Insurance Fund
  - Protocol Reserves
```

#### Key Ratios

**Collateral Ratio:**
```
CR = Total_Assets / Stablecoin_Supply
```
- CR > 100%: Overcollateralized
- CR = 100%: Fully collateralized
- CR < 100%: Undercollateralized (insolvency risk)

**Coverage Ratio:**
```
Coverage = (Cash_Reserves + Insurance_Fund) / Stablecoin_Supply
```
Measures ability to handle immediate redemptions.

#### Delta-Neutral Stablecoin
Maintains zero directional exposure while backing stablecoin liabilities:

1. Hold collateral in volatile assets (ETH, BTC)
2. Hedge with short perpetual positions
3. Maintain Δ_net ≈ 0 to eliminate price risk
4. Generate yield from funding rate arbitrage

**Advantages:**
- Capital efficiency vs. pure cash reserves
- Funding rate income potential
- Maintains value stability regardless of crypto prices

**Risks:**
- Liquidation risk if margin insufficient
- Funding rate reversal creating negative carry
- Basis risk during market dislocations
- Counterparty/venue failure risk

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

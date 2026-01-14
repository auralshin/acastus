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

### Risk Factor Sensitivities (Greeks for Delta-Neutral Strategies)

#### Delta (Δ) - Price Sensitivity
First derivative of portfolio value with respect to underlying price (directional exposure).

```
Δ = ∂V/∂S
```

- **Spot Asset**: Δ = +1 per unit
- **Perpetual Swap**: Δ ≈ +1 per contract (long), -1 (short)
- **Target**: Δ_portfolio ≈ 0 for delta neutrality

**Delta Hedging**: Continuously rebalance positions to maintain Δ_portfolio ≈ 0.

**Delta-Neutral Portfolio Example:**
```
Long 10 ETH Spot:     Δ = +10.0
Short 10 ETH Perp:    Δ = -10.0
Net Portfolio Delta:  Δ = 0.0
```

#### Gamma (Γ) - Convexity Risk
Second derivative of portfolio value with respect to underlying price (delta sensitivity).

```
Γ = ∂²V/∂S² = ∂Δ/∂S
```

- **Linear Instruments**: Spot and perpetual swaps have Γ ≈ 0
- **Liquidation Gamma**: Non-linear losses near margin thresholds create effective gamma
- **Portfolio Impact**: Delta-neutral portfolios typically have low gamma, but liquidation mechanics introduce convexity

**Liquidation Convexity:**
```
Effective_Γ = Risk_of_Forced_Unwind_at_Loss_Prices
```

This "gamma" comes from margin calls and liquidations rather than instrument characteristics.

#### Vega (ν) - Volatility Sensitivity
Sensitivity to changes in implied volatility.

```
ν = ∂V/∂σ
```

- **Spot/Perp Portfolios**: Direct vega ≈ 0 (no options)
- **Indirect Vega**: Volatility affects funding rates, basis, and liquidity
- **Realized Vol Impact**: Higher volatility increases rebalancing costs and slippage

**For Delta-Neutral Strategies:**
```
Indirect_Vega_Sources:
- Funding rate volatility correlation
- Basis volatility (spot-perp divergence)  
- Liquidity impact (wider spreads in volatile markets)
```

#### Theta (Θ) - Time Decay
Sensitivity to passage of time.

```
Θ = ∂V/∂t
```

- **Perpetual Contracts**: No expiration, so direct theta ≈ 0
- **Funding Theta**: Time decay from funding payments
- **Carry Theta**: P&L from holding positions over time

**Delta-Neutral Time Decay Sources:**
```
Total_Θ = Funding_Θ + Basis_Θ + Transaction_Cost_Θ

Where:
Funding_Θ = Position_Notional × Funding_Rate × dt
Basis_Θ = Basis_Mean_Reversion_PnL
Transaction_Cost_Θ = -Rehedging_Costs
```

#### Rho (ρ) - Rate Sensitivity
Sensitivity to interest rate or funding rate changes.

```
ρ = ∂V/∂r
```

**For perpetual swaps, this becomes funding rate sensitivity:**

```
Funding_ρ = ∂V/∂(Funding_Rate)
         = Position_Size × Mark_Price × Time_to_Next_Funding
```

- **Long Perp**: Negative rho (pays when funding increases)
- **Short Perp**: Positive rho (receives when funding increases)  
- **Delta-Neutral Impact**: Net funding exposure depends on hedge ratio precision

#### Basis Delta
Sensitivity to the perp-spot (or derivative-underlying) basis.

```
Basis_Δ = ∂V/∂(Perp_Price - Spot_Price)
```

**Critical for Delta-Neutral Strategies:**
- Even with perfect delta hedge, basis changes create P&L
- Basis risk cannot be hedged away without perfect correlation
- Primary risk factor when price delta is neutralized

#### Lambda (Liquidity Sensitivity)
Sensitivity to market liquidity and bid-ask spreads.

```
λ = ∂V/∂(Spread)
```

**Delta-Neutral Liquidity Risk:**
- Rebalancing frequency creates transaction cost drag
- Stressed markets widen spreads precisely when rehedging needed
- Asymmetric: easier to get long than short in crashes

**Liquidity-Adjusted Greeks:**
```
Effective_Δ = Theoretical_Δ ± Spread_Impact
Effective_Γ = Theoretical_Γ + Liquidity_Convexity
```

### Greek Interaction Matrix for Delta-Neutral Portfolios

| Greek | Direct Impact | Indirect Impact | Hedging Strategy |
|-------|---------------|-----------------|------------------|
| **Delta** | Target = 0 | Basis drift | Continuous rebalancing |
| **Gamma** | ~0 (linear instruments) | Liquidation risk | Margin management |
| **Vega** | ~0 (no options) | Funding volatility | Volatility regime awareness |
| **Theta** | Funding carry | Transaction costs | Optimize rebalance frequency |
| **Rho** | Funding sensitivity | Rate cycle impact | Funding rate hedging |

### Key Insights for Delta-Neutral Risk Management

1. **Primary Risk Shifts**: From directional (delta) to basis, funding, and liquidity risks
2. **Hidden Gamma**: Liquidation mechanics create convexity even with linear instruments  
3. **Funding Dominance**: Rho (funding sensitivity) often dominates P&L in delta-neutral strategies
4. **Correlation Breakdown**: When spot-perp correlation fails, "delta neutral" becomes directional
5. **Regime Sensitivity**: Greeks change dramatically across market regimes (calm vs. stressed)

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

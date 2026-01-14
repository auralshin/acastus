"""Streamlit dashboard for Acastus Risk Engine."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

from risk_engine.instruments.spot import SpotAsset
from risk_engine.instruments.perp import PerpetualSwap
from risk_engine.instruments.base import MarketData
from risk_engine.portfolio.positions import Portfolio, Position
from risk_engine.simulation.paths import PathGenerator
from risk_engine.simulation.engine import SimulationEngine
from risk_engine.analytics.metrics import RiskMetricsCalculator
from risk_engine.margin.models import MarginModel
from risk_engine.hedging.delta import DeltaHedger, HedgeTarget


st.set_page_config(
    page_title="Acastus Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


st.title("Acastus Risk Engine")
st.caption("Delta-neutral strategy and stablecoin solvency simulator.")
st.markdown("---")


st.sidebar.header("Inputs")

portfolio_type = st.sidebar.selectbox(
    "Portfolio",
    ["Trading Strategy", "Stablecoin"],
    help="Select the portfolio template to simulate."
)

if portfolio_type == "Trading Strategy":
    st.sidebar.subheader("Positions")

    asset_symbol = st.sidebar.selectbox("Asset", ["ETH", "BTC", "SOL"])
    spot_quantity = st.sidebar.number_input(
        "Spot Quantity (units)",
        min_value=0.0,
        value=100.0,
        step=1.0,
        format="%.2f"
    )
    perp_quantity = st.sidebar.number_input(
        "Perp Quantity (units)",
        value=-100.0,
        step=1.0,
        format="%.2f",
        help="Negative values represent short positions."
    )
    initial_price = st.sidebar.number_input(
        "Initial Price ($)",
        min_value=0.01,
        value=2000.0,
        step=50.0,
        format="%.2f"
    )
    initial_cash = st.sidebar.number_input(
        "Cash Balance ($)",
        min_value=0.0,
        value=1_000_000.0,
        step=50_000.0,
        format="%.0f"
    )


    stablecoin_supply = 0.0
    reserve_cash = 0.0
    insurance_fund = 0.0
    eth_collateral = 0.0
    eth_perp_hedge = 0.0

else:
    st.sidebar.subheader("Balance Sheet")

    stablecoin_supply = st.sidebar.number_input(
        "Stablecoin Supply ($)",
        min_value=0.0,
        value=10_000_000.0,
        step=250_000.0,
        format="%.0f"
    )
    reserve_cash = st.sidebar.number_input(
        "Reserve Cash ($)",
        min_value=0.0,
        value=2_000_000.0,
        step=100_000.0,
        format="%.0f"
    )
    insurance_fund = st.sidebar.number_input(
        "Insurance Fund ($)",
        min_value=0.0,
        value=500_000.0,
        step=50_000.0,
        format="%.0f"
    )
    eth_collateral = st.sidebar.number_input(
        "Collateral (ETH)",
        min_value=0.0,
        value=5_000.0,
        step=100.0,
        format="%.2f"
    )
    eth_perp_hedge = st.sidebar.number_input(
        "Perp Hedge (ETH)",
        value=-5_000.0,
        step=100.0,
        format="%.2f",
        help="Negative values represent short hedges."
    )
    initial_price = st.sidebar.number_input(
        "ETH Price ($)",
        min_value=0.01,
        value=2000.0,
        step=50.0,
        format="%.2f"
    )


    asset_symbol = "ETH"
    spot_quantity = 0.0
    perp_quantity = 0.0
    initial_cash = 0.0


st.sidebar.markdown("---")
st.sidebar.subheader("Simulation")

n_scenarios = st.sidebar.slider(
    "Scenarios",
    min_value=200,
    max_value=5000,
    value=1000,
    step=100,
    help="Number of Monte Carlo paths."
)
n_days = st.sidebar.slider(
    "Horizon (days)",
    min_value=7,
    max_value=180,
    value=30,
    step=1
)
volatility = st.sidebar.slider(
    "Spot Volatility (annualized)",
    min_value=0.2,
    max_value=1.5,
    value=0.6,
    step=0.05,
    format="%.2f"
)

with st.sidebar.expander("Risk Factors", expanded=False):
    funding_volatility = st.slider(
        "Funding Volatility (annualized)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Funding rate standard deviation."
    )
    basis_volatility = st.slider(
        "Basis Volatility (pct)",
        min_value=0.0,
        max_value=0.1,
        value=0.02,
        step=0.01,
        format="%.2f",
        help="Perp-spot spread volatility as a fraction of spot."
    )
    correlation_breakdown = st.slider(
        "Correlation Breakdown",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        format="%.2f",
        help="Higher values widen basis during stress."
    )

with st.sidebar.expander("Price Path", expanded=False):
    price_drift = st.slider(
        "Drift (annualized)",
        min_value=-0.2,
        max_value=0.2,
        value=0.0,
        step=0.01,
        format="%.2f",
        help="Expected annual drift in the spot price."
    )
    jump_intensity = st.slider(
        "Jump Intensity (per year)",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        format="%.2f",
        help="Expected number of jumps per year."
    )
    jump_mean = st.slider(
        "Jump Mean (log)",
        min_value=-0.2,
        max_value=0.2,
        value=-0.02,
        step=0.01,
        format="%.2f",
        help="Average jump size in log returns."
    )
    jump_std = st.slider(
        "Jump Std (log)",
        min_value=0.0,
        max_value=0.3,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="Jump size volatility in log returns."
    )

st.sidebar.caption("All rates are annualized. Simulation runs on hourly steps.")

run_simulation = st.sidebar.button(
    "Run Simulation", type="primary", width="stretch")


if run_simulation:

    asset = asset_symbol if portfolio_type == "Trading Strategy" else "ETH"
    initial_nav = 0.0
    margin_model = MarginModel()

    with st.spinner("Running Monte Carlo simulation..."):


        if portfolio_type == "Trading Strategy":
            spot = SpotAsset(asset_symbol)
            perp = PerpetualSwap(f"{asset_symbol}-PERP")

            portfolio = Portfolio(
                positions=[
                    Position(spot, quantity=spot_quantity,
                             entry_price=initial_price, venue="binance"),
                    Position(perp, quantity=perp_quantity,
                             entry_price=initial_price, venue="dydx"),
                ],
                cash=initial_cash
            )


            market = MarketData(
                timestamp=datetime.now(),
                spot_prices={asset_symbol: initial_price},
                perp_prices={f"{asset_symbol}-PERP": initial_price},
                funding_rates={f"{asset_symbol}-PERP": 0.0001}
            )


            initial_nav = portfolio.net_asset_value(market)
            net_delta = portfolio.net_delta(market)

            st.success("Portfolio created successfully")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial NAV", f"${initial_nav:,.2f}")
            with col2:
                delta_value = list(net_delta.values())[0] if net_delta else 0
                st.metric("Net Delta", f"{delta_value:.2f}")
            with col3:
                is_neutral = abs(delta_value) < 0.1 * abs(spot_quantity)
                st.metric("Delta Neutral", "Yes" if is_neutral else "No")

            st.markdown("### Risk Factor Sensitivities")

            exposures = portfolio.aggregate_risk_exposures(market)
            margin_reqs = margin_model.calculate_portfolio_margin(
                portfolio, market, by_venue=False
            )
            margin_ratio = margin_reqs["total"].margin_ratio
            liquidation_buffer = margin_reqs["total"].distance_to_liquidation_pct()

            basis_sensitivity = exposures.basis_delta
            funding_sensitivity_1bp = exposures.rho * 0.0001

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Net Delta (units)", f"{delta_value:.2f}")
            with col2:
                st.metric("Basis Sensitivity ($ per $1)", f"${basis_sensitivity:,.2f}")
            with col3:
                st.metric("Funding Sensitivity ($ per 1bp)", f"${funding_sensitivity_1bp:,.0f}")
            with col4:
                st.metric("Margin Ratio", f"{margin_ratio:.2f}x")
            with col5:
                st.metric("Buffer to Liquidation", f"{liquidation_buffer:.1f}%")

        else:
            spot = SpotAsset("ETH")
            perp = PerpetualSwap("ETH-PERP")


            portfolio = Portfolio(
                positions=[
                    Position(spot, quantity=eth_collateral,
                             entry_price=initial_price, venue="binance"),
                    Position(perp, quantity=eth_perp_hedge,
                             entry_price=initial_price, venue="dydx"),
                ],
                cash=reserve_cash + insurance_fund
            )

            market = MarketData(
                timestamp=datetime.now(),
                spot_prices={"ETH": initial_price},
                perp_prices={"ETH-PERP": initial_price},
                funding_rates={"ETH-PERP": 0.0001}
            )


            initial_nav = portfolio.net_asset_value(market)
            collateral_value = eth_collateral * initial_price
            collateral_ratio = collateral_value / \
                stablecoin_supply if stablecoin_supply > 0 else 0
            coverage_ratio = (collateral_value + reserve_cash + insurance_fund) / \
                stablecoin_supply if stablecoin_supply > 0 else 0
            net_delta_val = eth_collateral + eth_perp_hedge

            st.success("Stablecoin balance sheet created")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Collateral Ratio", f"{collateral_ratio:.2%}")
            with col2:
                st.metric("Coverage Ratio", f"{coverage_ratio:.2%}")
            with col3:
                st.metric("Stablecoin Supply", f"${stablecoin_supply:,.2f}")
            with col4:
                st.metric("Net Delta", f"{net_delta_val:.2f}")

            st.markdown("### Risk Factor Sensitivities")

            exposures = portfolio.aggregate_risk_exposures(market)
            margin_reqs = margin_model.calculate_portfolio_margin(
                portfolio, market, by_venue=False
            )
            margin_ratio = margin_reqs["total"].margin_ratio
            liquidation_buffer = margin_reqs["total"].distance_to_liquidation_pct()

            basis_sensitivity = exposures.basis_delta
            funding_sensitivity_1bp = exposures.rho * 0.0001

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Net Delta (units)", f"{net_delta_val:.2f}")
            with col2:
                st.metric("Basis Sensitivity ($ per $1)", f"${basis_sensitivity:,.2f}")
            with col3:
                st.metric("Funding Sensitivity ($ per 1bp)", f"${funding_sensitivity_1bp:,.0f}")
            with col4:
                st.metric("Margin Ratio", f"{margin_ratio:.2f}x")
            with col5:
                st.metric("Buffer to Liquidation", f"{liquidation_buffer:.1f}%")

        st.markdown("### Sensitivity to Price Shocks")

        shock_steps = 21
        shock_range = np.linspace(-0.5, 0.5, shock_steps)
        funding_sensitivities = []
        margin_ratios = []
        liquidation_buffers = []
        shocked_prices = []

        for shock in shock_range:
            shocked_price = initial_price * (1 + shock)
            shocked_prices.append(shocked_price)
            shocked_market = MarketData(
                timestamp=market.timestamp,
                spot_prices={asset: shocked_price},
                perp_prices={f"{asset}-PERP": shocked_price},
                funding_rates=market.funding_rates
            )
            shocked_exposures = portfolio.aggregate_risk_exposures(shocked_market)
            shocked_margin = margin_model.calculate_portfolio_margin(
                portfolio, shocked_market, by_venue=False
            )["total"]

            funding_sensitivities.append(shocked_exposures.rho * 0.0001)
            margin_ratios.append(shocked_margin.margin_ratio)
            liquidation_buffers.append(shocked_margin.distance_to_liquidation_pct())

        shock_pct = shock_range * 100

        price_fig = go.Figure()
        price_fig.add_trace(
            go.Scatter(
                x=shock_pct,
                y=shocked_prices,
                mode="lines",
                name=f"{asset} Spot Price"
            )
        )
        price_fig.update_layout(
            height=250,
            title=f"{asset} Spot Price vs Shock",
            xaxis_title="Spot Price Shock (%)",
            yaxis_title="Spot Price ($)"
        )
        st.plotly_chart(price_fig, width="stretch")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=shock_pct,
                y=funding_sensitivities,
                mode="lines",
                name="Funding Sensitivity ($/1bp)"
            ),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=shock_pct,
                y=margin_ratios,
                mode="lines",
                name="Margin Ratio (x)"
            ),
            secondary_y=True
        )
        fig.add_trace(
            go.Scatter(
                x=shock_pct,
                y=liquidation_buffers,
                mode="lines",
                name="Buffer to Liquidation (%)",
                line=dict(dash="dash")
            ),
            secondary_y=True
        )

        fig.update_layout(
            height=350,
            title="Funding Carry and Margin Resilience vs Price Shock",
            xaxis_title="Spot Price Shock (%)"
        )
        fig.update_yaxes(
            title_text="Funding Sensitivity ($/1bp)", secondary_y=False
        )
        fig.update_yaxes(
            title_text="Margin / Buffer", secondary_y=True
        )

        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Funding sensitivity scales with notional as price moves; "
            "margin ratio and liquidation buffer show non-linear risk near stress levels."
        )

        st.markdown("---")


        scenario_start = time.time()
        with st.spinner("Generating market scenarios..."):
            path_generator = PathGenerator([asset], seed=42)

            n_steps = n_days * 24
            time_step_hours = 1.0

            scenarios = path_generator.generate_multiple_scenarios(
                n_scenarios=n_scenarios,
                start_time=datetime.now(),
                n_steps=n_steps,
                time_step_hours=time_step_hours,
                initial_prices={asset: initial_price},
                initial_rates={asset: 0.0001},
                price_volatilities={asset: volatility},
                rate_params={
                    "mean_rates": {asset: 0.1},
                    "rate_volatilities": {asset: funding_volatility},
                    "mean_reversion_speed": 2.0
                },
                price_params={
                    "drift": price_drift,
                    "jump_intensity": jump_intensity,
                    "jump_mean": jump_mean,
                    "jump_std": jump_std
                },
                basis_params={
                    "volatility_pct": basis_volatility,
                    "correlation_breakdown": correlation_breakdown
                }
            )

        scenario_time = time.time() - scenario_start
        st.info(
            f"Generated {len(scenarios)} scenarios with {n_steps} steps each in {scenario_time:.2f}s")


        sim_start = time.time()
        with st.spinner("Running Monte Carlo simulation..."):

            hedge_targets = {
                asset: HedgeTarget(
                    symbol=asset,
                    target_delta=0.0,
                    tolerance_pct=0.05
                )
            }

            hedger = DeltaHedger(
                hedge_targets=hedge_targets,
                hedge_instrument="perp"
            )

            sim_engine = SimulationEngine(
                initial_portfolio=portfolio,
                margin_model=margin_model,
                hedger=hedger
            )


            progress_bar = st.progress(0, text="Running simulations...")

            def update_progress(completed, total):
                progress_bar.progress(
                    completed / total, text=f"Running simulations... {completed}/{total}")


            results = sim_engine.run_multiple_scenarios(
                scenarios,
                parallel=True,
                progress_callback=update_progress
            )

            progress_bar.empty()

        sim_time = time.time() - sim_start
        scenarios_per_sec = len(results) / sim_time if sim_time > 0 else 0

        st.success(
            f"Completed {len(results)} simulations in {sim_time:.2f}s ({scenarios_per_sec:.1f} scenarios/sec) - Parallel processing enabled")


        metrics_calc = RiskMetricsCalculator.calculate_from_simulations(
            results)


        st.markdown("## Simulation Results")

        st.info("""
        **Understanding the Results:**

        - **NAV (Net Asset Value)**: Total portfolio value including cash, spot positions, and unrealized P&L from perpetual swaps
        - **Max Drawdown**: Largest peak-to-trough decline in portfolio value. Critical for risk management and capital allocation
        - **VaR (Value at Risk)**: Maximum expected loss at a given confidence level. VaR 95% means losses should not exceed this value 95% of the time
        - **Sharpe Ratio**: Risk-adjusted return metric. Higher is better. Above 1.0 is considered good, above 2.0 is excellent
        """)


        final_navs = [r.final_nav for r in results]
        mean_final_nav = np.mean(final_navs)
        returns = [(nav - initial_nav) / initial_nav for nav in final_navs]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Avg Final NAV", f"${mean_final_nav:,.2f}", delta=f"{np.mean(returns):.2%}")
        with col2:
            st.metric("Max Drawdown", f"{metrics_calc.max_drawdown:.2%}")
        with col3:
            st.metric(
                "VaR (95%)", f"${initial_nav * metrics_calc.var_95:,.2f}")
        with col4:
            st.metric("Sharpe Ratio", f"{metrics_calc.sharpe_ratio:.2f}")

        st.markdown("---")


        tab1, tab2, tab3, tab4 = st.tabs(
            ["NAV Distribution", "Risk Decomposition", "Funding Impact", "Correlation Analysis"])

        with tab1:
            st.subheader("Portfolio Value Distribution")

            final_navs = [r.final_nav for r in results]

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=final_navs,
                nbinsx=50,
                name="Final NAV",
                marker_color='rgb(55, 83, 109)'
            ))


            var_95 = np.percentile(final_navs, 5)
            var_99 = np.percentile(final_navs, 1)

            fig.add_vline(x=var_95, line_dash="dash", line_color="orange",
                          annotation_text="VaR 95%", annotation_position="top")
            fig.add_vline(x=var_99, line_dash="dash", line_color="red",
                          annotation_text="VaR 99%", annotation_position="top")

            fig.update_layout(
                title="Final Portfolio Value Distribution",
                xaxis_title="Final NAV ($)",
                yaxis_title="Frequency",
                showlegend=False,
                height=500
            )

            st.plotly_chart(fig, width="stretch")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean NAV", f"${np.mean(final_navs):,.2f}")
            with col2:
                returns = [(nav - initial_nav) /
                           initial_nav for nav in final_navs]
                st.metric("Mean Return", f"{np.mean(returns):.2%}")
            with col3:
                st.metric("Volatility", f"{np.std(returns):.2%}")
            with col4:
                negative_returns = [r for r in returns if r < 0]
                st.metric("Loss Probability",
                          f"{len(negative_returns) / len(returns):.1%}")

        with tab2:
            st.subheader("Risk Factor Decomposition")


            st.markdown("#### Key Risk Factors for Delta-Neutral Strategies:")


            funding_costs = []
            basis_pnl = []
            for result in results:
                total_funding = 0
                total_basis = 0
                for i in range(1, len(result.states)):
                    prev_state = result.states[i-1]
                    curr_state = result.states[i]


                    for pos in curr_state.portfolio.positions:
                        if hasattr(pos.instrument, 'calculate_funding_payment'):
                            funding_rate = curr_state.market.funding_rates.get(
                                pos.instrument.symbol, 0.0)
                            funding_cost = pos.quantity * pos.entry_price * funding_rate / 8760
                            total_funding += funding_cost


                    if len(curr_state.market.spot_prices) > 0 and len(curr_state.market.perp_prices) > 0:
                        symbol = list(curr_state.market.spot_prices.keys())[0]
                        spot_price = curr_state.market.spot_prices[symbol]
                        perp_price = curr_state.market.perp_prices.get(
                            f"{symbol}-PERP", spot_price)
                        basis_diff = perp_price - spot_price
                        total_basis += basis_diff * 0.1

                funding_costs.append(total_funding)
                basis_pnl.append(total_basis)

            col1, col2 = st.columns(2)

            with col1:

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=funding_costs,
                    nbinsx=30,
                    name="Funding Costs",
                    marker_color='rgb(219, 64, 82)'
                ))
                fig.update_layout(
                    title="Funding Cost Distribution",
                    xaxis_title="Cumulative Funding Cost ($)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, width="stretch")

            with col2:

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=basis_pnl,
                    nbinsx=30,
                    name="Basis P&L",
                    marker_color='rgb(55, 128, 191)'
                ))
                fig.update_layout(
                    title="Basis Risk P&L Distribution",
                    xaxis_title="Basis P&L ($)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, width="stretch")


            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Funding Cost",
                          f"${np.mean(funding_costs):,.2f}")
            with col2:
                st.metric("Funding Volatility",
                          f"${np.std(funding_costs):,.2f}")
            with col3:
                st.metric("Avg Basis P&L", f"${np.mean(basis_pnl):,.2f}")
            with col4:
                st.metric("Basis Volatility", f"${np.std(basis_pnl):,.2f}")

        with tab3:
            st.subheader("Funding Rate Impact Analysis")


            sample_results = results[:min(10, len(results))]

            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=["Portfolio NAV Evolution", "Funding Rate Evolution"])

            for result in sample_results:
                nav_path = [state.nav for state in result.states]
                funding_path = []

                for state in result.states:
                    if state.market.funding_rates:
                        avg_funding = np.mean(
                            list(state.market.funding_rates.values()))
                        funding_path.append(avg_funding * 8760)
                    else:
                        funding_path.append(0)

                time_steps = list(range(len(nav_path)))

                fig.add_trace(go.Scatter(
                    x=time_steps,
                    y=nav_path,
                    mode='lines',
                    opacity=0.3,
                    showlegend=False,
                    line=dict(color='blue')
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=time_steps,
                    y=funding_path,
                    mode='lines',
                    opacity=0.3,
                    showlegend=False,
                    line=dict(color='red')
                ), row=2, col=1)

            fig.update_xaxes(title_text="Time Steps (hours)", row=2, col=1)
            fig.update_yaxes(title_text="Portfolio NAV ($)", row=1, col=1)
            fig.update_yaxes(
                title_text="Funding Rate (% annual)", row=2, col=1)
            fig.update_layout(
                height=600, title_text="Portfolio Performance vs Funding Rates")

            st.plotly_chart(fig, width="stretch")


            st.markdown("#### Key Insights for Delta-Neutral Strategies:")
            st.markdown("""
            - **Funding Rate Risk**: Even with perfect delta hedging, funding costs can erode returns
            - **Basis Risk**: Spot-perp spread changes create P&L even when delta-neutral
            - **Correlation Risk**: When spot-perp correlation breaks down, hedging becomes less effective
            - **Volatility Impact**: Higher volatility increases rehedging costs and basis instability
            """)

        with tab4:
            st.subheader("Correlation and Model Breakdown Analysis")

            if len(results) > 1:
                price_changes = []
                pnl_values = []

                for result in results:
                    if len(result.states) > 1:
                        initial_price = list(
                            result.states[0].market.spot_prices.values())[0]
                        final_price = list(
                            result.states[-1].market.spot_prices.values())[0]
                        price_change = (
                            final_price - initial_price) / initial_price
                        pnl = (result.final_nav - initial_nav) / initial_nav

                        price_changes.append(price_change)
                        pnl_values.append(pnl)


                correlation = np.corrcoef(price_changes, pnl_values)[0, 1]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_changes,
                    y=pnl_values,
                    mode='markers',
                    marker=dict(
                        color=pnl_values,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="P&L")
                    ),
                    hovertemplate="Price Change: %{x:.2%}<br>P&L: %{y:.2%}<extra></extra>"
                ))


                z = np.polyfit(price_changes, pnl_values, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(price_changes),
                                      max(price_changes), 100)
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend',
                    showlegend=False
                ))

                fig.update_layout(
                    title=f"Delta-Neutral Performance vs Price Changes (Correlation: {correlation:.3f})",
                    xaxis_title="Underlying Price Change (%)",
                    yaxis_title="Portfolio P&L (%)",
                    height=500
                )
                st.plotly_chart(fig, width="stretch")


                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Price-P&L Correlation", f"{correlation:.3f}")
                    if abs(correlation) < 0.1:
                        st.success("Good delta neutrality")
                    else:
                        st.warning("Residual directional exposure")

                with col2:
                    avg_pnl = np.mean(pnl_values)
                    st.metric("Average Return", f"{avg_pnl:.2%}")
                    if avg_pnl > 0:
                        st.success("Positive carry")
                    else:
                        st.warning("Negative carry")

                with col3:
                    pnl_vol = np.std(pnl_values)
                    st.metric("Return Volatility", f"{pnl_vol:.2%}")
                    if pnl_vol < 0.05:
                        st.success("Low volatility")
                    else:
                        st.info("Moderate volatility")

                st.markdown("#### Model Effectiveness:")
                st.markdown(f"""
                - **Delta Neutrality**: {('Excellent' if abs(correlation) < 0.05 else 'Good' if abs(correlation) < 0.15 else 'Poor')} (correlation: {correlation:.3f})
                - **Risk-Adjusted Return**: {avg_pnl/pnl_vol:.2f} (return/volatility ratio)
                - **Strategy Viability**: {'Strategy working well' if avg_pnl > 0 and abs(correlation) < 0.15 else 'Consider adjustments'}
                """)

            else:
                st.info("Run more scenarios for correlation analysis")

else:

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ## Welcome to Acastus Risk Engine Dashboard

        Configure your portfolio in the sidebar to begin analysis.

        ### Quick Start:
        1. Select portfolio type (Trading Strategy or Stablecoin)
        2. Configure positions and parameters
        3. Click "Run Simulation"

        The dashboard will generate Monte Carlo simulations and provide comprehensive risk analysis.
        """)

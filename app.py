"""Streamlit dashboard for the Backtest Robustness Lab.

Run from this folder with: python -m streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from robustness_lab import (
    StrategyConfig,
    download_prices,
    fit_bayesian_alpha,
    parameter_surface,
    posterior_predictive_paths,
    posterior_summary,
    robustness_summary,
    run_moving_average_backtest,
    run_stress_test,
)

st.set_page_config(page_title="Bayesian Backtest Robustness Lab", layout="wide")


@st.cache_data(ttl=3_600, show_spinner=False)
def cached_prices(symbol: str, start: str, end: str):
    return download_prices(symbol, start, end)


def percent(value: float) -> str:
    return f"{value:.1%}"


def density_figure(alpha_draws: np.ndarray) -> go.Figure:
    annual_alpha = alpha_draws * 252
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(x=annual_alpha, histnorm="probability density", nbinsx=45, marker_color="#2E74B5")
    )
    figure.add_vline(x=0.0, line_dash="dash", line_color="#B42318", annotation_text="zero alpha")
    figure.update_layout(
        title="Posterior distribution of annualized alpha",
        xaxis_title="annualized alpha",
        yaxis_title="posterior density",
        template="plotly_white",
        bargap=0.03,
    )
    return figure


def fan_chart(paths: np.ndarray) -> go.Figure:
    days = np.arange(1, paths.shape[1] + 1)
    low, median, high = np.quantile(paths, [0.05, 0.50, 0.95], axis=0)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=days, y=high, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    figure.add_trace(
        go.Scatter(
            x=days, y=low, fill="tonexty", fillcolor="rgba(46,116,181,0.18)",
            line=dict(width=0), name="90% predictive band", hoverinfo="skip",
        )
    )
    figure.add_trace(go.Scatter(x=days, y=median, line=dict(color="#0B2545", width=3), name="median path"))
    figure.update_layout(
        title="One-year posterior-predictive equity paths",
        xaxis_title="future trading days", yaxis_title="equity from $1.00", template="plotly_white",
    )
    return figure


st.title("Bayesian Backtest Robustness Lab")
st.caption("A research dashboard for evidence, and research only")

with st.sidebar:
    st.header("Research controls")
    symbol = st.text_input("ETF ticker", "SPY").upper().strip()
    start = st.text_input("Start date (YYYY-MM-DD)", "2010-01-01")
    end = st.text_input("End date (YYYY-MM-DD)", "2026-01-01")
    fast = st.slider("Fast moving average", 10, 120, 50, step=5)
    slow = st.slider("Slow moving average", 100, 320, 200, step=10)
    cost_bps = st.slider("One-way cost (bps)", 0.0, 50.0, 10.0, step=1.0)
    delay = st.slider("Extra execution delay (days)", 0, 2, 0)
    prior = st.selectbox("Alpha prior", ["skeptical", "neutral", "wide"], index=0)
    run = st.button("Run research", type="primary", use_container_width=True)

if fast >= slow:
    st.error("The fast window must be smaller than the slow window.")
    st.stop()

try:
    prices = cached_prices(symbol, start, end)
except Exception as error:
    st.error(f"Could not load price data: {error}")
    st.stop()

if not run:
    st.info("Set your assumptions, then select **Run research**. The Bayesian step takes longer than the charts.")
    st.line_chart(prices)
    st.stop()

config = StrategyConfig(fast, slow, cost_bps, delay)
with st.spinner("Running backtest, scenario stress tests, and Bayesian sampling..."):
    result = run_moving_average_backtest(prices, config)
    surface = parameter_surface(prices, range(20, 101, 10), range(100, 321, 20), cost_bps, delay)
    scenarios = run_stress_test(prices, n_scenarios=250, seed=7)
    bayes = fit_bayesian_alpha(
        result.frame["net_return"], result.frame["asset_return"], prior=prior, draws=500, tune=500, seed=42
    )
    evidence = posterior_summary(bayes)
    paths = posterior_predictive_paths(bayes, result.frame["asset_return"], n_paths=300, seed=99)

summary = robustness_summary(scenarios)
m1, m2, m3, m4 = st.columns(4)
m1.metric("P(alpha > 0)", percent(evidence["p_alpha_positive"]))
m2.metric("95% alpha interval", f"{percent(evidence['annual_alpha_low'])} to {percent(evidence['annual_alpha_high'])}")
m3.metric("Stress survival rate", percent(summary["survival_rate"]))
m4.metric("Headline Sharpe", f"{result.metrics['sharpe']:.2f}")

left, right = st.columns(2)
with left:
    equity = result.frame[["equity", "benchmark_equity"]].rename(columns={"equity": "strategy", "benchmark_equity": symbol})
    st.plotly_chart(px.line(equity, title="Net strategy equity versus buy-and-hold", template="plotly_white"), use_container_width=True)
    st.plotly_chart(density_figure(bayes.posterior["alpha"].values.reshape(-1)), use_container_width=True)
with right:
    matrix = surface.pivot(index="slow_window", columns="fast_window", values="sharpe").sort_index()
    heatmap = px.imshow(matrix, color_continuous_scale="viridis", origin="lower", aspect="auto", title="Parameter surface: Sharpe ratio")
    heatmap.update_layout(template="plotly_white", xaxis_title="fast window", yaxis_title="slow window")
    st.plotly_chart(heatmap, use_container_width=True)
    st.plotly_chart(fan_chart(paths), use_container_width=True)

st.subheader("Implementation fragility")
fragility = px.scatter(
    scenarios, x="cost_bps", y="sharpe", color="execution_delay_days", hover_data=["fast_window", "slow_window", "max_drawdown"],
    color_continuous_scale="Viridis", title="Every point is one pre-registered stress scenario",
)
fragility.update_layout(template="plotly_white")
st.plotly_chart(fragility, use_container_width=True)

with st.expander("Research diagnostics and assumptions"):
    st.write({key: round(value, 4) for key, value in evidence.items()})
    st.write("A healthy MCMC fit usually has R-hat close to 1.00 and a comfortably large effective sample size.")
    st.dataframe(scenarios.sort_values("sharpe", ascending=False).head(20), use_container_width=True)

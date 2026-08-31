"""Core research engine for the Backtest Robustness Lab.

The module is deliberately compact and explicit. It is designed for learning,
not for routing live orders or making investment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import yfinance as yf

TRADING_DAYS = 252


@dataclass(frozen=True)
class StrategyConfig:
    """Every assumption that changes a moving-average backtest."""

    fast_window: int = 50
    slow_window: int = 200
    cost_bps: float = 10.0
    execution_delay_days: int = 0

    def validate(self) -> None:
        if self.fast_window < 2:
            raise ValueError("fast_window must be at least 2.")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window.")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative.")
        if self.execution_delay_days < 0:
            raise ValueError("execution_delay_days cannot be negative.")


@dataclass
class BacktestResult:
    config: StrategyConfig
    frame: pd.DataFrame
    metrics: dict[str, float]


def download_prices(symbol: str, start: str = "2010-01-01", end: str | None = None) -> pd.Series:
    """Download adjusted daily closes and return one clean, named Series.

    `auto_adjust=True` adjusts historical prices for splits and dividends, so the
    return series is internally consistent for this educational project.
    """
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No price data returned for {symbol}. Check symbol and dates.")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.name = symbol.upper()
    if len(close) < 260:
        raise ValueError("Need at least 260 daily prices for a meaningful first run.")
    return close


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    growth = float((1.0 + returns).prod())
    return growth ** (periods_per_year / len(returns)) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Return the most negative peak-to-trough loss, e.g. -0.32 for -32%."""
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    """Compute metrics from a *net*, daily return series."""
    returns = returns.dropna()
    if len(returns) < 2:
        raise ValueError("At least two returns are required.")
    daily_vol = float(returns.std(ddof=1))
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS)
    sharpe = float(returns.mean() / daily_vol * np.sqrt(TRADING_DAYS)) if daily_vol else float("nan")
    equity = (1.0 + returns).cumprod()
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_return": annualized_return(returns),
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "observations": float(len(returns)),
    }


def run_moving_average_backtest(prices: pd.Series, config: StrategyConfig) -> BacktestResult:
    """Run a long/flat MA-cross strategy without same-bar look-ahead.

    The signal is observed at the end of day t. Even with zero additional delay,
    the position is shifted one business day: it can first earn return on t+1.
    """
    config.validate()
    price = prices.astype(float).dropna().sort_index().rename("price")
    frame = pd.DataFrame(index=price.index)
    frame["price"] = price
    frame["asset_return"] = price.pct_change().fillna(0.0)
    frame["fast_ma"] = price.rolling(config.fast_window, min_periods=config.fast_window).mean()
    frame["slow_ma"] = price.rolling(config.slow_window, min_periods=config.slow_window).mean()

    # The raw signal is a decision made after observing today's close.
    frame["raw_signal"] = (frame["fast_ma"] > frame["slow_ma"]).astype(float)
    implementation_lag = 1 + config.execution_delay_days
    frame["position"] = frame["raw_signal"].shift(implementation_lag).fillna(0.0)

    # A 0 -> 1 or 1 -> 0 change is one unit of turnover. Cost is charged once
    # per directional change and expressed in basis points: 10 bps = 0.0010.
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    cost_rate = config.cost_bps / 10_000.0
    frame["trading_cost"] = cost_rate * frame["turnover"]
    frame["gross_return"] = frame["position"] * frame["asset_return"]
    frame["net_return"] = frame["gross_return"] - frame["trading_cost"]
    frame["equity"] = (1.0 + frame["net_return"]).cumprod()
    frame["benchmark_equity"] = (1.0 + frame["asset_return"]).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return BacktestResult(config=config, frame=frame, metrics=performance_metrics(frame["net_return"]))


def parameter_surface(
    prices: pd.Series,
    fast_windows: Iterable[int],
    slow_windows: Iterable[int],
    cost_bps: float = 10.0,
    execution_delay_days: int = 0,
) -> pd.DataFrame:
    """Evaluate a full parameter grid; do not hide the bad settings."""
    rows: list[dict[str, float]] = []
    for fast in fast_windows:
        for slow in slow_windows:
            if fast >= slow:
                continue
            result = run_moving_average_backtest(
                prices,
                StrategyConfig(fast, slow, cost_bps, execution_delay_days),
            )
            rows.append({"fast_window": fast, "slow_window": slow, **result.metrics})
    if not rows:
        raise ValueError("No valid fast/slow window combinations were supplied.")
    return pd.DataFrame(rows)


def run_stress_test(
    prices: pd.Series,
    *,
    n_scenarios: int = 250,
    fast_choices: Iterable[int] = (20, 30, 40, 50, 60, 80),
    slow_choices: Iterable[int] = (100, 150, 200, 250, 300),
    min_cost_bps: float = 0.0,
    max_cost_bps: float = 40.0,
    max_execution_delay_days: int = 2,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a reproducible scenario ledger of implementation assumptions."""
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be at least 1.")
    rng = np.random.default_rng(seed)
    fast_choices, slow_choices = tuple(fast_choices), tuple(slow_choices)
    records: list[dict[str, float]] = []
    for scenario_id in range(n_scenarios):
        fast = int(rng.choice(fast_choices))
        valid_slow = tuple(s for s in slow_choices if s > fast)
        if not valid_slow:
            continue
        slow = int(rng.choice(valid_slow))
        cost = float(rng.uniform(min_cost_bps, max_cost_bps))
        delay = int(rng.integers(0, max_execution_delay_days + 1))
        config = StrategyConfig(fast, slow, cost, delay)
        result = run_moving_average_backtest(prices, config)
        records.append({"scenario_id": scenario_id, "seed": seed, **asdict(config), **result.metrics})
    return pd.DataFrame(records)


def robustness_summary(scenarios: pd.DataFrame) -> dict[str, float]:
    """A transparent summary; individual scenario plots remain the evidence."""
    if scenarios.empty:
        raise ValueError("scenarios cannot be empty.")
    return {
        "survival_rate": float((scenarios["annual_return"] > 0.0).mean()),
        "median_sharpe": float(scenarios["sharpe"].median()),
        "median_max_drawdown": float(scenarios["max_drawdown"].median()),
        "profitable_scenarios": float((scenarios["total_return"] > 0.0).sum()),
        "scenario_count": float(len(scenarios)),
    }


def moving_block_bootstrap(
    returns: pd.Series,
    *,
    n_paths: int = 300,
    horizon: int = TRADING_DAYS,
    block_size: int = 20,
    seed: int = 11,
) -> np.ndarray:
    """Preserve short-run return dependence by sampling consecutive blocks."""
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < block_size:
        raise ValueError("Need at least one full block of returns.")
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, horizon), dtype=float)
    max_start = len(values) - block_size
    for path_number in range(n_paths):
        pieces: list[np.ndarray] = []
        while sum(len(piece) for piece in pieces) < horizon:
            start = int(rng.integers(0, max_start + 1))
            pieces.append(values[start : start + block_size])
        paths[path_number] = np.concatenate(pieces)[:horizon]
    return np.cumprod(1.0 + paths, axis=1)


def fit_bayesian_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    prior: str = "skeptical",
    draws: int = 750,
    tune: int = 750,
    chains: int = 2,
    seed: int = 42,
) -> az.InferenceData:
    """Fit y_t = alpha + beta*x_t + epsilon_t, epsilon_t ~ Student-t.

    `alpha` is daily, factor-adjusted alpha in decimal-return units.  The
    Student-t likelihood makes the model less brittle to fat-tailed returns.
    """
    data = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    if len(data) < 100:
        raise ValueError("Bayesian inference needs at least 100 aligned returns.")

    y = data["strategy"].to_numpy(dtype=float)
    x = data["benchmark"].to_numpy(dtype=float)
    scale = max(float(np.std(y, ddof=1)), 1e-5)
    prior_scales = {"skeptical": 0.25, "neutral": 0.75, "wide": 1.50}
    if prior not in prior_scales:
        raise ValueError(f"prior must be one of {tuple(prior_scales)}")

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=scale * prior_scales[prior])
        beta = pm.Normal("beta", mu=1.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=scale)
        nu = pm.Deterministic("nu", pm.Exponential("nu_minus_two", 1 / 10) + 2.0)
        mu = alpha + beta * x
        pm.StudentT("observed_returns", nu=nu, mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=0.92,
            random_seed=seed,
            progressbar=False,
            return_inferencedata=True,
        )
    return idata


def posterior_summary(idata: az.InferenceData) -> dict[str, float]:
    """Translate posterior draws into decision-friendly quantities."""
    alpha = idata.posterior["alpha"].values.reshape(-1)
    sigma = idata.posterior["sigma"].values.reshape(-1)
    annual_alpha = alpha * TRADING_DAYS
    annualized_sharpe = alpha / sigma * np.sqrt(TRADING_DAYS)
    diagnostics = az.summary(idata, var_names=["alpha", "beta", "sigma"], round_to=4)
    return {
        "p_alpha_positive": float((alpha > 0.0).mean()),
        "daily_alpha_mean": float(alpha.mean()),
        "annual_alpha_mean": float(annual_alpha.mean()),
        "annual_alpha_low": float(np.quantile(annual_alpha, 0.025)),
        "annual_alpha_high": float(np.quantile(annual_alpha, 0.975)),
        "p_sharpe_above_one": float((annualized_sharpe > 1.0).mean()),
        "alpha_r_hat": float(diagnostics.loc["alpha", "r_hat"]),
        "alpha_ess_bulk": float(diagnostics.loc["alpha", "ess_bulk"]),
    }


def posterior_predictive_paths(
    idata: az.InferenceData,
    benchmark_returns: pd.Series,
    *,
    horizon: int = TRADING_DAYS,
    n_paths: int = 300,
    seed: int = 99,
) -> np.ndarray:
    """Simulate future strategy equity paths, including parameter uncertainty."""
    posterior = idata.posterior
    alpha = posterior["alpha"].values.reshape(-1)
    beta = posterior["beta"].values.reshape(-1)
    sigma = posterior["sigma"].values.reshape(-1)
    nu = posterior["nu"].values.reshape(-1)
    benchmark = benchmark_returns.dropna().to_numpy(dtype=float)
    if benchmark.size == 0:
        raise ValueError("benchmark_returns cannot be empty.")
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, horizon), dtype=float)
    indices = rng.integers(0, len(alpha), size=n_paths)
    for path_number, index in enumerate(indices):
        future_market = rng.choice(benchmark, size=horizon, replace=True)
        shocks = rng.standard_t(df=nu[index], size=horizon) * sigma[index]
        future_returns = alpha[index] + beta[index] * future_market + shocks
        # A daily return cannot be less than -100%; this guard only protects
        # numerical display if a very rare heavy-tail draw is extreme.
        future_returns = np.maximum(future_returns, -0.999)
        paths[path_number] = np.cumprod(1.0 + future_returns)
    return paths

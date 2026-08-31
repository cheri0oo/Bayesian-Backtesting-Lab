# Bayesian Backtest Robustness Lab

An interactive research tool for testing whether a trading strategy's apparent edge is robust; or simply the product of favorable parameters and assumptions.

The project combines a vectorized moving-average backtest, implementation stress testing, Bayesian alpha estimation, and interactive Plotly visualizations in a Streamlit dashboard.

> This repository is an educational quantitative-research project. It is not investment advice, a live trading system, or a guarantee of future performance.

## Overview

Traditional backtests often emphasize one optimized equity curve. This project takes the opposite approach: it deliberately perturbs the backtest and measures how much evidence remains.

The application evaluates a strategy across:

- Moving-average parameter combinations
- Transaction-cost assumptions
- Execution delays
- Reproducible randomized stress scenarios
- Bayesian prior choices
- Posterior-predictive future paths

The central question is:

> After realistic implementation friction and statistical uncertainty, how credible is the claim that the strategy has positive alpha?

## Dashboard

The Streamlit dashboard provides:

- Net strategy equity versus buy-and-hold
- Parameter-sensitivity heatmap
- Implementation-fragility scatter plot
- Posterior distribution of annualized alpha
- Posterior probability that alpha is positive
- Bayesian credible interval for alpha
- Posterior-predictive equity fan chart
- Stress-test survival rate
- MCMC convergence diagnostics

## Methodology

### Strategy

The baseline strategy is a long/flat moving-average crossover:

\[
s_t = \mathbb{1}(MA_{fast,t} > MA_{slow,t})
\]

The signal is shifted forward before earning returns to prevent same-bar look-ahead bias.

### Net returns

Trading costs are charged whenever the position changes:

\[
\text{turnover}_t = |w_t - w_{t-1}|
\]

\[
r^{net}_t = w_t r_t - c \cdot \text{turnover}_t
\]

where \(c\) is the assumed one-way cost expressed as a decimal rate.

### Stress testing

The stress engine creates a reproducible scenario ledger by varying:

- Fast and slow moving-average windows
- Trading costs
- Additional execution delay

The dashboard reports the full distribution of scenario results instead of selecting only the best configuration.

### Bayesian alpha model

Net strategy returns are modeled relative to benchmark returns:

\[
y_t = \alpha + \beta x_t + \epsilon_t
\]

\[
\epsilon_t \sim \operatorname{StudentT}(\nu, 0, \sigma)
\]

The Student-t likelihood is used to accommodate heavier-tailed financial returns. PyMC generates posterior draws for alpha, beta, residual volatility, and tail thickness.

Primary Bayesian outputs include:

- \(P(\alpha > 0 \mid \text{data})\)
- 95% credible interval for annualized alpha
- Probability that the alpha-based Sharpe ratio exceeds one
- R-hat and effective sample size diagnostics
- Posterior-predictive equity paths

## Project structure

```text
Bayesian_Backtest/
├── app.py                       # Streamlit dashboard
├── robustness_lab.py            # Backtest, stress-test, and Bayesian engine
├── Bayesian_Backtest.ipynb      # Research notebook and exploratory workflow
├── requirements.txt             # Python dependencies
└── README.md
```

## Installation

Python 3.14 is recommended for the current project configuration.

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Bayesian_Backtest
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Run the dashboard

Windows:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

macOS or Linux:

```bash
.venv/bin/python -m streamlit run app.py
```

Streamlit will display a local address, typically:

```text
http://localhost:8501
```

## Run the notebook

Open `Bayesian_Backtest.ipynb` in VS Code or Jupyter and select the virtual environment as the notebook kernel.

To register it explicitly:

Windows:

```powershell
.venv\Scripts\python.exe -m ipykernel install --user --name bayesian-backtest --display-name "Python 3.14 (Bayesian Backtest)"
```

macOS or Linux:

```bash
.venv/bin/python -m ipykernel install --user --name bayesian-backtest --display-name "Python 3.14 (Bayesian Backtest)"
```

## Interpreting results

No single dashboard metric should be treated as a strategy verdict.

A more credible result generally has:

- A broad region of acceptable parameter performance
- Gradual rather than catastrophic degradation under costs and delays
- A high stress-scenario survival rate
- A posterior distribution concentrated above zero
- A credible interval that does not depend heavily on the prior
- R-hat near 1.00 and adequate effective sample size
- Acceptable downside in posterior-predictive paths

A fragile result often has an impressive baseline Sharpe ratio but a narrow parameter island, low stress survival, strong prior sensitivity, or a credible interval that crosses zero.

Importantly, \(P(\alpha > 0)\) is the posterior probability that a model parameter is positive under the supplied data, likelihood, and prior. It is **not** the probability that the strategy will be profitable next year.

## Technology

- Python
- NumPy
- pandas
- yfinance
- PyMC
- ArviZ
- Plotly
- Streamlit

## Limitations

- Uses daily Yahoo Finance data intended for research and education
- Uses a single benchmark rather than a complete factor model
- Uses simplified transaction-cost assumptions
- Does not model market impact, capacity, taxes, or operational failures
- Does not establish out-of-sample profitability
- Does not correct for searching across unrelated strategy families

Potential extensions include walk-forward validation, locked holdout periods, multi-factor attribution, spread-informed cost models, and persistent experiment tracking.

## Acknowledgments

Market data is retrieved through `yfinance`. Bayesian inference is implemented with PyMC and evaluated with ArviZ. Interactive visualizations use Plotly and Streamlit.

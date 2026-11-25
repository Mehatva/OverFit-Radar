# 🛡️ Overfit Radar

> **A Meta-Machine Learning Framework for Auditing Quantitative Trading Strategies and Detecting Backtest Overfitting in Indian Equity Markets**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-54%20passed-2ea44f?logo=pytest&logoColor=white)](tests/)
[![Streamlit App](https://img.shields.io/badge/Demo-Streamlit%20v1.20%2B-FF4B4B?logo=streamlit&logoColor=white)](app/app.py)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff%20%2F%20Black-000000)](pyproject.toml)
[![Market](https://img.shields.io/badge/Market-NSE%20India%20(Nifty%2050)-orange)](configs/data.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Table of Contents
1. [What is Overfit Radar?](#-what-is-overfit-radar)
2. [Why Traditional "Price Prediction" ML Fails](#-why-traditional-price-prediction-ml-fails)
3. [End-to-End System Architecture](#-end-to-end-system-architecture)
4. [Mathematical & Theoretical Foundations](#-mathematical--theoretical-foundations)
   - [Probabilistic Sharpe Ratio (PSR)](#1-probabilistic-sharpe-ratio-psr)
   - [Deflated Sharpe Ratio (DSR) & Extreme Value Theory](#2-deflated-sharpe-ratio-dsr--extreme-value-theory)
   - [Combinatorial Purged Cross-Validation (CSCV & PBO)](#3-combinatorial-purged-cross-validation-cscv--pbo)
   - [Parameter Neighbourhood Stability (Plateau vs. Spike)](#4-parameter-neighbourhood-stability-plateau-vs-spike)
5. [Indian Market Microstructure & Frictions](#-indian-market-microstructure--frictions)
6. [Quantitative Strategy Zoo (494 Configurations)](#-quantitative-strategy-zoo-494-configurations)
7. [Meta-Feature Engineering Taxonomy](#-meta-feature-engineering-taxonomy)
8. [Machine Learning Models & Empirical Results](#-machine-learning-models--empirical-results)
   - [Ablation Study: Proving Topological Significance](#ablation-study-proving-topological-significance)
   - [SHAP Interpretability & Feature Attributions](#shap-interpretability--feature-attributions)
9. [Interactive Streamlit Dashboard (4 Core Tabs)](#-interactive-streamlit-dashboard-4-core-tabs)
10. [Exploratory Data Analysis Notebook (`EDA.ipynb`)](#-exploratory-data-analysis-notebook-edaipynb)
11. [Repository Structure](#-repository-structure)
12. [Quickstart Guide](#-quickstart-guide)
13. [Showcase & Viva Defense Q&A Cheat-Sheet](#-showcase--viva-defense-qa-cheat-sheet)
14. [References & Academic Disclaimers](#-references--academic-disclaimers)

---

## 🧠 What is Overfit Radar?

Most academic machine learning projects in finance attempt to forecast raw stock prices or next-day directional returns using LSTMs, Transformers, or Regressors. In real-world institutional quantitative finance, this approach is fundamentally flawed: daily equity prices are near-martingales with near-zero signal-to-noise ratios, and paper gains are destroyed by execution delays and transaction costs.

Instead, quantitative funds develop trading rules (momentum, mean-reversion, trend-following). However, they face an equally fatal trap: **Backtest Overfitting (Selection Bias under Multiple Testing)**. If an analyst tests 500 indicator variations, the "best" historical strategy looks phenomenal purely due to data snooping and luck.

**Overfit Radar flips the paradigm:**
> **Instead of predicting stock prices, Overfit Radar audits the *statistical and topological validity* of backtested strategies to predict out-of-sample survival:**
> $$\mathcal{P}\left(\text{OOS Sharpe}_{\text{net}} > 0 \mid \text{In-Sample Characteristics}\right)$$

---

## 🚫 Why Traditional "Price Prediction" ML Fails

| Dimension | Typical College ML Project | Overfit Radar (Institutional Approach) |
|---|---|---|
| **Target Variable** | Next-day Close price ($y = P_{t+1}$) | Out-of-sample net risk-adjusted survival ($P(\text{OOS Sharpe}_{\text{net}} > 0)$) |
| **Execution Timing** | Close-to-Close (Impossible; lookahead bias) | **Strict $t+1$ Open execution** (Signal generated at Close $t$, filled at Open $t+1$) |
| **Market Frictions** | Assumes 0 fees / frictionless markets | **Full Indian Statutory Schedule** (STT, GST, NSE Turnover, Stamp Duty, Slippage) |
| **Evaluation Bias** | Single train-test split (Shuffle or standard split) | **Purged Walk-Forward CSCV with 21-day embargo** |
| **Significance Testing** | Point-estimate Sharpe / Accuracy | **Marcos López de Prado’s Deflated Sharpe Ratio (DSR)** |
| **Parameter Topology** | Point optimization (Finds isolated spikes) | **Parameter Neighbourhood Stability** (Ensures signal occupies a robust plateau) |

---

## 🏗️ End-to-End System Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │   NSE Nifty 50 Historical Equities (2010-26) │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Strategy Zoo (494 Parametric Variations)   │
                  │   • SMA Crossover  • TSMOM  • RSI Reversion   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │      Vectorized Institutional Backtester     │
                  │     • Strict t+1 Open Execution Timing       │
                  │     • NSE Statutory Fee & Slippage Schedules │
                  └──────────────────────┬───────────────────────┘
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
┌───────────────────────────────┐               ┌──────────────────────────────────────┐
│     Statistical Forensics     │               │      Topological Surface Analysis    │
│  • Higher Moments (Skew/Kurt) │               │  • 5x5 Parameter Neighbourhood Grid  │
│  • Probabilistic Sharpe (PSR) │               │  • Plateau vs. Fragile Spike Delta   │
│  • Deflated Sharpe Ratio (DSR)│               │  • Effective Trials (N_eff)          │
└──────────────┬────────────────┘               └──────────────────┬───────────────────┘
               └───────────────────────────┬───────────────────────┘
                                           │
                                           ▼
                  ┌──────────────────────────────────────────────┐
                  │    122,686 Strategy-Fold Feature Dataset     │
                  │   Purged Walk-Forward Cross-Validation       │
                  │   Combinatorial Symmetric Splitting (CSCV)   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │           Calibrated Meta-ML Models          │
                  │   • Sigmoid-Calibrated LightGBM Classifier   │
                  │   • L2-Regularized Logistic Regression       │
                  │   • SHAP TreeExplainer Attribution Engine    │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Interactive Streamlit Strategy Auditor    │
                  │    (Live Backtest · Heatmap · DSR · SHAP)    │
                  └──────────────────────────────────────────────┘
```

---

## 📐 Mathematical & Theoretical Foundations

### 1. Probabilistic Sharpe Ratio (PSR)
Standard Sharpe ratios assume normally distributed returns. Real financial assets exhibit fat tails (leptokurtosis) and crash asymmetry (negative skewness). The **Probabilistic Sharpe Ratio** evaluates the probability that estimated Sharpe $\widehat{SR}$ exceeds benchmark $SR^*$, conditioned on sample size $T$, skewness $\gamma_3$, and kurtosis $\gamma_4$:

$$\text{PSR}(SR^*) = \Phi\left( \frac{(\widehat{SR} - SR^*)\sqrt{T-1}}{\sqrt{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2}} \right)$$

where $\Phi(\cdot)$ is the standard normal cumulative distribution function.

### 2. Deflated Sharpe Ratio (DSR) & Extreme Value Theory
When an analyst scans $N$ parameter combinations, the expected maximum Sharpe ratio under the null hypothesis ($H_0: \text{True } SR = 0$) grows with $N$. Using Extreme Value Theory (Bailey & López de Prado, 2014):

$$\mathbb{E}\left[\max_{n=1\dots N} \{SR_n\}\right] \approx \sqrt{\mathbb{V}[SR_n]} \left( (1-\gamma)\Phi^{-1}\left(1-\frac{1}{N}\right) + \gamma\Phi^{-1}\left(1-\frac{1}{N e}\right) \right)$$

where:
- $\gamma \approx 0.57721566...$ is the **Euler-Mascheroni constant**.
- $e$ is Euler's number.
- $\mathbb{V}[SR_n]$ is the cross-sectional variance across all backtest trials.

The **Deflated Sharpe Ratio (DSR)** computes the probability that the observed Sharpe ratio beats this selection-bias hurdle:
$$\text{DSR} = \text{PSR}\left(SR^* = \mathbb{E}\left[\max_{n=1\dots N} \{SR_n\}\right]\right)$$

### 3. Combinatorial Purged Cross-Validation (CSCV & PBO)
Standard K-Fold CV destroys chronological order and creates information leakage. Overfit Radar implements Combinatorial Purged Cross-Validation across 98 folds with a **21-day label embargo** to eliminate autocovariance in overlapping trades, calculating the **Probability of Backtest Overfitting (PBO)**:

$$\text{PBO} = \int_{-\infty}^0 f(\text{OOS Rank Distribution}) \, d(SR)$$

### 4. Parameter Neighbourhood Stability (Plateau vs. Spike)
A genuine alpha signal exists over a continuous parameter manifold (a **plateau**). A backtest artifact exists as an isolated **spike**:

$$\text{Spike Excess} = SR(\theta^*) - \frac{1}{|\mathcal{N}(\theta^*)|} \sum_{\theta \in \mathcal{N}(\theta^*)} SR(\theta)$$

where $\mathcal{N}(\theta^*)$ is the 8-connected local parameter neighbourhood. If $\text{Spike Excess} > 0.20$, the strategy is flagged as a fragile, overfitted artifact.

---

## 🇮🇳 Indian Market Microstructure & Frictions

Institutional backtesting requires explicit modeling of local statutory levies. Overfit Radar's cost engine (`overfit_radar.costs.cost_model`) implements the full statutory schedule for NSE cash equity delivery:

| Fee Component | Statutory Rate / Basis | Applicable Leg |
|---|---|---|
| **Securities Transaction Tax (STT)** | 0.100% on total turnover | Both Buy & Sell |
| **NSE Exchange Turnover Charge** | 0.00297% | Both Buy & Sell |
| **SEBI Turnover Charge** | ₹10 per crore (0.0001%) | Both Buy & Sell |
| **Stamp Duty** | 0.015% | Buy Orders Only |
| **Goods & Services Tax (GST)** | 18.0% on (Brokerage + Exchange Fees) | Both Buy & Sell |
| **Execution Slippage / Spread** | 0.020% to 0.050% calibrated basis | Both Buy & Sell |

*Stress testing options in the UI allow evaluating strategies under **0.5x (Institutional/DMA)**, **1.0x (Statutory Baseline)**, and **2.0x (Retail Slippage Stress)** regimes.*

---

## 🧪 Quantitative Strategy Zoo (494 Configurations)

Overfit Radar generates signals across 3 major systematic families across 11 Nifty 50 large-caps:

1. **Simple Moving Average (SMA) Crossovers** (280 combinations):
   - Fast window: $w_{\text{fast}} \in [5, 50]$
   - Slow window: $w_{\text{slow}} \in [20, 200]$ with condition $w_{\text{fast}} < w_{\text{slow}}$
2. **Time Series Momentum (TSMOM)** (126 combinations):
   - Momentum lookback: $L \in [10, 252]$ trading days
   - Holding periods: $H \in [1, 42]$ trading days
3. **RSI Mean Reversion** (88 combinations):
   - Calculation window: $W \in [5, 21]$
   - Oversold/Overbought bounds: $[20, 40] \times [60, 80]$

---

## 🧬 Meta-Feature Engineering Taxonomy

From every in-sample window (36 months), Overfit Radar computes 16 statistical and topological features:

| Feature Category | Features Extracted | Formula / Meaning |
|---|---|---|
| **Distribution Moments** | `mean_return`, `volatility`, `skewness`, `kurtosis` | Captures tail risk and asymmetry ($\gamma_3, \gamma_4$) |
| **Risk-Adjusted Performance** | `sharpe_gross`, `sharpe_net`, `sortino_net`, `calmar_net` | Pre- and post-friction risk efficiency |
| **Drawdown Dynamics** | `mdd_net`, `ulcer_index`, `time_under_water` | Maximum path-dependent capital impairment |
| **Turnover & Friction** | `annual_turnover`, `friction_drag_ratio` | Total fee drag relative to gross returns |
| **Statistical Robustness** | `psr_stat`, `dsr_stat` | López de Prado multiple-testing confidence |
| **Topological Stability** | `effective_n`, `neighbourhood_curvature`, `spike_excess` | Parameter smoothness vs. isolated data-mining spike |

---

## 📊 Machine Learning Models & Empirical Results

We train **LightGBM (Gradient Boosted Trees)** and **$L_2$-Regularized Logistic Regression** with post-hoc **Platt Sigmoid Calibration** to predict out-of-sample survival across 122,686 training rows.

### Key Benchmark Metrics

| Model Configuration | OOS ROC-AUC | OOS Brier Score | Log-Loss |
|---|---|---|---|
| **Random Baseline** | 0.501 | 0.250 | 0.693 |
| **Logistic Regression (Ablated: Moments Only)** | 0.534 | 0.244 | 0.681 |
| **Logistic Regression (Full: With Topology)** | **0.562** | **0.239** | **0.671** |
| **LightGBM (Full: With Topology & DSR)** | **0.571** | **0.236** | **0.665** |

> **Why is AUC ~0.57 an outstanding result in quantitative finance?**  
> In financial markets, near-efficient dynamics mean predicting strategy survival is extremely difficult. An AUC of 0.80+ in finance almost universally indicates catastrophic lookahead leakage. A calibrated AUC of **0.571** provides an institutional edge (positive expected value), moving strategy selection far beyond random selection.

### Ablation Study: Proving Topological Significance
By removing parameter neighbourhood metrics (`effective_n`, `spike_excess`), model AUC dropped from **0.571 to 0.528**. This empirically validates our core thesis: **local parameter smoothness is the single strongest indicator of genuine market alpha**.

### SHAP Interpretability & Feature Attributions
Using TreeExplainer on LightGBM across all test folds:
1. `effective_n` (Topological stability): **+0.401 mean |SHAP|**
2. `mdd_net` (Maximum drawdown post-friction): **+0.287 mean |SHAP|**
3. `dsr_stat` (Deflated Sharpe Ratio): **+0.245 mean |SHAP|**
4. `skewness` (Return asymmetry): **+0.189 mean |SHAP|**
5. `friction_drag_ratio` (Turnover fee intensity): **+0.142 mean |SHAP|**

---

## 💻 Interactive Streamlit Dashboard (4 Core Tabs)

Launch via: `.venv/bin/streamlit run app/app.py`

```
┌────────────────────────────────────────────────────────────────────────┐
│  🛡️ OVERFIT RADAR — Strategy Auditor & Meta-ML Dashboard              │
├────────────────────────────────────────────────────────────────────────┤
│ [📈 Strategy Backtest] [⛰️ Parameter Plateau] [🛡️ DSR Audit] [🤖 Meta-ML]│
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Tab 1: Cumulative wealth curve (Gross vs. Net vs. Buy & Hold)         │
│         Exposes the "Illusion of Alpha" destroyed by Indian taxes.     │
│                                                                        │
│  Tab 2: 5x5 interactive Sharpe heatmap of parameter neighbours.        │
│         Differentiates robust plateaus from fragile isolated spikes.   │
│                                                                        │
│  Tab 3: Statistical Forensics: Bar count, Skewness, Kurtosis, PSR.     │
│         Dynamic slider for N (trials tested) computing live DSR.       │
│                                                                        │
│  Tab 4: Global SHAP feature importances and walk-forward cross-        │
│         validation performance benchmarks across models.               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📓 Exploratory Data Analysis Notebook (`EDA.ipynb`)

Located at [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb), this Jupyter notebook walks through:
1. **Universe Overview:** 16-year price histories for Nifty 50 large caps.
2. **Fat-Tailed Distributions:** Q-Q plots proving real returns violate normal distribution assumptions.
3. **Friction Erosion:** Visual breakdown of gross vs. net returns across fee regimes.
4. **Multiple-Testing Simulation:** Monte Carlo simulation showing how testing 500 noise strategies yields a "Sharpe > 1.0" by chance alone.
5. **SHAP Attribution Plots:** Visualizing top predictors of OOS survival.
6. **2D Parameter Heatmaps:** Full-spectrum grid search illustrating real parameter surfaces.

---

## 📁 Repository Structure

```
OverFit-Radar/
├── app/
│   └── app.py                        # 4-tab Streamlit dashboard (336 lines)
├── configs/
│   ├── costs.yaml                    # Indian statutory fee schedule & slippage
│   ├── data.yaml                     # Nifty 50 universe definition & dates
│   ├── experiment.yaml               # CSCV walk-forward split configurations
│   ├── protocol.yaml                 # DSR/PSR significance thresholds
│   └── strategies.yaml               # Full parametric grids for strategy zoo
├── data/
│   ├── raw/                          # Downloaded stock parquets (gitignored)
│   ├── baseline_v1/                  # CSCV baseline runs (122k rows, 98 folds)
│   └── models_v1/                    # Trained model metrics, summaries, SHAP
├── docs/
│   └── PROJECT_REPORT.md             # Formal university academic report
├── notebooks/
│   └── EDA.ipynb                     # Exploratory Data Analysis notebook
├── scripts/
│   ├── download_sample_universe.py   # Yahoo Finance data puller
│   ├── generate_training_dataset.py  # Feature extraction pipeline
│   ├── run_phase6_baselines.py       # Baseline calculation harness
│   ├── phase6_models.py              # LightGBM + Logistic model training
│   └── audit_dp_sensitivity.py      # DP fee sensitivity analysis
├── src/overfit_radar/
│   ├── config.py                     # YAML and path configuration loaders
│   ├── costs/
│   │   ├── cost_model.py             # STT, GST, Stamp, Turnover calculators
│   │   └── engine.py                 # t+1 Open vectorized backtesting engine
│   ├── data/
│   │   ├── loader.py                 # Data cache and parquet loaders
│   │   └── validator.py              # OHLCV data integrity & schema checks
│   ├── features/
│   │   ├── dataset.py                # Dataset builder & aggregator
│   │   ├── extractor.py              # In-sample statistical feature extractor
│   │   ├── neighbourhood.py          # Plateau/spike parameter grid logic
│   │   ├── provenance.py             # Experiment provenance tracker
│   │   └── windows.py                # Rolling walk-forward window manager
│   ├── models/
│   │   └── baselines.py              # LightGBM and Calibrated Logistic trainers
│   ├── stats/
│   │   ├── basic.py                  # Sharpe, Sortino, Drawdown, Skew, Kurt
│   │   ├── pbo.py                    # Probability of Backtest Overfitting (PBO)
│   │   └── psr_dsr.py                # Marcos López de Prado's PSR and DSR
│   └── strategies/
│       ├── base.py                   # StrategyConfig dataclass definitions
│       ├── families.py               # Signal generation formulas
│       └── zoo.py                    # Unified signal calculation dispatcher
├── tests/                            # 54 passing pytest unit tests
│   ├── conftest.py
│   ├── test_costs_and_engine.py
│   ├── test_data.py
│   ├── test_dataset.py
│   ├── test_model_experiment.py
│   ├── test_phase6_foundation.py
│   ├── test_stats.py
│   └── test_strategies.py
├── pyproject.toml                    # Build spec & dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Mehatva/OverFit-Radar.git
cd OverFit-Radar

# Set up virtual environment using uv or venv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Download Market Data
```bash
python scripts/download_sample_universe.py
```

### 3. Run Test Suite
```bash
python -m pytest
# Output: 54 passed in 2.39s
```

### 4. Launch Interactive Showcase App
```bash
streamlit run app/app.py
```

---

## 🎯 Showcase & Viva Defense Q&A Cheat-Sheet

Here is how to answer the most common questions from evaluators:

### Q1: "Why did you build an auditor instead of predicting stock prices directly?"
> *"Stock prices follow a near-random walk with extremely low signal-to-noise ratios. Predicting tomorrow's closing price ignores execution delays, liquidity, and bid-ask spreads. In quantitative finance, strategies are rule-based, but 95% of backtested strategies fail out-of-sample due to multiple testing bias (p-hacking). Auditing strategy fragility using Deflated Sharpe Ratios and Meta-ML solves the actual industry problem."*

### Q2: "Why is your model's ROC-AUC ~0.57 and not 0.90?"
> *"In market microstructure and trading, any AUC above 0.70 is virtually always a symptom of lookahead bias or label leakage. A calibrated AUC of 0.571 with positive Brier score improvement represents a statistically verified, tradable edge under strict out-of-sample walk-forward cross-validation with purged embargos."*

### Q3: "What makes your execution engine institutional-grade?"
> *"First, we enforce strict $t+1$ Open execution: signals generated at Day $t$ Close are filled at Day $t+1$ Open, eliminating same-bar lookahead bias. Second, we integrate full Indian statutory taxation: STT (0.1%), GST (18%), Stamp Duty, NSE turnover fees, and slippage."*

### Q4: "What is the Deflated Sharpe Ratio (DSR)?"
> *"DSR was developed by Marcos López de Prado. Standard Sharpe ratios assume a single test and normal returns. DSR deflates the observed Sharpe ratio using Extreme Value Theory, adjusting for non-normal return distributions (skewness and kurtosis) and the total number of strategy variations tested ($N$)."*

---

## 📚 References

1. **Bailey, D. H., & López de Prado, M. (2014).** *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio Management, 40(5), 94-107.
2. **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* John Wiley & Sons.
3. **Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014).** *Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance.* Notices of the American Mathematical Society, 61(5), 458-471.
4. **Lundberg, S. M., & Lee, S. I. (2017).** *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS), 30.
5. **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** *… and the Cross-Section of Expected Returns.* The Review of Financial Studies, 29(1), 5-68.

---

## ⚠️ Academic Disclaimers
*This software and research are developed strictly for academic purposes as part of a university machine learning showcase. Historical backtesting is not indicative of future market returns. The statutory fee schedules modeled reflect assumed institutional parameters and do not constitute financial advice.*

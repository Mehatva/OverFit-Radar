# 🛡️ Overfit Radar

> **A Meta-Machine Learning Framework for Auditing Quantitative Trading Strategies and Detecting Backtest Overfitting in Indian Equity Markets**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/Tests-54%20passing-brightgreen)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What is Overfit Radar?

Most "stock prediction" ML projects share the same fatal flaw: they test hundreds of parameters, pick the one with the best historical chart, and call it a strategy. That is not alpha — it is **data snooping bias**.

**Overfit Radar flips the paradigm.** Instead of predicting stock prices, it audits the *statistical robustness* of backtested strategies. It answers one question:

> **Given what I observed in-sample, what is the probability this strategy actually works out-of-sample?**

Built specifically for **NSE Nifty 50 large-cap equities**, it implements the same institutional-grade forensic tools used by quantitative hedge funds:

| Component | Method |
|---|---|
| Multiple-testing correction | Deflated Sharpe Ratio (DSR) — López de Prado 2014 |
| Cross-validation for trading | Combinatorial Symmetric CV (CSCV) — PBO metric |
| Statistical Sharpe confidence | Probabilistic Sharpe Ratio (PSR) |
| Parameter robustness | Neighbourhood Stability — plateau vs. spike detection |
| Indian market costs | Full statutory schedule: STT, GST, NSE fees, Stamp Duty |
| ML survival prediction | LightGBM + Calibrated Logistic Regression + SHAP |

---

## Architecture

```
Raw Strategy Zoo (494 Configurations)
        │
        ▼
Vectorized Backtest Engine (t+1 Open — zero lookahead)
        │
        ▼
Indian Cost Schedule (STT · GST · NSE Turnover · Stamp Duty)
        │
        ├──► Statistical Forensics
        │       ├── Probabilistic Sharpe Ratio (PSR)
        │       ├── Deflated Sharpe Ratio (DSR)
        │       ├── Parameter Neighbourhood Stability
        │       └── Higher Moments (Skewness & Kurtosis)
        │
        └──► Purged Walk-Forward Cross-Validation (21-day embargo)
                ├── Calibrated LightGBM  ──► SHAP Attribution
                └── L2 Logistic Regression
                        │
                        ▼
            P(OOS Sharpe_net > 0) — Out-of-Sample Survival Probability
```

---

## 🚀 Quickstart

### Prerequisites

- Python 3.9+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### 1. Clone & Install

```bash
git clone https://github.com/Mehatva/OverFit-Radar.git
cd OverFit-Radar

# Create virtual environment and install all dependencies
uv venv --python 3.12
uv pip install -e ".[dev]"
```

### 2. Download Market Data

```bash
# Downloads Nifty 50 universe OHLCV from yfinance (2010–2026)
.venv/bin/python scripts/download_sample_universe.py
```

### 3. Launch the Demo Dashboard

```bash
.venv/bin/streamlit run app/app.py
```

> Pre-trained model artifacts and baseline results are already included — the app works immediately after install + data download.

### 4. Run Tests

```bash
.venv/bin/python -m pytest          # 54 tests, ~2.5s
```

---

## 📊 Interactive Demo — 4 Tabs

### Tab 1 · Strategy Backtest & Execution
Live equity curve comparison between **Gross (zero-cost)** and **Net (post-friction)** strategy returns against a Buy & Hold benchmark. Demonstrates how Indian statutory costs systematically eliminate apparent alpha.

### Tab 2 · Parameter Plateau vs. Spike  
5×5 Sharpe heatmap across the **neighbourhood of your chosen parameters**. A genuine signal occupies a smooth plateau; a data-mined spike collapses at adjacent parameter values.

### Tab 3 · DSR & Overfit Audit  
Computes **PSR** (return distribution-adjusted confidence) and **DSR** (multiple-testing deflation). Adjust the number of strategy variations tested (N) and watch the survival probability fall in real time.

### Tab 4 · Meta-ML Benchmarks  
SHAP feature importance from the trained LightGBM model — which in-sample characteristics most predict out-of-sample survival. Calibrated AUC scores vs. a random baseline.

---

## 📁 Project Structure

```
OverFit-Radar/
├── app/
│   └── app.py                   # Streamlit demo (4 tabs)
├── configs/
│   ├── costs.yaml               # Indian statutory fee schedule
│   ├── strategies.yaml          # Strategy parameter grids
│   ├── data.yaml                # Universe and date range
│   ├── experiment.yaml          # CSCV fold configuration
│   └── protocol.yaml            # DSR/PSR benchmark parameters
├── data/
│   ├── raw/                     # Downloaded OHLCV parquets (gitignored)
│   ├── baseline_v1/             # CSCV baseline metrics (122,686 rows, 98 folds)
│   └── models_v1/               # ML experiment results & SHAP outputs
├── docs/
│   └── PROJECT_REPORT.md        # Full formal report with math & viva Q&A
├── notebooks/
│   └── EDA.ipynb                # Exploratory data analysis walkthrough
├── scripts/
│   ├── download_sample_universe.py
│   ├── generate_training_dataset.py
│   ├── run_phase6_baselines.py
│   └── phase6_models.py
├── src/overfit_radar/
│   ├── config.py
│   ├── costs/                   # Cost model & backtest engine
│   ├── data/                    # Loader & validator
│   ├── features/                # Feature extraction & CSCV
│   ├── models/                  # LightGBM & logistic baselines
│   ├── stats/                   # PSR, DSR, PBO, Sharpe, drawdown
│   └── strategies/              # SMA, TSMOM, RSI signal generators
└── tests/                       # 54 pytest tests
```

---

## 📐 Key Results (Nifty 50, 2010–2026)

| Metric | Value |
|---|---|
| Universe | 11 Nifty 50 large-caps |
| CSCV Folds | 98 walk-forward windows |
| Training rows | 122,686 (strategy × fold combinations) |
| LightGBM AUC (OOS) | 0.571 |
| Calibrated Logistic AUC | 0.562 |
| Random baseline AUC | 0.501 |
| Top SHAP predictor | `effective_n` (parameter neighbourhood stability) — mean SHAP 0.401 |
| DSR (SMA 15/50 on BHARTIARTL) | 14.7% — **fails** multiple-testing hurdle at N=494 trials |

> The modest AUC lift above random is **the expected result** — if strategies were trivially predictable, markets would already be efficient at a different level. The value is in the *calibrated probability*, not in predicting a winner.

---

## 🏗️ Pipeline Execution Order

```bash
# Step 1 — Download data
.venv/bin/python scripts/download_sample_universe.py

# Step 2 — Generate CSCV training dataset
.venv/bin/python scripts/generate_training_dataset.py

# Step 3 — Compute baselines (Sharpe/DSR across 98 folds)
.venv/bin/python scripts/run_phase6_baselines.py --output data/baseline_v1

# Step 4 — Train ML models + SHAP
.venv/bin/python scripts/phase6_models.py
```

---

## 📚 References

- Bailey, D.H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality.* Journal of Portfolio Management.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Bailey, D.H. et al. (2014). *Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance.* Notices of the AMS.
- Lundberg, S.M. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS.

---

## ⚠️ Limitations & Disclaimers

- **Static universe** — Nifty 50 survivorship bias; delisted stocks excluded
- **Approximate costs** — based on publicly available statutory schedules; actual execution costs vary
- **Correlated trials** — CSCV partially accounts for this; regime dependence remains
- **Not investment advice** — this is an academic research project

---

*College ML Project Showcase — November 2026*

"""Overfit Radar - Interactive Strategy Auditor & Meta-ML Demonstration Dashboard.

Streamlit application demonstrating backtesting, Indian transaction costs,
Deflated Sharpe Ratio (DSR), Parameter Neighbourhood Stability, and ML survival prediction.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from overfit_radar.config import get_project_root, load_costs_config, load_yaml
from overfit_radar.costs.engine import BacktestEngine
from overfit_radar.stats.basic import (
    calculate_sharpe,
    calculate_sortino,
    calculate_max_drawdown,
    calculate_skewness,
    calculate_kurtosis,
)
from overfit_radar.stats.psr_dsr import compute_psr, compute_dsr, compute_effective_n
from overfit_radar.strategies.base import StrategyConfig
from overfit_radar.strategies.zoo import compute_strategy_signal

st.set_page_config(
    page_title="Overfit Radar | Quant Strategy ML Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = get_project_root()

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 12px;
    }
    .badge-pass {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-fail {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_available_universe():
    raw_files = sorted((ROOT / "data/raw").glob("*.parquet"))
    symbols = [p.stem.replace("_NS", ".NS") for p in raw_files]
    return symbols

@st.cache_data
def load_symbol_data(symbol: str):
    clean_sym = symbol.replace(".NS", "_NS")
    path = ROOT / f"data/raw/{clean_sym}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

@st.cache_data
def load_model_summary():
    p = ROOT / "data/models_v1/summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return None

@st.cache_data
def load_shap_importance():
    p = ROOT / "data/models_v1/shap_importance.json"
    if p.exists():
        return pd.read_json(p)
    return None

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("## 🛡️")
st.sidebar.title("Overfit Radar")
st.sidebar.caption("Institutional Quantitative Strategy Auditor")

symbols = get_available_universe()
selected_symbol = st.sidebar.selectbox("Target Nifty Stock", symbols, index=0)

strategy_family = st.sidebar.selectbox(
    "Strategy Family",
    ["SMA Crossover", "Time Series Momentum", "RSI Mean Reversion"]
)

# Dynamic Parameters
if strategy_family == "SMA Crossover":
    fast_window = st.sidebar.slider("Fast Window (Days)", 5, 50, 15)
    slow_window = st.sidebar.slider("Slow Window (Days)", 20, 200, 50)
    strat_cfg = StrategyConfig(
        strategy_id=f"sma_{fast_window}_{slow_window}",
        family="sma_crossover",
        params={"fast_window": fast_window, "slow_window": slow_window}
    )
elif strategy_family == "Time Series Momentum":
    lookback = st.sidebar.slider("Lookback (Days)", 10, 252, 63)
    holding_period = st.sidebar.slider("Holding Period (Days)", 1, 42, 5)
    strat_cfg = StrategyConfig(
        strategy_id=f"tsmom_{lookback}_{holding_period}",
        family="time_series_momentum",
        params={"lookback": lookback, "holding_period": holding_period}
    )
else:
    rsi_window = st.sidebar.slider("RSI Window (Days)", 5, 21, 14)
    oversold = st.sidebar.slider("Oversold Level", 20, 40, 30)
    overbought = st.sidebar.slider("Overbought Level", 60, 80, 70)
    strat_cfg = StrategyConfig(
        strategy_id=f"rsi_{rsi_window}_{oversold}_{overbought}",
        family="rsi_mean_reversion",
        params={"window": rsi_window, "oversold": oversold, "overbought": overbought}
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Indian Market Frictions")
cost_level = st.sidebar.select_slider(
    "Assumed Cost Regime",
    options=["0.5x (Institutional/Direct)", "1.0x (Baseline Statutory)", "2.0x (Retail Slippage Stress)"],
    value="1.0x (Baseline Statutory)"
)
cost_mult = 0.5 if "0.5x" in cost_level else (2.0 if "2.0x" in cost_level else 1.0)

# --- MAIN PAGE TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Strategy Backtest & Execution",
    "⛰️ Parameter Plateau vs. Spike",
    "🛡️ DSR & Overfit Audit",
    "🤖 Meta-ML Research Benchmarks"
])

prices_df = load_symbol_data(selected_symbol)

if prices_df is None or prices_df.empty:
    st.error(f"Price history not found for {selected_symbol}.")
    st.stop()

costs_config = load_costs_config()
engine = BacktestEngine(costs_config, cost_multiplier=cost_mult)
signal = compute_strategy_signal(prices_df, strat_cfg)
result = engine.run(prices_df, signal)

# --- TAB 1: STRATEGY BACKTEST & EXECUTION ---
with tab1:
    st.header(f"Strategy Performance: {strat_cfg.strategy_id} on {selected_symbol}")
    st.caption("Execution is strictly shifted to t+1 Open to prevent lookahead bias. All Indian statutory fees (STT, GST, Stamp Duty, NSE Turnover) applied.")

    cum_gross = (1.0 + result["gross_return"]).cumprod()
    cum_net = (1.0 + result["net_return"]).cumprod()
    benchmark = (1.0 + prices_df["close"].pct_change().fillna(0.0)).cumprod()

    col1, col2, col3, col4, col5 = st.columns(5)
    sr_gross = calculate_sharpe(result["gross_return"])
    sr_net = calculate_sharpe(result["net_return"])
    mdd = calculate_max_drawdown(result["net_return"])
    trades = int((result["trade_action"].abs() == 1.0).sum() // 2)
    turnover = float(result["trade_action"].abs().sum() / (len(result) / 252.0))

    col1.metric("Gross Sharpe", f"{sr_gross:.2f}")
    col2.metric("Net Sharpe (Post-Friction)", f"{sr_net:.2f}", delta=f"{sr_net - sr_gross:.2f}")
    col3.metric("Max Drawdown", f"{mdd*100:.1f}%")
    col4.metric("Completed Trades", f"{trades}")
    col5.metric("Annual Turnover", f"{turnover:.1f}x")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(cum_gross.index, cum_gross, label="Gross Strategy (Zero Cost)", color="#4CAF50", alpha=0.7)
    ax.plot(cum_net.index, cum_net, label=f"Net Strategy ({cost_mult}x Cost Friction)", color="#1976D2", linewidth=2)
    ax.plot(benchmark.index, benchmark, label="Buy & Hold Benchmark", color="#9E9E9E", linestyle="--")
    ax.set_ylabel("Cumulative Wealth (₹1 Base)")
    ax.set_title(f"Impact of Transaction Costs on {selected_symbol}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    if sr_gross > 0.5 and sr_net < 0.0:
        st.warning("⚠️ **Illusion of Alpha Detected:** This strategy appears profitable gross of fees, but is completely eliminated by Indian market transaction frictions.")

# --- TAB 2: PARAMETER PLATEAU VS. SPIKE ---
with tab2:
    st.header("Parameter Stability Surface: Detecting Overfit Spikes")
    st.write("""
    A fundamental thesis of **Overfit Radar** is that genuine market signals occupy **robust plateaus** in parameter space.
    Strategies that perform well only at an isolated point while neighbors fail are **overfitted spikes** (data mining artifacts).
    """)

    if strategy_family == "SMA Crossover":
        fast_range = [fast_window - 4, fast_window - 2, fast_window, fast_window + 2, fast_window + 4]
        slow_range = [slow_window - 10, slow_window - 5, slow_window, slow_window + 5, slow_window + 10]
        fast_range = [f for f in fast_range if f >= 5]
        slow_range = [s for s in slow_range if s > max(fast_range)]

        heatmap_data = np.zeros((len(fast_range), len(slow_range)))
        for i, f_w in enumerate(fast_range):
            for j, s_w in enumerate(slow_range):
                sub_strat = StrategyConfig(
                    strategy_id=f"sma_{f_w}_{s_w}",
                    family="sma_crossover",
                    params={"fast_window": f_w, "slow_window": s_w}
                )
                sig = compute_strategy_signal(prices_df, sub_strat)
                res = engine.run(prices_df, sig)
                heatmap_data[i, j] = calculate_sharpe(res["net_return"])

        fig, ax = plt.subplots(figsize=(8, 5))
        cax = ax.matshow(heatmap_data, cmap="coolwarm")
        fig.colorbar(cax)
        ax.set_xticks(range(len(slow_range)))
        ax.set_xticklabels(slow_range)
        ax.set_yticks(range(len(fast_range)))
        ax.set_yticklabels(fast_range)
        ax.set_xlabel("Slow Window")
        ax.set_ylabel("Fast Window")
        ax.set_title("Net Sharpe Surface Across Parameter Neighbours", pad=20)
        st.pyplot(fig)
        plt.close(fig)

        center_sr = heatmap_data[fast_range.index(fast_window), slow_range.index(slow_window)] if fast_window in fast_range and slow_window in slow_range else sr_net
        mean_sr = np.mean(heatmap_data)
        std_sr = np.std(heatmap_data)
        spike_excess = center_sr - mean_sr

        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Sharpe", f"{center_sr:.2f}")
        col2.metric("Neighbourhood Mean Sharpe", f"{mean_sr:.2f}")
        col3.metric("Spike Excess (Fragility)", f"{spike_excess:.2f}",
                    delta="Plateau" if spike_excess < 0.2 else "Fragile Spike",
                    delta_color="normal" if spike_excess < 0.2 else "inverse")
    else:
        st.info("Parameter surface visualization is implemented for SMA Crossover grids.")

# --- TAB 3: DSR & OVERFIT AUDIT ---
with tab3:
    st.header("López de Prado Statistical Audit (PSR & DSR)")
    st.write("""
    Standard Sharpe ratios overestimate confidence when researchers test multiple configurations.
    The **Deflated Sharpe Ratio (DSR)** adjusts for the **number of trials**, **return skewness**, and **fat-tailed kurtosis**.
    """)

    rets = result["net_return"]
    sr_daily = calculate_sharpe(rets, annualize=False)
    skew = calculate_skewness(rets)
    kurt = calculate_kurtosis(rets)
    t_obs = len(rets)

    psr = compute_psr(sr_daily, 0.0, t_obs, skew, kurt)

    st.subheader("Statistical Properties of Strategy Returns")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations (T)", f"{t_obs} bars")
    c2.metric("Return Skewness (γ₃)", f"{skew:.2f}")
    c3.metric("Kurtosis (γ₄)", f"{kurt:.2f}")
    c4.metric("Probabilistic Sharpe (PSR)", f"{psr*100:.1f}%")

    st.markdown("---")
    st.subheader("Multiple Testing Adjustment (Deflated Sharpe Ratio)")
    n_trials = st.slider("Number of Strategy Variations Tested (N)", 1, 1000, 494)
    var_sharpe = 0.00015  # Cross-sectional variance empirical benchmark

    dsr = compute_dsr(sr_daily, var_sharpe, float(n_trials), t_obs, skew, kurt)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Deflated Sharpe Ratio (DSR)", f"{dsr*100:.1f}%")
        if dsr >= 0.95:
            st.success("✅ **Statistically Significant Alpha:** Strategy passes DSR test at 95% confidence level.")
        elif dsr >= 0.50:
            st.warning("⚠️ **Borderline Confidence:** Substantial risk that performance is a statistical fluke.")
        else:
            st.error("❌ **False Discovery:** Strategy fails multiple testing hurdle; expected maximum Sharpe under null exceeds observed return.")
    with col2:
        st.write("""
        **Theoretical Formula:**
        $$\\text{DSR} = \\Phi\\left( \\frac{(\\widehat{SR} - SR^*) \\sqrt{T-1}}{\\sqrt{1 - \\gamma_3 \\widehat{SR} + \\frac{\\gamma_4 - 1}{4}\\widehat{SR}^2}} \\right)$$
        where the benchmark $SR^* = \\mathbb{E}\\left[\\max_{n=1\\dots N}\\{SR_n\\}\\right]$ is derived from Extreme Value Theory using the Euler-Mascheroni constant ($\\gamma \\approx 0.5772$).
        """)

# --- TAB 4: META-ML BENCHMARKS ---
with tab4:
    st.header("Meta-ML Model Evaluation & SHAP Feature Importance")
    st.write("""
    Overfit Radar trains calibrated ML classifiers (LightGBM & Regularized Logistic Regression) to predict
    whether a strategy will achieve positive net Sharpe out-of-sample ($P(\\text{OOS Sharpe}_{\\text{net}} > 0)$).
    """)

    importance_df = load_shap_importance()
    if importance_df is not None:
        st.subheader("Global SHAP Feature Importance (TreeExplainer on LightGBM)")
        st.caption("Features that most strongly predict whether an in-sample backtest survives in out-of-sample forward trading.")

        top_features = importance_df.head(10)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(top_features["feature"][::-1], top_features["mean_absolute_shap"][::-1], color="#1E88E5")
        ax.set_xlabel("Mean Absolute SHAP Value (Impact on Log-Odds of Survival)")
        ax.set_title("Top 10 Predictors of Out-of-Sample Survival")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)

    summary_data = load_model_summary()
    if summary_data is not None:
        st.subheader("Cross-Validation Summary Across Folds")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.head(15), use_container_width=True)
    else:
        st.info("Model experiment execution is completing in the background. Artifacts will populate here automatically.")

st.sidebar.markdown("---")
st.sidebar.info("""
**Overfit Radar v1.0**  
College Project Showcase Edition  
Built on Nifty 50 Large-Cap Universe (2010–2026)
""")


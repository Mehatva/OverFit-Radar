"""Script to generate and persist the master ML training dataset across cached large caps."""

from __future__ import annotations

import logging
import time
from pathlib import Path
import pandas as pd

from overfit_radar.config import load_data_config, load_costs_config, load_experiment_config
from overfit_radar.data.loader import load_from_cache
from overfit_radar.features.dataset import build_strategy_dataset
from overfit_radar.strategies.zoo import build_strategy_zoo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    data_cfg = load_data_config()
    costs_cfg = load_costs_config()
    exp_cfg = load_experiment_config()

    # Load cached symbols across key sectors
    target_symbols = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "ITC.NS",
        "SBIN.NS",
        "TATASTEEL.NS",
    ]

    universe_data = {}
    for sym in target_symbols:
        df = load_from_cache(sym, data_cfg.cache_dir)
        if df is not None:
            universe_data[sym] = df
            logger.info(f"Loaded {sym}: {len(df)} daily bars.")
        else:
            logger.warning(f"Symbol {sym} not found in cache.")

    if not universe_data:
        raise RuntimeError("No cached symbols found in data/raw.")

    zoo = build_strategy_zoo()
    logger.info(f"Loaded strategy zoo with {len(zoo)} configurations.")

    t0 = time.time()
    dataset = build_strategy_dataset(
        universe_data=universe_data,
        zoo=zoo,
        costs_config=costs_cfg,
        experiment_config=exp_cfg,
    )
    elapsed = time.time() - t0

    out_file = data_cfg.processed_dir / "training_dataset.parquet"
    data_cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_file)

    logger.info(f"\n==========================================")
    logger.info(f"DATASET GENERATION COMPLETE in {elapsed:.2f}s")
    logger.info(f"Total Rows: {len(dataset):,}")
    logger.info(f"Total Features: {dataset.shape[1]}")
    logger.info(f"Survival Rate: {dataset['survived'].mean():.2%}")
    logger.info(f"Saved to: {out_file} ({out_file.stat().st_size / 1024 / 1024:.2f} MB)")
    logger.info(f"==========================================\n")

    # Display sample inspection
    sample_cols = [
        "strategy_id",
        "symbol",
        "window_id",
        "is_sharpe",
        "is_dsr_raw",
        "neighbourhood_sharpe_mean",
        "neighbourhood_sharpe_std",
        "oos_sharpe",
        "survived",
    ]
    print(dataset[sample_cols].head(10).to_string())


if __name__ == "__main__":
    main()

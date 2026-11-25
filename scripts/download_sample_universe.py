"""Download and cache sample universe of 12 liquid Nifty stocks from 2010 to 2026."""

from __future__ import annotations

import logging
from pathlib import Path
from overfit_radar.config import load_data_config
from overfit_radar.data.loader import fetch_symbol_data
from overfit_radar.data.validator import check_split_adjustment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Sample of 12 diverse, high-liquidity large caps spanning key sectors
BENCHMARK_SYMBOLS = [
    "RELIANCE.NS",   # Energy / Conglomerate (1:1 bonus in Sep 2017 & Oct 2024)
    "TCS.NS",        # IT
    "INFY.NS",       # IT (1:1 bonus in June 2015 & Sep 2018)
    "HDFCBANK.NS",   # Private Banking (2:1 split in July 2011, 2:1 split in Sep 2019)
    "ICICIBANK.NS",  # Private Banking (5:1 split in Dec 2014)
    "ITC.NS",        # FMCG (1:2 bonus in July 2016)
    "SBIN.NS",       # Public Banking (10:1 split in Nov 2014)
    "BHARTIARTL.NS", # Telecom
    "LT.NS",         # Infrastructure / Engineering
    "TATASTEEL.NS",  # Metals (10:1 split on 2022-07-28)
    "ASIANPAINT.NS", # Consumer
    "MARUTI.NS",     # Auto
]


def main():
    config = load_data_config()
    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {len(BENCHMARK_SYMBOLS)} symbols from {config.start_date} to {config.end_date}...")
    
    success_count = 0
    for sym in BENCHMARK_SYMBOLS:
        try:
            df = fetch_symbol_data(
                symbol=sym,
                start_date=config.start_date,
                end_date=config.end_date,
                cache_dir=cache_dir,
                force_refresh=False,
                pause_seconds=0.5,
            )
            logger.info(f"✓ {sym}: {len(df)} bars (from {df.index[0].date()} to {df.index[-1].date()})")
            success_count += 1
        except Exception as e:
            logger.error(f"✗ Failed to download {sym}: {e}")

    logger.info(f"\nSuccessfully cached {success_count}/{len(BENCHMARK_SYMBOLS)} symbols to {cache_dir}.")
    
    # Verify corporate action adjustments
    logger.info("\nVerifying known corporate action adjustments on downloaded series:")
    
    # 1. Tata Steel 10:1 split on 2022-07-28
    ts_df = fetch_symbol_data("TATASTEEL.NS", config.start_date, config.end_date, cache_dir)
    ts_adjusted = check_split_adjustment(ts_df, "2022-07-28", max_allowed_abs_return=0.20)
    logger.info(f"Tata Steel (2022-07-28 10:1 split) adjustment continuous: {ts_adjusted}")
    assert ts_adjusted, "Tata Steel split was NOT properly adjusted!"

    # 2. SBI 10:1 split on 2014-11-20
    sbi_df = fetch_symbol_data("SBIN.NS", config.start_date, config.end_date, cache_dir)
    sbi_adjusted = check_split_adjustment(sbi_df, "2014-11-20", max_allowed_abs_return=0.20)
    logger.info(f"State Bank of India (2014-11-20 10:1 split) adjustment continuous: {sbi_adjusted}")
    assert sbi_adjusted, "SBIN split was NOT properly adjusted!"

    # 3. HDFC Bank 2:1 split on 2019-09-19
    hdfc_df = fetch_symbol_data("HDFCBANK.NS", config.start_date, config.end_date, cache_dir)
    hdfc_adjusted = check_split_adjustment(hdfc_df, "2019-09-19", max_allowed_abs_return=0.20)
    logger.info(f"HDFC Bank (2019-09-19 2:1 split) adjustment continuous: {hdfc_adjusted}")
    assert hdfc_adjusted, "HDFCBANK split was NOT properly adjusted!"

    logger.info("\nALL CORPORATE ACTION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

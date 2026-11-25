"""Data loading, yfinance ingestion, and local Parquet caching module."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf

from overfit_radar.data.validator import validate_ohlcv

logger = logging.getLogger(__name__)


def _sanitize_symbol(symbol: str) -> str:
    """Convert ticker symbol to safe filename (e.g. 'RELIANCE.NS' -> 'RELIANCE_NS')."""
    return symbol.replace(".", "_").replace("^", "")


def save_to_cache(df: pd.DataFrame, symbol: str, cache_dir: Path) -> Path:
    """Save cleaned OHLCV dataframe to a local Parquet file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = cache_dir / f"{_sanitize_symbol(symbol)}.parquet"
    
    # Store with date column explicitly or preserve index
    to_save = df.copy()
    if not isinstance(to_save.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex to save to cache.")
    
    to_save.to_parquet(filename, index=True)
    return filename


def load_from_cache(symbol: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    """Load OHLCV dataframe from local Parquet cache if available."""
    filename = cache_dir / f"{_sanitize_symbol(symbol)}.parquet"
    if not filename.is_file():
        return None
    try:
        df = pd.read_parquet(filename)
        # Ensure DatetimeIndex with no timezone (timezone-naive for clean slicing)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "date"
        return df
    except Exception as e:
        logger.warning(f"Failed to read cache for {symbol} at {filename}: {e}")
        return None


def fetch_symbol_data(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    force_refresh: bool = False,
    pause_seconds: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch historical daily OHLCV for a symbol.
    
    Checks local Parquet cache first. If not found or if force_refresh is True,
    downloads from yfinance, standardizes columns, validates integrity, and saves to cache.
    """
    if not force_refresh:
        cached_df = load_from_cache(symbol, cache_dir)
        if cached_df is not None:
            # Filter to requested date window
            filtered = cached_df.loc[
                (cached_df.index >= pd.to_datetime(start_date)) &
                (cached_df.index <= pd.to_datetime(end_date))
            ]
            if len(filtered) > 50:
                validate_ohlcv(filtered, symbol)
                return filtered

    logger.info(f"Downloading historical data for {symbol} from yfinance...")
    
    # Download raw history with auto_adjust=False to get both Close and Adj Close
    ticker = yf.Ticker(symbol)
    raw = ticker.history(start=start_date, end=end_date, auto_adjust=False)
    
    if raw.empty:
        raise ValueError(f"No price data returned for {symbol} from {start_date} to {end_date}.")

    # Standardize column names to lowercase
    # yfinance columns typically: ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    }
    
    df = raw.rename(columns=col_map)
    
    # If Adj Close is missing (rare in yf), fallback to close
    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df["close"]
        
    df = df[["open", "high", "low", "close", "adjusted_close", "volume"]].copy()
    
    # Strip timezone from index for clean comparisons
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    
    # Sort chronologically and drop duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    
    # Validate integrity
    validate_ohlcv(df, symbol)
    
    # Save to Parquet cache
    save_to_cache(df, symbol, cache_dir)
    
    # Respect rate limits
    if pause_seconds > 0:
        time.sleep(pause_seconds)
        
    return df


def load_universe(
    symbols: List[str],
    start_date: str,
    end_date: str,
    cache_dir: Path,
    force_refresh: bool = False,
    pause_seconds: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """
    Download/load a collection of symbols and return a mapping of symbol -> OHLCV DataFrame.
    """
    universe_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = fetch_symbol_data(
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                force_refresh=force_refresh,
                pause_seconds=pause_seconds,
            )
            universe_data[sym] = df
            logger.info(f"Loaded {sym}: {len(df)} daily bars.")
        except Exception as e:
            logger.error(f"Error loading {sym}: {e}")
            raise e
            
    return universe_data

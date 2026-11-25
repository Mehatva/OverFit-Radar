"""Unit tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
import pytest
from overfit_radar.config import (
    load_costs_config,
    load_data_config,
    load_experiment_config,
    get_project_root,
)


def test_project_root_exists():
    root = get_project_root()
    assert root.is_dir()
    assert (root / "configs").is_dir()
    assert (root / "pyproject.toml").is_file()


def test_load_data_config():
    config = load_data_config()
    assert config.start_date == "2010-01-01"
    assert config.timing == "t_plus_1_open"
    assert config.risk_free_rate == 0.0
    assert len(config.universe) >= 40
    assert "RELIANCE.NS" in config.universe
    assert config.cache_dir.name == "raw"


def test_load_costs_config():
    config = load_costs_config()
    assert config.default_slippage_bps > 0
    assert config.cost_multiplier == 1.0
    assert len(config.regimes) >= 2
    
    # Chronological order of cost regimes
    dates = [r.effective_date for r in config.regimes]
    assert dates == sorted(dates), "Cost regimes must be strictly in chronological order"
    
    # Verify rate magnitudes (should be decimals, not percentages, e.g. 0.001 for 0.1%)
    for r in config.regimes:
        assert 0.0 < r.stt_delivery_buy < 0.01, f"STT buy rate unreasonable: {r.stt_delivery_buy}"
        assert 0.0 < r.stt_delivery_sell < 0.01, f"STT sell rate unreasonable: {r.stt_delivery_sell}"
        assert 0.0 < r.stamp_duty_buy < 0.01, f"Stamp duty rate unreasonable: {r.stamp_duty_buy}"
        assert r.gst_rate == 0.18, f"Expected 18% GST, got {r.gst_rate}"


def test_load_experiment_config():
    config = load_experiment_config()
    assert config.random_seed == 42
    assert config.walk_forward.in_sample_months == 36
    assert config.walk_forward.out_of_sample_months == 12
    assert config.walk_forward.embargo_days == 21
    assert config.cscv.num_blocks == 16
    assert config.cscv.train_blocks == 8
    assert config.group_by == "family"

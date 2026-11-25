"""Configuration loading and validation module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass(frozen=True)
class CostRegime:
    effective_date: str
    name: str
    stt_delivery_buy: float
    stt_delivery_sell: float
    stamp_duty_buy: float
    stamp_duty_sell: float
    exchange_txn_fee: float
    sebi_turnover_fee: float
    brokerage_delivery: float
    gst_rate: float
    dp_charges_per_sell: float


@dataclass(frozen=True)
class CostsConfig:
    default_slippage_bps: float
    cost_multiplier: float
    regimes: List[CostRegime]


@dataclass(frozen=True)
class DataConfig:
    start_date: str
    end_date: str
    timing: str
    risk_free_rate: float
    cache_dir: Path
    processed_dir: Path
    primary_provider: str
    rate_limit_pause_seconds: float
    universe: List[str]


@dataclass(frozen=True)
class WalkForwardConfig:
    in_sample_months: int
    out_of_sample_months: int
    step_months: int
    embargo_days: int
    min_is_trades: int


@dataclass(frozen=True)
class CSCVConfig:
    num_blocks: int
    train_blocks: int


@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int
    walk_forward: WalkForwardConfig
    cscv: CSCVConfig
    survival_threshold: float
    train_window_ratio: float
    group_by: str


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(path: Path) -> Dict[str, Any]:
    """Safely load a YAML file."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML content at {path} must be a dictionary")
    return data


def load_data_config(config_dir: Optional[Path] = None) -> DataConfig:
    """Load and parse data.yaml."""
    base_dir = config_dir or (get_project_root() / "configs")
    data = load_yaml(base_dir / "data.yaml")
    
    return DataConfig(
        start_date=data["date_range"]["start_date"],
        end_date=data["date_range"]["end_date"],
        timing=data["execution"]["timing"],
        risk_free_rate=float(data["execution"]["risk_free_rate"]),
        cache_dir=get_project_root() / data["storage"]["cache_dir"],
        processed_dir=get_project_root() / data["storage"]["processed_dir"],
        primary_provider=data["provider"]["primary"],
        rate_limit_pause_seconds=float(data["provider"]["rate_limit_pause_seconds"]),
        universe=list(data["universe"]),
    )


def load_costs_config(config_dir: Optional[Path] = None) -> CostsConfig:
    """Load and parse costs.yaml."""
    base_dir = config_dir or (get_project_root() / "configs")
    data = load_yaml(base_dir / "costs.yaml")
    
    regimes = [
        CostRegime(
            effective_date=r["effective_date"],
            name=r["name"],
            stt_delivery_buy=float(r["stt_delivery_buy"]),
            stt_delivery_sell=float(r["stt_delivery_sell"]),
            stamp_duty_buy=float(r["stamp_duty_buy"]),
            stamp_duty_sell=float(r["stamp_duty_sell"]),
            exchange_txn_fee=float(r["exchange_txn_fee"]),
            sebi_turnover_fee=float(r["sebi_turnover_fee"]),
            brokerage_delivery=float(r["brokerage_delivery"]),
            gst_rate=float(r["gst_rate"]),
            dp_charges_per_sell=float(r["dp_charges_per_sell"]),
        )
        for r in data["regimes"]
    ]
    
    return CostsConfig(
        default_slippage_bps=float(data["default_slippage_bps"]),
        cost_multiplier=float(data["cost_multiplier"]),
        regimes=regimes,
    )


def load_experiment_config(config_dir: Optional[Path] = None) -> ExperimentConfig:
    """Load and parse experiment.yaml."""
    base_dir = config_dir or (get_project_root() / "configs")
    data = load_yaml(base_dir / "experiment.yaml")
    
    wf = data["walk_forward"]
    cscv = data["cscv"]
    
    return ExperimentConfig(
        random_seed=int(data["random_seed"]),
        walk_forward=WalkForwardConfig(
            in_sample_months=int(wf["in_sample_months"]),
            out_of_sample_months=int(wf["out_of_sample_months"]),
            step_months=int(wf["step_months"]),
            embargo_days=int(wf["embargo_days"]),
            min_is_trades=int(wf["min_is_trades"]),
        ),
        cscv=CSCVConfig(
            num_blocks=int(cscv["num_blocks"]),
            train_blocks=int(cscv["train_blocks"]),
        ),
        survival_threshold=float(data["labeling"]["survival_threshold"]),
        train_window_ratio=float(data["cv_split"]["train_window_ratio"]),
        group_by=str(data["cv_split"]["group_by"]),
    )

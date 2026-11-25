"""Base data structures and interfaces for strategy zoo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class StrategyConfig:
    """Immutable specification for a single strategy configuration."""
    strategy_id: str
    family: str
    params: Dict[str, Any]

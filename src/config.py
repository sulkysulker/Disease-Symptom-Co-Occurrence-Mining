from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MiningConfig:
    """Configuration values for mining and preprocessing."""

    min_support: float = 0.05
    min_confidence: float = 0.4
    max_len: int | None = None  # max size of itemsets, None = no explicit limit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SAMPLE_PATH = DATA_DIR / "sample_disease_symptom.csv"


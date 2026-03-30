from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {"patient_id", "disease", "symptom"}


def load_disease_symptom_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a disease–symptom CSV file and validate required columns.

    Expected columns:
    - patient_id
    - disease
    - symptom
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Basic cleaning: drop rows with any NA in required columns
    df = df.dropna(subset=list(REQUIRED_COLUMNS))

    # Normalize to string
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    return df


def list_unique_values(df: pd.DataFrame, column: str) -> list[str]:
    """Return sorted unique values from a column."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in DataFrame")
    return sorted(v for v in df[column].dropna().astype(str).unique())


def filter_by_disease(df: pd.DataFrame, diseases: Iterable[str]) -> pd.DataFrame:
    """Filter rows where disease is in the given list."""
    diseases_set = {str(d).strip() for d in diseases}
    if not diseases_set:
        return df
    return df[df["disease"].isin(diseases_set)]


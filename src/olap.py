from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class WarehouseTables:
    """
    A tiny logical warehouse representation (in-memory).

    - fact: the event/fact table (patient-disease-symptom rows)
    - disease_dim: unique diseases
    - symptom_dim: unique symptoms
    - patient_dim: unique patients
    """

    fact: pd.DataFrame
    disease_dim: pd.DataFrame
    symptom_dim: pd.DataFrame
    patient_dim: pd.DataFrame


def build_logical_warehouse(df: pd.DataFrame) -> WarehouseTables:
    """
    Build a simple star-schema-like logical warehouse in-memory.

    Expected columns in df: patient_id, disease, symptom
    """
    required = {"patient_id", "disease", "symptom"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for warehouse: {', '.join(sorted(missing))}")

    fact = df[list(required)].copy()
    for col in ["patient_id", "disease", "symptom"]:
        fact[col] = fact[col].astype(str).str.strip()
    fact = fact.dropna()

    disease_dim = pd.DataFrame({"disease": sorted(fact["disease"].unique())})
    symptom_dim = pd.DataFrame({"symptom": sorted(fact["symptom"].unique())})
    patient_dim = pd.DataFrame({"patient_id": sorted(fact["patient_id"].unique())})

    return WarehouseTables(
        fact=fact,
        disease_dim=disease_dim,
        symptom_dim=symptom_dim,
        patient_dim=patient_dim,
    )


def slice_df(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """OLAP Slice: fix one dimension value (e.g., disease='Flu')."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in DataFrame")
    return df[df[column].astype(str) == str(value)]


def dice_df(df: pd.DataFrame, filters: dict[str, Iterable[str]]) -> pd.DataFrame:
    """OLAP Dice: filter by multiple values across multiple dimensions."""
    out = df
    for col, values in filters.items():
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not in DataFrame")
        vals = {str(v) for v in values}
        if vals:
            out = out[out[col].astype(str).isin(vals)]
    return out


def olap_aggregate(
    fact: pd.DataFrame,
    dimensions: list[str],
    measure: str = "row_count",
) -> pd.DataFrame:
    """
    Aggregate the fact table by dimensions (core OLAP cube operation).

    Measures:
    - row_count: number of fact rows
    - patient_count: distinct patient_id count
    - symptom_count: distinct symptom count
    """
    if not dimensions:
        # Grand total
        return pd.DataFrame({"row_count": [len(fact)]})

    for d in dimensions:
        if d not in fact.columns:
            raise KeyError(f"Dimension '{d}' not in fact table")

    gb = fact.groupby(dimensions, dropna=False)

    if measure == "row_count":
        out = gb.size().reset_index(name="row_count")
    elif measure == "patient_count":
        if "patient_id" not in fact.columns:
            raise KeyError("patient_id required for patient_count")
        out = gb["patient_id"].nunique().reset_index(name="patient_count")
    elif measure == "symptom_count":
        if "symptom" not in fact.columns:
            raise KeyError("symptom required for symptom_count")
        out = gb["symptom"].nunique().reset_index(name="symptom_count")
    else:
        raise ValueError(f"Unknown measure: {measure}")

    return out


def pivot_cube(
    agg_df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    fill_value: int = 0,
) -> pd.DataFrame:
    """Pivot an aggregated cube into a 2D OLAP view."""
    if agg_df.empty:
        return pd.DataFrame()
    if index not in agg_df.columns or columns not in agg_df.columns or values not in agg_df.columns:
        raise KeyError("Pivot requires index/columns/values to exist in aggregated DataFrame")
    pv = agg_df.pivot_table(index=index, columns=columns, values=values, aggfunc="sum", fill_value=fill_value)
    return pv


def roll_up_dimensions(dimensions: list[str]) -> list[str]:
    """Roll-up: remove the last (most detailed) dimension."""
    if not dimensions:
        return []
    return dimensions[:-1]


def drill_down_dimensions(dimensions: list[str], add_dimension: str) -> list[str]:
    """Drill-down: add a new dimension for finer detail."""
    if add_dimension in dimensions:
        return dimensions
    return dimensions + [add_dimension]


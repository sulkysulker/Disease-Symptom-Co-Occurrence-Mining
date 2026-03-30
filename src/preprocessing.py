from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


def build_patient_symptom_transactions(
    df: pd.DataFrame,
    group_by_cols: Iterable[str] = ("patient_id", "disease"),
) -> pd.DataFrame:
    """
    Build transactions where each row corresponds to a (patient, disease) pair and
    `symptoms` column is a list of symptoms observed.

    Parameters
    ----------
    df : DataFrame
        Input rows with columns: patient_id, disease, symptom.
    group_by_cols : iterable of str
        Columns to group by when creating transactions.

    Returns
    -------
    DataFrame
        Columns: group_by_cols..., symptoms (list of strings)
    """
    for col in group_by_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

    grouped = (
        df.groupby(list(group_by_cols))["symptom"]
        .apply(lambda s: sorted(set(str(x).strip() for x in s if pd.notna(x))))
        .reset_index(name="symptoms")
    )

    # Drop empty symptom lists (if any)
    grouped = grouped[grouped["symptoms"].map(len) > 0]
    return grouped


def symptom_transactions_to_list(transactions_df: pd.DataFrame) -> List[List[str]]:
    """Convert the transactions DataFrame to a plain Python list of symptom lists."""
    if "symptoms" not in transactions_df.columns:
        raise KeyError("Expected 'symptoms' column in transactions DataFrame")
    return [list(symptoms) for symptoms in transactions_df["symptoms"]]


def build_one_hot_encoding(
    transactions: List[List[str]],
) -> pd.DataFrame:
    """
    Convert list-of-list transactions into a one-hot encoded DataFrame suitable
    for frequent itemset mining (Apriori).

    Each column corresponds to a symptom and values are bool (present/absent).
    """
    all_items = sorted({item for tx in transactions for item in tx})
    rows: List[Dict[str, bool]] = []

    for tx in transactions:
        tx_set = set(tx)
        row = {item: (item in tx_set) for item in all_items}
        rows.append(row)

    return pd.DataFrame(rows, columns=all_items)


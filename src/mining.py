from __future__ import annotations

from typing import Optional

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def mine_frequent_itemsets(
    one_hot_df: pd.DataFrame,
    min_support: float = 0.05,
    max_len: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run Apriori frequent itemset mining on one-hot encoded data.

    Parameters
    ----------
    one_hot_df : DataFrame
        One-hot encoded symptom matrix (0/1 or True/False).
    min_support : float
        Minimum support threshold.
    max_len : int or None
        Maximum length of itemsets. None means no explicit limit.
    """
    if one_hot_df.empty:
        return pd.DataFrame(columns=["support", "itemsets"])

    frequent = apriori(
        one_hot_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    frequent.sort_values("support", ascending=False, inplace=True)
    frequent.reset_index(drop=True, inplace=True)
    return frequent


def mine_association_rules(
    frequent_itemsets: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets.

    Parameters
    ----------
    frequent_itemsets : DataFrame
        Output of `mine_frequent_itemsets`.
    metric : str
        Metric used for thresholding (e.g., 'confidence', 'lift').
    min_threshold : float
        Minimum value of the selected metric.
    """
    if frequent_itemsets.empty:
        return pd.DataFrame(
            columns=[
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
        )

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    # Sort rules by chosen metric (descending)
    if metric in rules.columns:
        rules = rules.sort_values(metric, ascending=False)

    rules.reset_index(drop=True, inplace=True)
    return rules


def compute_symptom_cooccurrence_matrix(
    one_hot_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a simple symptom co-occurrence matrix (count-based).

    Returns a square DataFrame where entry (i, j) is the number of
    transactions where both symptom i and symptom j appear.
    """
    if one_hot_df.empty:
        return pd.DataFrame()

    # Use matrix multiplication on boolean values
    bool_df = one_hot_df.astype(bool)
    co_matrix = bool_df.T.dot(bool_df)
    return co_matrix


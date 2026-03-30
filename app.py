from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from src.config import DEFAULT_SAMPLE_PATH, MiningConfig
from src.data_loader import load_disease_symptom_csv, list_unique_values, filter_by_disease
from src.mining import (
    compute_symptom_cooccurrence_matrix,
    mine_association_rules,
    mine_frequent_itemsets,
)
from src.preprocessing import (
    build_one_hot_encoding,
    build_patient_symptom_transactions,
    symptom_transactions_to_list,
)
from src.visualization import plot_cooccurrence_heatmap, plot_top_frequent_itemsets
from src.olap import (
    build_logical_warehouse,
    dice_df,
    drill_down_dimensions,
    olap_aggregate,
    pivot_cube,
    roll_up_dimensions,
    slice_df,
)


def _set_to_str(x) -> str:
    """Convert frozenset/set/list/tuple of strings to a stable display string."""
    try:
        return ", ".join(sorted(list(x)))
    except Exception:
        return str(x)


def load_data_ui() -> Optional[pd.DataFrame]:
    """UI section for loading CSV data (upload or sample)."""
    st.sidebar.header("1. Load Data")

    use_sample = st.sidebar.radio(
        "Dataset",
        options=["Use sample dataset", "Upload your own CSV"],
        index=0,
    )

    if use_sample == "Use sample dataset":
        st.sidebar.info(f"Using sample dataset: `{DEFAULT_SAMPLE_PATH.name}`")
        try:
            return load_disease_symptom_csv(DEFAULT_SAMPLE_PATH)
        except Exception as e:  # pragma: no cover - UI error path
            st.error(f"Failed to load sample dataset: {e}")
            return None

    uploaded = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Expected columns: patient_id, disease, symptom",
    )
    if uploaded is None:
        st.info("Upload a CSV file to begin.")
        return None

    try:
        data = pd.read_csv(io.BytesIO(uploaded.read()))
        # Validate columns via loader logic
        # Write to a temp buffer so we can reuse loader
        with io.StringIO() as buf:
            data.to_csv(buf, index=False)
            buf.seek(0)
            data = pd.read_csv(buf)
        # Manually check required columns
        missing = {"patient_id", "disease", "symptom"} - set(data.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(sorted(missing))}")
            return None
        # Normalize
        for col in ["patient_id", "disease", "symptom"]:
            data[col] = data[col].astype(str).str.strip()
        return data
    except Exception as e:  # pragma: no cover - UI error path
        st.error(f"Error reading uploaded CSV: {e}")
        return None


def sidebar_mining_params() -> MiningConfig:
    """UI for configuring mining parameters."""
    st.sidebar.header("2. Mining Parameters")
    min_support = st.sidebar.slider(
        "Minimum Support",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Itemsets that appear in fewer transactions than this will be filtered out.",
    )
    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Rules with confidence below this threshold will be filtered out.",
    )
    max_len_options = [None, 2, 3, 4]
    max_len_label = st.sidebar.selectbox(
        "Maximum Itemset Size",
        options=range(len(max_len_options)),
        format_func=lambda i: "No limit" if max_len_options[i] is None else str(max_len_options[i]),
        index=2,
    )
    max_len = max_len_options[max_len_label]
    return MiningConfig(min_support=min_support, min_confidence=min_confidence, max_len=max_len)


def main() -> None:
    st.set_page_config(
        page_title="Disease–Symptom Co-occurrence Mining",
        layout="wide",
    )

    st.title("Disease–Symptom Co-occurrence Mining")
    st.markdown(
        """
        Analyze patterns of **co-occurring symptoms** across diseases using
        frequent itemset mining and association rules.
        """
    )

    cfg = sidebar_mining_params()
    df = load_data_ui()

    if df is None or df.empty:
        st.warning("No data loaded yet. Use the sidebar to select a dataset.")
        return

    tabs = st.tabs(["Mining", "OLAP Explorer"])

    # --------------------
    # Mining tab
    # --------------------
    with tabs[0]:
        with st.expander("Preview Raw Data", expanded=False):
            st.write(df.head(20))
            st.caption(f"Total rows: {len(df)}")

        # Disease filter
        st.subheader("Disease Selection")
        all_diseases = list_unique_values(df, "disease")
        selected_diseases = st.multiselect(
            "Filter by disease (optional)",
            options=all_diseases,
            default=[],
            help="If none selected, all diseases are used.",
        )
        filtered_df = filter_by_disease(df, selected_diseases) if selected_diseases else df

        if filtered_df.empty:
            st.warning("No rows left after disease filtering.")
            return

        st.markdown(
            f"Using **{len(filtered_df)}** rows from **{len(all_diseases if not selected_diseases else selected_diseases)}** diseases."
        )

        # Build transactions
        transactions_df = build_patient_symptom_transactions(filtered_df)
        transactions = symptom_transactions_to_list(transactions_df)

        st.subheader("Transaction View")
        with st.expander("Sample Transactions", expanded=False):
            st.write(transactions_df.head(15))
            st.caption(f"Total transactions (patient–disease): {len(transactions_df)}")

        if not transactions:
            st.error("No symptom transactions could be built from the data.")
            return

        # One-hot encode for mining
        one_hot_df = build_one_hot_encoding(transactions)

        # Frequent itemsets
        frequent_itemsets = mine_frequent_itemsets(
            one_hot_df,
            min_support=cfg.min_support,
            max_len=cfg.max_len,
        )

        st.subheader("Frequent Symptom Itemsets")
        if frequent_itemsets.empty:
            st.info("No frequent itemsets found with the current support threshold.")
        else:
            display_itemsets = frequent_itemsets.copy()
            if "itemsets" in display_itemsets.columns:
                display_itemsets["itemsets"] = display_itemsets["itemsets"].apply(_set_to_str)
            st.dataframe(display_itemsets.head(20))
            fig_itemsets = plot_top_frequent_itemsets(frequent_itemsets, top_n=15)
            if fig_itemsets is not None:
                st.pyplot(fig_itemsets)

        # Association rules
        st.subheader("Association Rules (Symptoms → Symptoms)")
        rules = mine_association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=cfg.min_confidence,
        )
        if rules.empty:
            st.info("No association rules found with the current thresholds.")
        else:
            display_rules = rules.copy()
            display_rules["antecedents"] = display_rules["antecedents"].apply(_set_to_str)
            display_rules["consequents"] = display_rules["consequents"].apply(_set_to_str)
            st.dataframe(
                display_rules[
                    [
                        "antecedents",
                        "consequents",
                        "support",
                        "confidence",
                        "lift",
                    ]
                ].head(50)
            )

        # Co-occurrence heatmap
        st.subheader("Symptom Co-occurrence Heatmap")
        co_matrix = compute_symptom_cooccurrence_matrix(one_hot_df)
        fig_heatmap = plot_cooccurrence_heatmap(co_matrix, max_symptoms=20)
        if fig_heatmap is None:
            st.info("Not enough data to build co-occurrence matrix.")
        else:
            st.pyplot(fig_heatmap)

        st.markdown("---")
        st.caption(
            "This app was built as a Data Warehousing & Data Mining course project for mining "
            "disease–symptom co-occurrence patterns."
        )

    # --------------------
    # OLAP Explorer tab
    # --------------------
    with tabs[1]:
        st.subheader("OLAP Explorer (Logical Data Warehouse)")
        st.markdown(
            """
            This section models your CSV as a **logical data warehouse** (star schema):
            - **Fact table**: patient–disease–symptom rows
            - **Dimensions**: Patient, Disease, Symptom

            Then it performs OLAP operations like **slice**, **dice**, **roll-up**, and **drill-down**.
            """
        )

        wh = build_logical_warehouse(df)
        fact = wh.fact

        with st.expander("Warehouse Tables Preview", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("Disease dimension")
                st.dataframe(wh.disease_dim.head(10))
            with c2:
                st.write("Symptom dimension")
                st.dataframe(wh.symptom_dim.head(10))
            with c3:
                st.write("Patient dimension")
                st.dataframe(wh.patient_dim.head(10))
            st.write("Fact table (preview)")
            st.dataframe(fact.head(15))

        st.markdown("#### 1) Slice / Dice Filters")
        disease_values = sorted(fact["disease"].unique())
        symptom_values = sorted(fact["symptom"].unique())

        slice_disease = st.selectbox("Slice by single disease (optional)", options=["(none)"] + disease_values)
        dice_diseases = st.multiselect("Dice: diseases (optional)", options=disease_values, default=[])
        dice_symptoms = st.multiselect("Dice: symptoms (optional)", options=symptom_values, default=[])

        filtered = fact
        if slice_disease != "(none)":
            filtered = slice_df(filtered, "disease", slice_disease)
        filtered = dice_df(filtered, {"disease": dice_diseases, "symptom": dice_symptoms})

        st.caption(f"Filtered fact rows: {len(filtered)}")

        st.markdown("#### 2) Cube Aggregation + Pivot")
        dims_available = ["disease", "symptom"]
        selected_dims = st.multiselect(
            "Group by dimensions (OLAP cube)",
            options=dims_available,
            default=["disease", "symptom"],
        )
        measure = st.selectbox(
            "Measure",
            options=[
                ("row_count", "Row count (facts)"),
                ("patient_count", "Distinct patients"),
                ("symptom_count", "Distinct symptoms"),
            ],
            format_func=lambda x: x[1],
        )[0]

        agg = olap_aggregate(filtered, selected_dims, measure=measure)
        st.write("Aggregated cube (table)")
        st.dataframe(agg.head(200))

        if len(selected_dims) >= 2:
            index_dim = st.selectbox("Pivot index", options=selected_dims, index=0)
            col_dim = st.selectbox("Pivot columns", options=[d for d in selected_dims if d != index_dim], index=0)
            pv = pivot_cube(agg, index=index_dim, columns=col_dim, values=measure)
            st.write("Pivot (OLAP view)")
            st.dataframe(pv)

        st.markdown("#### 3) Roll-up / Drill-down")
        st.caption(
            "Roll-up removes one dimension (less detail). Drill-down adds a dimension (more detail). "
            "This is just a guided way to change your grouping levels."
        )
        colA, colB = st.columns(2)
        with colA:
            if st.button("Roll-up (remove last dimension)"):
                st.session_state["olap_dims"] = roll_up_dimensions(selected_dims)
        with colB:
            drill_dim = st.selectbox("Drill-down add dimension", options=[d for d in dims_available if d not in selected_dims] or ["(none)"])
            if st.button("Drill-down (add selected)") and drill_dim != "(none)":
                st.session_state["olap_dims"] = drill_down_dimensions(selected_dims, drill_dim)

        if "olap_dims" in st.session_state and st.session_state["olap_dims"] != selected_dims:
            st.info(f"Suggested new dimensions: {st.session_state['olap_dims']}")


if __name__ == "__main__":
    main()


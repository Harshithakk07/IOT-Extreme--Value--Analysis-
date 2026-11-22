import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import genpareto

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="IoT Extreme Value Analysis",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📊 Duplicate-Aware Extreme Value Analysis")
st.caption(
    "IoT Attack Traffic Analysis using CICIoT2023 | "
    "POT and Generalized Pareto Distribution"
)

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Dataset Upload",
        "Data Cleaning",
        "Duplicate Detection",
        "Extreme Value Analysis",
        "Comparison Dashboard",
        "Research Results"
    ]
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "data" not in st.session_state:
    st.session_state.data = None

if "clean_data" not in st.session_state:
    st.session_state.clean_data = None

if "dedup_data" not in st.session_state:
    st.session_state.dedup_data = None

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def get_numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def get_analysis_data():
    if st.session_state.clean_data is not None:
        return st.session_state.clean_data
    return st.session_state.data


def remove_duplicates(df):
    return df.drop_duplicates().copy()


def perform_gpd_analysis(data, column, quantile):
    values = pd.to_numeric(data[column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) < 20:
        return None

    threshold = values.quantile(quantile)
    exceedances = values[values > threshold] - threshold

    if len(exceedances) < 10:
        return None

    try:
        shape, loc, scale = genpareto.fit(exceedances, floc=0)

        return {
            "values": values,
            "threshold": threshold,
            "exceedances": exceedances,
            "shape": shape,
            "scale": scale,
            "loc": loc,
            "quantile": quantile
        }
    except Exception:
        return None


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if page == "Overview":

    st.subheader("Research Overview")

    st.markdown("""
    This dashboard explores extreme traffic behaviour in IoT network data.

    ### Research Objectives

    - Study the effect of duplicate records on extreme value estimation.
    - Analyse extreme traffic behaviour using Peak Over Threshold (POT).
    - Fit the Generalized Pareto Distribution (GPD).
    - Compare original and deduplicated traffic data.
    - Explore attack-type-specific traffic behaviour.

    ### Research Questions

    **RQ1:** How do duplicates affect extreme value estimation?

    **RQ2:** How do attack subtypes differ in tail behaviour?

    **RQ3:** How can extreme traffic observations be analysed using
    threshold-based modelling?
    """)

    if st.session_state.data is not None:

        df = st.session_state.data

        col1, col2, col3 = st.columns(3)

        col1.metric("Dataset Rows", len(df))
        col2.metric("Dataset Columns", len(df.columns))
        col3.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.subheader("Uploaded Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

    else:
        st.info("Upload a CSV dataset from the Dataset Upload page.")


# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------
elif page == "Dataset Upload":

    st.subheader("📁 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            st.session_state.data = df
            st.session_state.clean_data = df.copy()
            st.session_state.dedup_data = remove_duplicates(df)

            st.success("Dataset uploaded successfully!")

            st.metric("Rows", len(df))
            st.metric("Columns", len(df.columns))

            st.subheader("Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)

            st.subheader("Column Information")

            info_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isna().sum().values
            })

            st.dataframe(info_df, use_container_width=True)

        except Exception as e:
            st.error(f"Unable to read CSV: {e}")

    elif st.session_state.data is not None:

        st.info("Your previously uploaded dataset is still available.")

        st.dataframe(
            st.session_state.data.head(),
            use_container_width=True
        )


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------
elif page == "Data Cleaning":

    st.subheader("🧹 Data Cleaning")

    if st.session_state.data is None:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state.data.copy()

        col1, col2 = st.columns(2)

        col1.metric("Rows Before Cleaning", len(df))
        col2.metric("Missing Values", int(df.isna().sum().sum()))

        st.subheader("Missing Values by Column")

        missing = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isna().sum().values
        })

        st.dataframe(missing, use_container_width=True)

        if st.button("Remove Rows with Missing Values"):

            cleaned = df.dropna().copy()

            st.session_state.clean_data = cleaned

            st.success("Missing values removed from rows!")

            st.metric("Rows After Cleaning", len(cleaned))
            st.metric("Rows Removed", len(df) - len(cleaned))

        if st.session_state.clean_data is not None:

            st.subheader("Current Cleaned Dataset")

            st.dataframe(
                st.session_state.clean_data.head(),
                use_container_width=True
            )


# --------------------------------------------------
# DUPLICATE DETECTION
# --------------------------------------------------
elif page == "Duplicate Detection":

    st.subheader("🔍 Duplicate Detection")

    if st.session_state.data is None:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state.data.copy()

        duplicate_count = int(df.duplicated().sum())
        unique_df = remove_duplicates(df)

        col1, col2, col3 = st.columns(3)

        col1.metric("Original Rows", len(df))
        col2.metric("Duplicate Rows", duplicate_count)
        col3.metric("Rows After Deduplication", len(unique_df))

        st.subheader("Duplicate Records")

        if duplicate_count > 0:

            st.dataframe(
                df[df.duplicated(keep=False)],
                use_container_width=True
            )

        else:

            st.info("No exact duplicate rows detected.")

        if st.button("Apply Duplicate Removal"):

            st.session_state.dedup_data = unique_df

            st.success("Exact duplicate records removed!")

            st.metric(
                "Rows Remaining",
                len(st.session_state.dedup_data)
            )

        st.caption(
            "This module detects exact row duplicates. "
            "It does not detect near-duplicates or identify duplicates "
            "using a flow-specific identity rule."
        )


# --------------------------------------------------
# EXTREME VALUE ANALYSIS
# --------------------------------------------------
elif page == "Extreme Value Analysis":

    st.subheader("📈 Peak Over Threshold (POT) Analysis")

    df = get_analysis_data()

    if df is None:

        st.warning("Please upload a dataset first.")

    else:

        numeric_cols = get_numeric_columns(df)

        if len(numeric_cols) == 0:

            st.error("No numeric columns are available for analysis.")

        else:

            selected_column = st.selectbox(
                "Select Numeric Variable",
                numeric_cols
            )

            quantile = st.slider(
                "Threshold Percentile",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01
            )

            st.caption(
                "The threshold is selected using the chosen percentile. "
                "Sensitivity to threshold choice should be assessed."
            )

            result = perform_gpd_analysis(
                df,
                selected_column,
                quantile
            )

            if result is None:

                st.warning(
                    "Not enough valid observations or exceedances to "
                    "fit a GPD. Try a lower threshold percentile."
                )

            else:

                threshold = result["threshold"]
                exceedances = result["exceedances"]
                shape = result["shape"]
                scale = result["scale"]
                values = result["values"]

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Threshold",
                    f"{threshold:.4f}"
                )

                col2.metric(
                    "Exceedances",
                    len(exceedances)
                )

                col3.metric(
                    "GPD Shape (ξ)",
                    f"{shape:.4f}"
                )

                col4.metric(
                    "GPD Scale (β)",
                    f"{scale:.4f}"
                )

                st.subheader("Threshold Visualization")

                fig = px.histogram(
                    values,
                    nbins=50,
                    title=f"Distribution of {selected_column}"
                )

                fig.add_vline(
                    x=threshold,
                    line_dash="dash",
                    annotation_text="Threshold"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader("Exceedance Distribution")

                fig2 = px.histogram(
                    exceedances,
                    nbins=30,
                    title="Values Exceeding the Threshold"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

                st.subheader("GPD Fitted Distribution")

                x = np.linspace(
                    0,
                    float(exceedances.max()),
                    200
                )

                fitted_pdf = genpareto.pdf(
                    x,
                    shape,
                    loc=0,
                    scale=scale
                )

                fig3 = px.histogram(
                    exceedances,
                    nbins=30,
                    histnorm="probability density",
                    title="Exceedances and GPD Density"
                )

                fig3.add_scatter(
                    x=x,
                    y=fitted_pdf,
                    mode="lines",
                    name="GPD Density"
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True
                )

                st.subheader("Tail Probability Estimates")

                probability = st.number_input(
                    "Enter an excess value above threshold",
                    min_value=0.0,
                    value=float(exceedances.median()),
                    step=1.0
                )

                tail_probability = genpareto.sf(
                    probability,
                    shape,
                    loc=0,
                    scale=scale
                )

                st.metric(
                    "Estimated Conditional Tail Probability",
                    f"{tail_probability:.6f}"
                )

                st.caption(
                    "This estimates the probability that an exceedance "
                    "above the threshold exceeds the entered excess value, "
                    "conditional on exceeding the threshold."
                )

                st.subheader("Return-Level Estimate")

                return_period = st.number_input(
                    "Return period (in observations)",
                    min_value=2,
                    max_value=1000000,
                    value=100,
                    step=10
                )

                exceedance_rate = len(exceedances) / len(values)

                if exceedance_rate > 0:

                    try:

                        p = 1 - (
                            1 / (return_period * exceedance_rate)
                        )

                        p = np.clip(p, 0.000001, 0.999999)

                        if abs(shape) > 1e-8:

                            return_level = (
                                threshold +
                                scale / shape *
                                ((1 - p) ** (-shape) - 1)
                            )

                        else:

                            return_level = (
                                threshold -
                                scale * np.log(1 - p)
                            )

                        st.metric(
                            "Estimated Return Level",
                            f"{return_level:.4f}"
                        )

                        st.caption(
                            "This is a model-based return-level estimate "
                            "under the fitted POT model and the chosen "
                            "threshold. The return period is expressed "
                            "in observation units, not calendar time."
                        )

                    except Exception as e:

                        st.error(
                            f"Unable to calculate return level: {e}"
                        )

                st.subheader("Model Parameters")

                parameters = pd.DataFrame({
                    "Parameter": [
                        "Threshold",
                        "Shape (ξ)",
                        "Scale (β)",
                        "Number of Exceedances",
                        "Threshold Percentile"
                    ],
                    "Value": [
                        threshold,
                        shape,
                        scale,
                        len(exceedances),
                        quantile
                    ]
                })

                st.dataframe(
                    parameters,
                    use_container_width=True
                )

                st.warning(
                    "These estimates are exploratory. Check threshold "
                    "stability, independence of exceedances, fit quality, "
                    "and confidence intervals before interpreting them "
                    "as research findings."
                )


# --------------------------------------------------
# COMPARISON DASHBOARD
# --------------------------------------------------
elif page == "Comparison Dashboard":

    st.subheader("⚖️ Original vs Deduplicated Comparison")

    if st.session_state.data is None:

        st.warning("Please upload a dataset first.")

    else:

        original = st.session_state.data.copy()
        deduplicated = remove_duplicates(original)

        numeric_cols = get_numeric_columns(original)

        if len(numeric_cols) == 0:

            st.error("No numeric columns available.")

        else:

            selected_column = st.selectbox(
                "Select Variable for Comparison",
                numeric_cols
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Original Rows",
                len(original)
            )

            col2.metric(
                "Deduplicated Rows",
                len(deduplicated)
            )

            comparison = pd.DataFrame({
                "Dataset": [
                    "Original",
                    "Deduplicated"
                ],
                "Mean": [
                    original[selected_column].mean(),
                    deduplicated[selected_column].mean()
                ],
                "Maximum": [
                    original[selected_column].max(),
                    deduplicated[selected_column].max()
                ],
                "Standard Deviation": [
                    original[selected_column].std(),
                    deduplicated[selected_column].std()
                ]
            })

            st.subheader("Summary Statistics")

            st.dataframe(
                comparison,
                use_container_width=True
            )

            fig = px.box(
                pd.concat([
                    original[[selected_column]].assign(
                        Dataset="Original"
                    ),
                    deduplicated[[selected_column]].assign(
                        Dataset="Deduplicated"
                    )
                ]),
                x="Dataset",
                y=selected_column,
                color="Dataset",
                title="Original vs Deduplicated Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("Extreme Threshold Comparison")

            q = st.slider(
                "Comparison percentile",
                0.80,
                0.99,
                0.95,
                0.01,
                key="comparison_quantile"
            )

            threshold_original = original[selected_column].quantile(q)

            threshold_deduplicated = (
                deduplicated[selected_column].quantile(q)
            )

            result_table = pd.DataFrame({
                "Dataset": [
                    "Original",
                    "Deduplicated"
                ],
                "Threshold": [
                    threshold_original,
                    threshold_deduplicated
                ]
            })

            st.dataframe(
                result_table,
                use_container_width=True
            )

            fig2 = px.bar(
                result_table,
                x="Dataset",
                y="Threshold",
                title="Threshold Comparison"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            st.caption(
                "This comparison is descriptive. For your research, "
                "compare fitted tail parameters and uncertainty "
                "under the same variable and threshold-selection "
                "procedure."
            )


# --------------------------------------------------
# RESEARCH RESULTS
# --------------------------------------------------
elif page == "Research Results":

    st.subheader("📑 Research Results Summary")

    if st.session_state.data is None:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state.data.copy()
        deduplicated = remove_duplicates(df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Observations",
            len(df)
        )

        col2.metric(
            "Exact Duplicates",
            int(df.duplicated().sum())
        )

        col3.metric(
            "Unique Observations",
            len(deduplicated)
        )

        col4.metric(
            "Columns",
            len(df.columns)
        )

        st.subheader("Dataset Overview")

        st.dataframe(
            df.describe().T,
            use_container_width=True
        )

        st.subheader("Duplicate Summary")

        duplicate_summary = pd.DataFrame({
            "Metric": [
                "Original Records",
                "Duplicate Records",
                "Unique Records"
            ],
            "Value": [
                len(df),
                int(df.duplicated().sum()),
                len(deduplicated)
            ]
        })

        st.dataframe(
            duplicate_summary,
            use_container_width=True
        )

        st.subheader("Attack Type Distribution")

        if "Attack_Type" in df.columns:

            attack_counts = df["Attack_Type"].value_counts().reset_index()

            attack_counts.columns = [
                "Attack_Type",
                "Count"
            ]

            fig = px.bar(
                attack_counts,
                x="Attack_Type",
                y="Count",
                title="Attack Type Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        elif "label" in df.columns:

            label_counts = df["label"].value_counts().reset_index()

            label_counts.columns = [
                "label",
                "Count"
            ]

            fig = px.bar(
                label_counts,
                x="label",
                y="Count",
                title="Label Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No recognized attack-type column was found. "
                "You can use the column selection in EVA to analyse "
                "numeric variables."
            )

        st.subheader("Export Dataset")

        csv = deduplicated.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Deduplicated CSV",
            data=csv,
            file_name="deduplicated_dataset.csv",
            mime="text/csv"
        )

        st.info(
            "This page summarizes the uploaded data. It does not "
            "automatically generate verified research conclusions."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("IoT Extreme Value Analysis Prototype")
st.sidebar.caption("POT | GPD | Duplicate Analysis")
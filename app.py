import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import (
    load_csv,
    validate_wind_dataset,
    validate_no2_dataset,
    validate_lst_dataset,
    preprocess_dataframe,
    merge_datasets,
    generate_sample_datasets,
    identify_pollution_columns,
    identify_wind_columns,
    identify_temperature_columns
)

from utils.statistics import (
    calculate_correlation_matrix,
    pearson_analysis,
    spearman_analysis,
    weather_pollution_significance,
    lag_correlation_analysis,
    calculate_basic_kpis,
    strongest_correlations
)

from utils.clustering import (
    build_pattern_recognition_pipeline,
    calculate_cluster_distribution,
    environmental_event_detector,
    get_top_anomalies
)

from utils.forecasting import (
    train_forecasting_model,
    forecast_future_values,
    get_feature_importance
)


st.set_page_config(
    page_title="Urban Environmental Correlation Engine",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Urban Environmental Correlation Engine")

st.markdown("""
Analyze relationships between:

- Wind Speed
- Land Surface Temperature (LST)
- NO₂ Pollution

Perform:

- Correlation Analysis
- Lag Analysis
- Statistical Significance Testing
- Pattern Recognition
- Anomaly Detection
- Predictive Forecasting
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("📂 Upload Datasets")

wind_file = st.sidebar.file_uploader(
    "Upload Wind Dataset",
    type=["csv"]
)

no2_file = st.sidebar.file_uploader(
    "Upload NO₂ Dataset",
    type=["csv"]
)

lst_file = st.sidebar.file_uploader(
    "Upload LST Dataset",
    type=["csv"]
)

use_sample_data = False

if (
    wind_file is None
    or no2_file is None
    or lst_file is None
):

    st.sidebar.warning(
        "Upload all three files or generate sample data."
    )

    if st.sidebar.button(
        "Generate & Load 3 Sample Interrelated Datasets"
    ):
        use_sample_data = True


# =====================================================
# DATA LOADING
# =====================================================

merged_df = None

try:

    if use_sample_data:

        wind_df, no2_df, lst_df = (
            generate_sample_datasets()
        )

        merged_df = merge_datasets(
            wind_df,
            no2_df,
            lst_df
        )

        st.success(
            "Synthetic datasets loaded successfully."
        )

    elif (
        wind_file is not None
        and no2_file is not None
        and lst_file is not None
    ):

        wind_df = load_csv(
            wind_file
        )

        no2_df = load_csv(
            no2_file
        )

        lst_df = load_csv(
            lst_file
        )

        validate_wind_dataset(
            wind_df
        )

        validate_no2_dataset(
            no2_df
        )

        validate_lst_dataset(
            lst_df
        )

        wind_df = preprocess_dataframe(
            wind_df
        )

        no2_df = preprocess_dataframe(
            no2_df
        )

        lst_df = preprocess_dataframe(
            lst_df
        )

        merged_df = merge_datasets(
            wind_df,
            no2_df,
            lst_df
        )

except KeyError as e:

    st.error(
        f"Missing required column: {e}"
    )

except ValueError as e:

    st.error(
        str(e)
    )

except Exception as e:

    st.error(
        f"Unexpected Error: {e}"
    )
# --- INSERT THIS IMMEDIATELY AFTER LOADING YOUR DATAFRAME ---
# Clean target columns and strip unit text strings
for col in df.columns:
    if df[col].dtype == 'object':
        # Strip out common unit text patterns safely
        df[col] = df[col].astype(str).str.replace('micromol/m2', '', case=False).str.strip()
        # Convert to numeric values, forcing remaining string errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle any NaN gaps created by the string conversions
df = df.ffill().bfill()
# -------------------------------------------------------------

# =====================================================
# STOP IF NO DATA
# =====================================================

if merged_df is None:

    st.info(
        "Upload all datasets or generate sample datasets to begin."
    )

    st.stop()

# =====================================================
# COLUMN IDENTIFICATION
# =====================================================

pollution_cols = identify_pollution_columns(
    merged_df
)

wind_cols = identify_wind_columns(
    merged_df
)

temperature_cols = identify_temperature_columns(
    merged_df
)

numeric_cols = merged_df.select_dtypes(
    include=np.number
).columns.tolist()

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Data Fusion & KPIs",
        "📈 Correlation Analytics",
        "📉 Lag & Significance",
        "🤖 Pattern & Prediction"
    ]
)
# =====================================================
# TAB 1 : DATA FUSION & KPIs
# =====================================================

with tab1:

    st.header("📊 Data Fusion & KPI Dashboard")

    st.success(
        f"Successfully merged {len(merged_df):,} matching records across all datasets."
    )

    st.subheader("Merged Dataset Preview")

    st.dataframe(
        merged_df.head(20),
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================
    # KPI CARDS
    # ==========================================

    st.subheader("📌 Key Performance Indicators")

    try:

        kpis = calculate_basic_kpis(
            merged_df
        )

        numeric_cols_available = list(
            kpis.keys()
        )

        if len(numeric_cols_available) >= 4:

            col1, col2, col3, col4 = st.columns(4)

            metric_1 = numeric_cols_available[0]
            metric_2 = numeric_cols_available[1]
            metric_3 = numeric_cols_available[2]
            metric_4 = numeric_cols_available[3]

            with col1:

                st.metric(
                    metric_1,
                    f"{kpis[metric_1]['mean']:.2f}"
                )

            with col2:

                st.metric(
                    metric_2,
                    f"{kpis[metric_2]['mean']:.2f}"
                )

            with col3:

                st.metric(
                    metric_3,
                    f"{kpis[metric_3]['mean']:.2f}"
                )

            with col4:

                st.metric(
                    metric_4,
                    f"{kpis[metric_4]['mean']:.2f}"
                )

    except Exception as e:

        st.error(
            f"KPI generation failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # DATASET INFORMATION
    # ==========================================

    st.subheader("📋 Dataset Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.metric(
            "Rows",
            merged_df.shape[0]
        )

    with info_col2:

        st.metric(
            "Columns",
            merged_df.shape[1]
        )

    with info_col3:

        st.metric(
            "Numeric Features",
            len(numeric_cols)
        )

    st.markdown("---")

    # ==========================================
    # MISSING VALUES
    # ==========================================

    st.subheader("🔍 Missing Values Audit")

    missing_df = pd.DataFrame({
        "Column":
            merged_df.columns,
        "Missing Values":
            merged_df.isna().sum().values,
        "Missing %":
            (
                merged_df.isna().mean()
                * 100
            ).round(2).values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================
    # DATE RANGE
    # ==========================================

    st.subheader("📅 Temporal Coverage")

    if "date" in merged_df.columns:

        date_col1, date_col2 = st.columns(2)

        with date_col1:

            st.metric(
                "Start Date",
                str(
                    merged_df["date"]
                    .min()
                    .date()
                )
            )

        with date_col2:

            st.metric(
                "End Date",
                str(
                    merged_df["date"]
                    .max()
                    .date()
                )
            )

    st.markdown("---")

    # ==========================================
    # NUMERIC SUMMARY
    # ==========================================

    st.subheader("📈 Descriptive Statistics")

    st.dataframe(
        merged_df.describe().T,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================
    # FEATURE DISTRIBUTION
    # ==========================================

    st.subheader("📊 Feature Distribution Explorer")

    selected_feature = st.selectbox(
        "Choose Variable",
        numeric_cols,
        key="dist_feature"
    )

    try:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.histplot(
            merged_df[selected_feature],
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"Distribution of {selected_feature}"
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Distribution chart failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # TIME SERIES OVERVIEW
    # ==========================================

    st.subheader("📈 Time Series Explorer")

    ts_variable = st.selectbox(
        "Select Variable",
        numeric_cols,
        key="timeseries_variable"
    )

    try:

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.plot(
            merged_df["date"],
            merged_df[ts_variable]
        )

        ax.set_title(
            f"{ts_variable} Over Time"
        )

        ax.set_xlabel("Date")

        ax.set_ylabel(
            ts_variable
        )

        plt.xticks(
            rotation=45
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Time series visualization failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # STRONGEST CORRELATIONS
    # ==========================================

    st.subheader(
        "🔥 Strongest Relationships Detected"
    )

    try:

        strong_corr = (
            strongest_correlations(
                merged_df,
                top_n=15
            )
        )

        st.dataframe(
            strong_corr,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Strong correlation extraction failed: {e}"
        )
  # =====================================================
# TAB 2 : CORRELATION ANALYTICS
# =====================================================

with tab2:

    st.header("📈 Correlation Analytics")

    st.markdown("""
    Explore relationships between environmental variables using:

    - Correlation Heatmaps
    - Pearson Correlation
    - Spearman Correlation
    - Scatter Analysis
    - Linear Trend Detection
    """)

    st.markdown("---")

    # ==========================================
    # CORRELATION MATRIX
    # ==========================================

    st.subheader("🔥 Correlation Heatmap")

    try:

        corr_matrix = calculate_correlation_matrix(
            merged_df
        )

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            ax=ax
        )

        ax.set_title(
            "Environmental Correlation Matrix"
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Heatmap generation failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # VARIABLE EXPLORER
    # ==========================================

    st.subheader("🔍 Variable Relationship Explorer")

    col1, col2 = st.columns(2)

    with col1:

        x_variable = st.selectbox(
            "Select X Variable",
            numeric_cols,
            key="x_variable"
        )

    with col2:

        y_variable = st.selectbox(
            "Select Y Variable",
            numeric_cols,
            index=min(
                1,
                len(numeric_cols) - 1
            ),
            key="y_variable"
        )

    st.markdown("---")

    # ==========================================
    # PEARSON ANALYSIS
    # ==========================================

    pearson_result = None

    try:

        pearson_result = pearson_analysis(
            merged_df,
            x_variable,
            y_variable
        )

        if pearson_result:

            p_col1, p_col2 = st.columns(2)

            with p_col1:

                st.metric(
                    "Pearson Correlation (r)",
                    f"{pearson_result['correlation']:.4f}"
                )

            with p_col2:

                st.metric(
                    "Pearson p-value",
                    f"{pearson_result['p_value']:.6f}"
                )

    except Exception as e:

        st.error(
            f"Pearson analysis failed: {e}"
        )

    # ==========================================
    # SPEARMAN ANALYSIS
    # ==========================================

    try:

        spearman_result = spearman_analysis(
            merged_df,
            x_variable,
            y_variable
        )

        if spearman_result:

            s_col1, s_col2 = st.columns(2)

            with s_col1:

                st.metric(
                    "Spearman Correlation (ρ)",
                    f"{spearman_result['correlation']:.4f}"
                )

            with s_col2:

                st.metric(
                    "Spearman p-value",
                    f"{spearman_result['p_value']:.6f}"
                )

    except Exception as e:

        st.error(
            f"Spearman analysis failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # SCATTER PLOT WITH TRENDLINE
    # ==========================================

    st.subheader("📊 Scatter Plot with Trendline")

    try:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.regplot(
            x=merged_df[x_variable],
            y=merged_df[y_variable],
            scatter_kws={"alpha": 0.7},
            line_kws={"linewidth": 2},
            ax=ax
        )

        ax.set_title(
            f"{x_variable} vs {y_variable}"
        )

        ax.set_xlabel(
            x_variable
        )

        ax.set_ylabel(
            y_variable
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Scatter plot failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # RELATIONSHIP INTERPRETATION
    # ==========================================

    st.subheader("🧠 Automated Interpretation")

    try:

        if pearson_result:

            corr_value = abs(
                pearson_result["correlation"]
            )

            if corr_value >= 0.8:

                strength = (
                    "Very Strong"
                )

            elif corr_value >= 0.6:

                strength = (
                    "Strong"
                )

            elif corr_value >= 0.4:

                strength = (
                    "Moderate"
                )

            elif corr_value >= 0.2:

                strength = (
                    "Weak"
                )

            else:

                strength = (
                    "Very Weak"
                )

            direction = (
                "Positive"
                if pearson_result["correlation"] > 0
                else "Negative"
            )

            significance = (
                "Statistically Significant"
                if pearson_result["p_value"] < 0.05
                else "Not Statistically Significant"
            )

            st.info(
                f"""
                Relationship Strength: {strength}

                Direction: {direction}

                Statistical Result: {significance}
                """
            )

    except Exception as e:

        st.error(
            f"Interpretation failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # MULTI-VARIABLE CORRELATION RANKING
    # ==========================================

    st.subheader(
        "🏆 Strongest Correlations in Dataset"
    )

    try:

        top_corr = strongest_correlations(
            merged_df,
            top_n=20
        )

        st.dataframe(
            top_corr,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Correlation ranking failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # POLLUTION RELATIONSHIPS
    # ==========================================

    st.subheader(
        "🌫 Pollution Relationship Analysis"
    )

    try:

        pollution_relationships = []

        for pollution in pollution_cols:

            for variable in numeric_cols:

                if variable == pollution:
                    continue

                result = pearson_analysis(
                    merged_df,
                    variable,
                    pollution
                )

                if result:

                    pollution_relationships.append(
                        {
                            "Pollution Variable":
                                pollution,
                            "Related Variable":
                                variable,
                            "Pearson_r":
                                round(
                                    result["correlation"],
                                    4
                                ),
                            "p_value":
                                round(
                                    result["p_value"],
                                    6
                                )
                        }
                    )

        if pollution_relationships:

            pollution_df = pd.DataFrame(
                pollution_relationships
            )

            pollution_df = (
                pollution_df.sort_values(
                    "Pearson_r",
                    key=abs,
                    ascending=False
                )
            )

            st.dataframe(
                pollution_df,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Pollution analysis failed: {e}"
        )

# =====================================================
# TAB 3 : LAG ANALYSIS & SIGNIFICANCE TESTING
# =====================================================

with tab3:

    st.header("📉 Lag Analysis & Statistical Significance")

    st.markdown("""
    Evaluate delayed environmental effects:

    - Wind → NO₂ Lag Effects
    - LST → NO₂ Lag Effects
    - 1-Period Lag Correlations
    - 2-Period Lag Correlations
    - Statistical Significance Testing
    """)

    st.markdown("---")

    # ==========================================
    # WEATHER VARIABLES
    # ==========================================

    weather_variables = (
        wind_cols +
        temperature_cols
    )

    # ==========================================
    # SIGNIFICANCE TESTING
    # ==========================================

    st.subheader(
        "🧪 Weather Impact Significance Testing"
    )

    try:

        significance_df = (
            weather_pollution_significance(
                merged_df,
                weather_variables,
                pollution_cols
            )
        )

        if not significance_df.empty:

            st.dataframe(
                significance_df,
                use_container_width=True
            )

        else:

            st.warning(
                "No significance results available."
            )

    except Exception as e:

        st.error(
            f"Significance analysis failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # SIGNIFICANCE SUMMARY CARDS
    # ==========================================

    st.subheader(
        "📋 Significance Summary"
    )

    try:

        if (
            "significance_df" in locals()
            and not significance_df.empty
        ):

            significant_count = (
                significance_df[
                    significance_df[
                        "Significance"
                    ]
                    ==
                    "Statistically Significant"
                ].shape[0]
            )

            total_tests = (
                significance_df.shape[0]
            )

            nonsignificant_count = (
                total_tests -
                significant_count
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Total Tests",
                    total_tests
                )

            with c2:

                st.metric(
                    "Significant",
                    significant_count
                )

            with c3:

                st.metric(
                    "Not Significant",
                    nonsignificant_count
                )

    except Exception as e:

        st.error(
            f"Summary card generation failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # LAG ANALYSIS
    # ==========================================

    st.subheader(
        "⏳ Lag Correlation Analysis"
    )

    try:

        lag_results = (
            lag_correlation_analysis(
                merged_df,
                weather_variables,
                pollution_cols,
                lags=[1, 2]
            )
        )

        if not lag_results.empty:

            st.dataframe(
                lag_results,
                use_container_width=True
            )

        else:

            st.warning(
                "No lag correlations available."
            )

    except Exception as e:

        st.error(
            f"Lag analysis failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # LAG VISUALIZATION
    # ==========================================

    st.subheader(
        "📈 Lag Correlation Visualization"
    )

    try:

        if (
            "lag_results" in locals()
            and not lag_results.empty
        ):

            lag_plot_data = (
                lag_results.copy()
            )

            fig, ax = plt.subplots(
                figsize=(12, 6)
            )

            sns.barplot(
                data=lag_plot_data,
                x="Lag",
                y="Correlation",
                hue="Target",
                ax=ax
            )

            ax.set_title(
                "Lag Correlation Strength"
            )

            st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Lag visualization failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # TOP LAG RELATIONSHIPS
    # ==========================================

    st.subheader(
        "🏆 Strongest Lag Relationships"
    )

    try:

        if (
            "lag_results" in locals()
            and not lag_results.empty
        ):

            strongest_lags = (
                lag_results.copy()
            )

            strongest_lags[
                "Absolute_Correlation"
            ] = (
                strongest_lags[
                    "Correlation"
                ].abs()
            )

            strongest_lags = (
                strongest_lags
                .sort_values(
                    "Absolute_Correlation",
                    ascending=False
                )
                .head(15)
            )

            st.dataframe(
                strongest_lags,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Strong lag extraction failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # ENVIRONMENTAL IMPACT MATRIX
    # ==========================================

    st.subheader(
        "🌍 Environmental Impact Matrix"
    )

    try:

        if (
            "significance_df" in locals()
            and not significance_df.empty
        ):

            impact_matrix = (
                significance_df.pivot_table(
                    index="Weather Variable",
                    columns="Pollution Variable",
                    values="Pearson_r"
                )
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            sns.heatmap(
                impact_matrix,
                annot=True,
                cmap="RdYlGn",
                center=0,
                fmt=".2f",
                ax=ax
            )

            ax.set_title(
                "Weather vs Pollution Impact Matrix"
            )

            st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Impact matrix failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # STATISTICAL INTERPRETATION
    # ==========================================

    st.subheader(
        "🧠 Statistical Interpretation"
    )

    try:

        if (
            "significance_df" in locals()
            and not significance_df.empty
        ):

            strongest = (
                significance_df.iloc[
                    significance_df[
                        "Pearson_r"
                    ]
                    .abs()
                    .idxmax()
                ]
            )

            st.info(
                f"""
Strongest Relationship Detected

Weather Variable:
{strongest['Weather Variable']}

Pollution Variable:
{strongest['Pollution Variable']}

Correlation:
{strongest['Pearson_r']}

P-Value:
{strongest['P_Value']}

Result:
{strongest['Significance']}
"""
            )

    except Exception as e:

        st.error(
            f"Interpretation failed: {e}"
        )

  # =====================================================
# TAB 4 : PATTERN RECOGNITION & PREDICTION
# =====================================================

with tab4:

    st.header("🤖 Pattern Recognition & Prediction Sandbox")

    st.markdown("""
    Advanced Analytics Modules:

    - KMeans Pattern Recognition
    - Cluster Profiling
    - Isolation Forest Anomaly Detection
    - Environmental Event Detection
    - Random Forest Forecasting
    - Feature Importance Analysis
    - Future NO₂ Prediction
    """)

    st.markdown("---")

    # ==========================================
    # KMEANS CLUSTERING
    # ==========================================

    st.subheader("🎯 KMeans Pattern Recognition")

    try:

        pattern_results = (
            build_pattern_recognition_pipeline(
                merged_df,
                n_clusters=4
            )
        )

        clustered_df = pattern_results[
            "clustered_data"
        ]

        cluster_profiles = pattern_results[
            "cluster_profiles"
        ]

        st.success(
            "Pattern recognition completed successfully."
        )

        st.markdown("### Cluster Profiles")

        st.dataframe(
            cluster_profiles,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Pattern recognition failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # CLUSTER VISUALIZATION
    # ==========================================

    st.subheader("📊 Cluster Visualization")

    try:

        numeric_cluster_cols = [
            col
            for col in clustered_df.columns
            if col not in [
                "Cluster",
                "Cluster_Label"
            ]
        ]

        if len(numeric_cluster_cols) >= 2:

            x_cluster = st.selectbox(
                "Cluster X Axis",
                numeric_cluster_cols,
                key="cluster_x"
            )

            y_cluster = st.selectbox(
                "Cluster Y Axis",
                numeric_cluster_cols,
                index=min(
                    1,
                    len(numeric_cluster_cols)-1
                ),
                key="cluster_y"
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            sns.scatterplot(
                data=clustered_df,
                x=x_cluster,
                y=y_cluster,
                hue="Cluster",
                palette="tab10",
                s=80,
                ax=ax
            )

            ax.set_title(
                "KMeans Environmental Clusters"
            )

            st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Cluster visualization failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # CLUSTER DISTRIBUTION
    # ==========================================

    st.subheader("📈 Cluster Distribution")

    try:

        distribution = (
            calculate_cluster_distribution(
                clustered_df
            )
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.pie(
            distribution["Percentage"],
            labels=distribution["Cluster"],
            autopct="%1.1f%%"
        )

        ax.set_title(
            "Cluster Distribution"
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Distribution calculation failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # ANOMALY DETECTION
    # ==========================================

    st.subheader("🚨 Isolation Forest Anomaly Detection")

    try:

        anomaly_df = pattern_results[
            "anomaly_data"
        ]

        anomalies = (
            get_top_anomalies(
                anomaly_df,
                n=15
            )
        )

        st.write(
            f"Detected {len(anomalies)} major anomalies."
        )

        st.dataframe(
            anomalies,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Anomaly detection failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # ENVIRONMENTAL EVENTS
    # ==========================================

    st.subheader("🌍 Environmental Event Detector")

    try:

        events = (
            environmental_event_detector(
                merged_df.select_dtypes(
                    include=np.number
                )
            )
        )

        if not events.empty:

            st.dataframe(
                events.head(20),
                use_container_width=True
            )

        else:

            st.info(
                "No major environmental events detected."
            )

    except Exception as e:

        st.error(
            f"Event detection failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # FORECAST TARGET SELECTION
    # ==========================================

    st.subheader("📈 Predictive Forecasting")

    try:

        if len(pollution_cols) == 0:

            st.warning(
                "No pollution columns available for forecasting."
            )

        else:

            forecast_target = st.selectbox(
                "Select NO₂ Variable",
                pollution_cols
            )

            forecast_results = (
                train_forecasting_model(
                    merged_df,
                    forecast_target
                )
            )

            model = forecast_results[
                "model"
            ]

            metrics = forecast_results[
                "metrics"
            ]

            prediction_df = forecast_results[
                "prediction_df"
            ]

            feature_columns = forecast_results[
                "feature_columns"
            ]

    except Exception as e:

        st.error(
            f"Forecast model failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # MODEL METRICS
    # ==========================================

    try:

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "R² Score",
                metrics["R2"]
            )

        with c2:

            st.metric(
                "MAE",
                metrics["MAE"]
            )

        with c3:

            st.metric(
                "RMSE",
                metrics["RMSE"]
            )

    except Exception:
        pass

    st.markdown("---")

    # ==========================================
    # ACTUAL VS PREDICTED
    # ==========================================

    st.subheader("📉 Actual vs Predicted")

    try:

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.plot(
            prediction_df["Actual"].values,
            label="Actual"
        )

        ax.plot(
            prediction_df["Predicted"].values,
            label="Predicted"
        )

        ax.legend()

        ax.set_title(
            "Actual vs Predicted NO₂"
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Prediction plot failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    st.subheader("⭐ Feature Importance")

    try:

        importance_df = (
            get_feature_importance(
                model,
                feature_columns
            )
        )

        st.dataframe(
            importance_df,
            use_container_width=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.barplot(
            data=importance_df.head(10),
            x="Importance",
            y="Feature",
            ax=ax
        )

        ax.set_title(
            "Top Feature Importance"
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Importance analysis failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # FUTURE FORECAST
    # ==========================================

    st.subheader("🔮 Future Forecast")

    try:

        forecast_df = (
            forecast_future_values(
                merged_df,
                forecast_target,
                periods=30
            )
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.plot(
            forecast_df["Date"],
            forecast_df["Forecast"]
        )

        ax.set_title(
            f"30-Day Forecast: {forecast_target}"
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"Future forecasting failed: {e}"
        )

    st.markdown("---")

    # ==========================================
    # DOWNLOAD MERGED DATA
    # ==========================================

    st.subheader("⬇ Download Results")

    try:

        csv = merged_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Merged Dataset",
            data=csv,
            file_name="merged_environmental_data.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Download preparation failed: {e}"
        )

# =====================================================
# END OF APPLICATION
# =====================================================

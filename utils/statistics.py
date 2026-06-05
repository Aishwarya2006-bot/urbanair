# utils/statistics.py

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


def get_numeric_columns(df):
    """
    Return all numeric columns.
    """
    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def calculate_correlation_matrix(df):
    """
    Correlation matrix for all numeric columns.
    """

    try:

        numeric_df = df.select_dtypes(
            include=np.number
        )

        return numeric_df.corr()

    except Exception as e:
        raise ValueError(
            f"Correlation matrix error: {e}"
        )


def pearson_analysis(
    df,
    col_x,
    col_y
):
    """
    Pearson correlation.
    """

    try:

        temp = df[
            [col_x, col_y]
        ].dropna()

        if len(temp) < 3:
            return None

        r, p = pearsonr(
            temp[col_x],
            temp[col_y]
        )

        return {
            "correlation": r,
            "p_value": p
        }

    except Exception as e:
        raise ValueError(
            f"Pearson analysis failed: {e}"
        )


def spearman_analysis(
    df,
    col_x,
    col_y
):
    """
    Spearman correlation.
    """

    try:

        temp = df[
            [col_x, col_y]
        ].dropna()

        if len(temp) < 3:
            return None

        rho, p = spearmanr(
            temp[col_x],
            temp[col_y]
        )

        return {
            "correlation": rho,
            "p_value": p
        }

    except Exception as e:
        raise ValueError(
            f"Spearman analysis failed: {e}"
        )


def significance_label(
    p_value
):
    """
    Statistical significance.
    """

    if p_value < 0.05:
        return "Statistically Significant"

    return "Not Statistically Significant"


def weather_pollution_significance(
    df,
    weather_cols,
    pollution_cols
):
    """
    Wind/LST vs NO2 significance.
    """

    results = []

    try:

        for weather in weather_cols:

            for pollution in pollution_cols:

                subset = df[
                    [weather, pollution]
                ].dropna()

                if len(subset) < 5:
                    continue

                r, p = pearsonr(
                    subset[weather],
                    subset[pollution]
                )

                results.append({
                    "Weather Variable": weather,
                    "Pollution Variable": pollution,
                    "Pearson_r": round(r, 4),
                    "P_Value": round(p, 6),
                    "Significance":
                        significance_label(p)
                })

        return pd.DataFrame(results)

    except Exception as e:
        raise ValueError(
            f"Weather significance analysis failed: {e}"
        )


def create_lag_columns(
    df,
    target_columns,
    lags=[1, 2]
):
    """
    Create lag features.
    """

    temp = df.copy()

    try:

        for col in target_columns:

            for lag in lags:

                temp[
                    f"{col}_lag_{lag}"
                ] = temp[col].shift(lag)

        return temp

    except Exception as e:
        raise ValueError(
            f"Lag feature creation failed: {e}"
        )


def lag_correlation_analysis(
    df,
    source_columns,
    target_columns,
    lags=[1, 2]
):
    """
    Analyze delayed effects.
    """

    results = []

    try:

        temp = create_lag_columns(
            df,
            source_columns,
            lags
        )

        for source in source_columns:

            for target in target_columns:

                for lag in lags:

                    lag_col = (
                        f"{source}_lag_{lag}"
                    )

                    subset = temp[
                        [lag_col, target]
                    ].dropna()

                    if len(subset) < 5:
                        continue

                    r, p = pearsonr(
                        subset[lag_col],
                        subset[target]
                    )

                    results.append({
                        "Source":
                            source,
                        "Target":
                            target,
                        "Lag":
                            f"{lag}",
                        "Correlation":
                            round(r, 4),
                        "P_Value":
                            round(p, 6),
                        "Significance":
                            significance_label(p)
                    })

        return pd.DataFrame(results)

    except Exception as e:
        raise ValueError(
            f"Lag analysis failed: {e}"
        )


def calculate_basic_kpis(df):
    """
    Generate KPI metrics.
    """

    try:

        numeric_cols = get_numeric_columns(df)

        kpis = {}

        for col in numeric_cols:

            kpis[col] = {
                "mean":
                    round(
                        df[col].mean(),
                        2
                    ),
                "max":
                    round(
                        df[col].max(),
                        2
                    ),
                "min":
                    round(
                        df[col].min(),
                        2
                    ),
                "std":
                    round(
                        df[col].std(),
                        2
                    )
            }

        return kpis

    except Exception as e:
        raise ValueError(
            f"KPI calculation failed: {e}"
        )


def pollution_summary(
    df,
    pollution_columns
):
    """
    Summary statistics
    for NO2 variables.
    """

    try:

        results = []

        for col in pollution_columns:

            results.append({
                "Pollution Variable": col,
                "Mean":
                    round(
                        df[col].mean(),
                        2
                    ),
                "Median":
                    round(
                        df[col].median(),
                        2
                    ),
                "Maximum":
                    round(
                        df[col].max(),
                        2
                    ),
                "Minimum":
                    round(
                        df[col].min(),
                        2
                    ),
                "Std Dev":
                    round(
                        df[col].std(),
                        2
                    )
            })

        return pd.DataFrame(results)

    except Exception as e:
        raise ValueError(
            f"Pollution summary failed: {e}"
        )


def environmental_risk_score(
    value
):
    """
    Pollution risk classification.
    """

    if value < 40:
        return "Low Risk"

    elif value < 80:
        return "Moderate Risk"

    elif value < 120:
        return "High Risk"

    return "Extreme Risk"


def create_risk_table(
    df,
    pollution_column
):
    """
    Assign risk classes.
    """

    try:

        risk_df = df[
            ["date", pollution_column]
        ].copy()

        risk_df["Risk_Level"] = (
            risk_df[
                pollution_column
            ].apply(
                environmental_risk_score
            )
        )

        return risk_df

    except Exception as e:
        raise ValueError(
            f"Risk table failed: {e}"
        )


def strongest_correlations(
    df,
    top_n=10
):
    """
    Find strongest correlations.
    """

    try:

        corr = (
            df.select_dtypes(
                include=np.number
            )
            .corr()
            .abs()
        )

        pairs = []

        cols = corr.columns

        for i in range(len(cols)):

            for j in range(i + 1, len(cols)):

                pairs.append({
                    "Variable_1":
                        cols[i],
                    "Variable_2":
                        cols[j],
                    "Correlation":
                        corr.iloc[i, j]
                })

        result = pd.DataFrame(
            pairs
        )

        result = result.sort_values(
            "Correlation",
            ascending=False
        )

        return result.head(top_n)

    except Exception as e:
        raise ValueError(
            f"Strong correlation extraction failed: {e}"
        )

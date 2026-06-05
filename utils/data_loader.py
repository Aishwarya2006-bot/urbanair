# utils/data_loader.py

import pandas as pd
import numpy as np
import streamlit as st


REQUIRED_WIND_COLUMNS = ["date"]
REQUIRED_NO2_COLUMNS = ["date"]
REQUIRED_LST_COLUMNS = ["date"]


@st.cache_data
def load_csv(uploaded_file):
    """
    Safely load CSV file.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df

    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


def validate_wind_dataset(df):
    """
    Validate Wind dataset.
    """
    try:
        if "date" not in df.columns:
            raise KeyError(
                "Wind dataset must contain a 'date' column."
            )

        return True

    except Exception as e:
        raise ValueError(str(e))


def validate_no2_dataset(df):
    """
    Validate NO2 dataset.
    """
    try:
        if "date" not in df.columns:
            raise KeyError(
                "NO2 dataset must contain a 'date' column."
            )

        return True

    except Exception as e:
        raise ValueError(str(e))


def validate_lst_dataset(df):
    """
    Validate LST dataset.
    """
    try:
        if "date" not in df.columns:
            raise KeyError(
                "LST dataset must contain a 'date' column."
            )

        return True

    except Exception as e:
        raise ValueError(str(e))


def preprocess_dataframe(df):
    """
    Standard preprocessing.
    """

    df = df.copy()

    try:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.dropna(subset=["date"])

        df = df.drop_duplicates(
            subset=["date"]
        )

        df = df.sort_values(
            "date"
        ).reset_index(drop=True)

        return df

    except Exception as e:
        raise ValueError(
            f"Date preprocessing failed: {e}"
        )


@st.cache_data
def merge_datasets(
    wind_df,
    no2_df,
    lst_df
):
    """
    Merge all datasets using INNER JOIN.
    """

    try:

        merged = pd.merge(
            wind_df,
            no2_df,
            on="date",
            how="inner"
        )

        merged = pd.merge(
            merged,
            lst_df,
            on="date",
            how="inner"
        )

        merged = merged.sort_values(
            "date"
        ).reset_index(drop=True)

        return merged

    except Exception as e:
        raise ValueError(
            f"Dataset merge failed: {e}"
        )


@st.cache_data
def generate_sample_datasets(
    n_samples=365,
    seed=42
):
    """
    Generate realistic synthetic
    environmental datasets.
    """

    np.random.seed(seed)

    dates = pd.date_range(
        start="2023-01-01",
        periods=n_samples,
        freq="D"
    )

    seasonal = (
        np.sin(
            np.linspace(
                0,
                8 * np.pi,
                n_samples
            )
        )
    )

    # -----------------------------
    # WIND DATASET
    # -----------------------------

    wind_speed = (
        15
        + 5 * seasonal
        + np.random.normal(
            0,
            2,
            n_samples
        )
    )

    wind_speed = np.clip(
        wind_speed,
        2,
        None
    )

    wind_df = pd.DataFrame({
        "date": dates,
        "Bangalore_Urban_speed": wind_speed,
        "Bangalore_Rural_speed":
            wind_speed + np.random.normal(
                0,
                1,
                n_samples
            ),
        "Delhi_Urban_speed":
            wind_speed + np.random.normal(
                1,
                1.5,
                n_samples
            ),
        "Delhi_Rural_speed":
            wind_speed + np.random.normal(
                -1,
                1,
                n_samples
            )
    })

    # -----------------------------
    # LST DATASET
    # -----------------------------

    lst_base = (
        30
        + 8 * seasonal
        + np.random.normal(
            0,
            1.5,
            n_samples
        )
    )

    lst_df = pd.DataFrame({
        "date": dates,
        "Bangalore_Urban":
            lst_base + 2,
        "Bangalore_Rural":
            lst_base - 1,
        "Delhi_Urban":
            lst_base + 4,
        "Delhi_Rural":
            lst_base + 1
    })

    # -----------------------------
    # NO2 DATASET
    # -----------------------------

    pollution = (
        80
        + 0.8 * lst_base
        - 1.5 * wind_speed
        + np.random.normal(
            0,
            5,
            n_samples
        )
    )

    pollution = np.clip(
        pollution,
        10,
        None
    )

    no2_df = pd.DataFrame({
        "date": dates,
        "Bangalore_Urban_NO2":
            pollution,
        "Bangalore_Rural_NO2":
            pollution - 8,
        "Delhi_Urban_NO2":
            pollution + 15,
        "Delhi_Rural_NO2":
            pollution + 5
    })

    return (
        preprocess_dataframe(wind_df),
        preprocess_dataframe(no2_df),
        preprocess_dataframe(lst_df)
    )


def get_numeric_columns(df):
    """
    Return numeric columns.
    """

    return df.select_dtypes(
        include=[
            np.number
        ]
    ).columns.tolist()


def create_lag_features(
    df,
    columns,
    lags=[1, 2]
):
    """
    Create lag variables.
    """

    df = df.copy()

    try:

        for col in columns:

            for lag in lags:

                df[
                    f"{col}_lag_{lag}"
                ] = df[col].shift(lag)

        return df

    except Exception as e:
        raise ValueError(
            f"Lag feature creation failed: {e}"
        )


def identify_pollution_columns(df):
    """
    Auto-detect NO2 columns.
    """

    return [
        col
        for col in df.columns
        if "NO2" in col
    ]


def identify_wind_columns(df):
    """
    Auto-detect wind columns.
    """

    return [
        col
        for col in df.columns
        if "speed" in col.lower()
    ]


def identify_temperature_columns(df):
    """
    Auto-detect LST columns.
    """

    lst_cols = []

    for col in df.columns:

        if (
            "NO2" not in col
            and "speed" not in col.lower()
            and col != "date"
        ):
            lst_cols.append(col)

    return lst_cols

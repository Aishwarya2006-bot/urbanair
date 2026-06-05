# utils/forecasting.py

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


def create_lag_features(
    df,
    columns,
    lags=[1, 2]
):
    """
    Create lag variables.
    """

    try:

        temp = df.copy()

        for col in columns:

            for lag in lags:

                temp[
                    f"{col}_lag_{lag}"
                ] = temp[col].shift(lag)

        return temp

    except Exception as e:
        raise ValueError(
            f"Lag feature generation failed: {e}"
        )


def identify_pollution_columns(
    df
):
    """
    Detect NO2 columns.
    """

    return [
        col
        for col in df.columns
        if "NO2" in col
    ]


def identify_weather_columns(
    df
):
    """
    Detect Wind + LST variables.
    """

    weather_cols = []

    for col in df.columns:

        if col == "date":
            continue

        if "NO2" in col:
            continue

        weather_cols.append(col)

    return weather_cols


def build_feature_dataset(
    df,
    target_column
):
    """
    Build ML-ready dataset.
    """

    try:

        weather_cols = (
            identify_weather_columns(df)
        )

        temp = create_lag_features(
            df,
            [target_column]
        )

        feature_cols = (
            weather_cols
            + [
                f"{target_column}_lag_1",
                f"{target_column}_lag_2"
            ]
        )

        temp = temp.dropna()

        X = temp[feature_cols]

        y = temp[target_column]

        return (
            X,
            y,
            temp,
            feature_cols
        )

    except Exception as e:
        raise ValueError(
            f"Feature engineering failed: {e}"
        )


def train_forecasting_model(
    df,
    target_column,
    test_size=0.2,
    random_state=42
):
    """
    Train Random Forest model.
    """

    try:

        (
            X,
            y,
            temp,
            feature_cols
        ) = build_feature_dataset(
            df,
            target_column
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                shuffle=False
            )
        )

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        metrics = {
            "R2": round(r2, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4)
        }

        prediction_df = pd.DataFrame({
            "Actual":
                y_test.values,
            "Predicted":
                predictions
        })

        prediction_df.index = (
            y_test.index
        )

        return {
            "model": model,
            "metrics": metrics,
            "prediction_df":
                prediction_df,
            "feature_columns":
                feature_cols,
            "X_test":
                X_test,
            "y_test":
                y_test
        }

    except Exception as e:
        raise ValueError(
            f"Model training failed: {e}"
        )


def get_feature_importance(
    model,
    feature_columns
):
    """
    Random Forest importance.
    """

    try:

        importance_df = pd.DataFrame({
            "Feature":
                feature_columns,
            "Importance":
                model.feature_importances_
        })

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        return importance_df

    except Exception as e:
        raise ValueError(
            f"Feature importance failed: {e}"
        )


def forecast_future_values(
    df,
    target_column,
    periods=30
):
    """
    Forecast future NO2 values.
    """

    try:

        training = train_forecasting_model(
            df,
            target_column
        )

        model = training["model"]

        weather_cols = (
            identify_weather_columns(df)
        )

        temp = create_lag_features(
            df,
            [target_column]
        )

        temp = temp.dropna()

        future_records = []

        current = temp.copy()

        for step in range(periods):

            latest_row = (
                current.iloc[-1]
            )

            feature_vector = []

            for col in weather_cols:

                feature_vector.append(
                    latest_row[col]
                )

            feature_vector.append(
                latest_row[
                    f"{target_column}_lag_1"
                ]
            )

            feature_vector.append(
                latest_row[
                    f"{target_column}_lag_2"
                ]
            )

            prediction = (
                model.predict(
                    [feature_vector]
                )[0]
            )

            future_records.append(
                prediction
            )

            next_row = (
                latest_row.copy()
            )

            next_row[
                f"{target_column}_lag_2"
            ] = next_row[
                f"{target_column}_lag_1"
            ]

            next_row[
                f"{target_column}_lag_1"
            ] = prediction

            current.loc[
                len(current)
            ] = next_row

        future_dates = (
            pd.date_range(
                start=df["date"].max()
                + pd.Timedelta(
                    days=1
                ),
                periods=periods,
                freq="D"
            )
        )

        forecast_df = pd.DataFrame({
            "Date":
                future_dates,
            "Forecast":
                future_records
        })

        return forecast_df

    except Exception as e:
        raise ValueError(
            f"Forecast generation failed: {e}"
        )


def train_all_pollution_models(
    df
):
    """
    Train models for all NO2 variables.
    """

    try:

        pollution_cols = (
            identify_pollution_columns(df)
        )

        results = {}

        for target in pollution_cols:

            results[target] = (
                train_forecasting_model(
                    df,
                    target
                )
            )

        return results

    except Exception as e:
        raise ValueError(
            f"Multi-model training failed: {e}"
        )


def generate_all_forecasts(
    df,
    periods=30
):
    """
    Generate forecasts for all
    NO2 variables.
    """

    try:

        pollution_cols = (
            identify_pollution_columns(df)
        )

        forecasts = {}

        for target in pollution_cols:

            forecasts[target] = (
                forecast_future_values(
                    df,
                    target,
                    periods
                )
            )

        return forecasts

    except Exception as e:
        raise ValueError(
            f"Forecast pipeline failed: {e}"
        )


def prediction_summary(
    forecast_df
):
    """
    Forecast statistics.
    """

    try:

        return {
            "Average Forecast":
                round(
                    forecast_df[
                        "Forecast"
                    ].mean(),
                    2
                ),
            "Maximum Forecast":
                round(
                    forecast_df[
                        "Forecast"
                    ].max(),
                    2
                ),
            "Minimum Forecast":
                round(
                    forecast_df[
                        "Forecast"
                    ].min(),
                    2
                )
        }

    except Exception as e:
        raise ValueError(
            f"Forecast summary failed: {e}"
        )

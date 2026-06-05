# utils/clustering.py

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def prepare_clustering_data(df):
    """
    Extract numeric columns and
    prepare data for clustering.
    """

    try:

        numeric_df = df.select_dtypes(
            include=np.number
        )

        numeric_df = numeric_df.dropna()

        return numeric_df

    except Exception as e:
        raise ValueError(
            f"Clustering preparation failed: {e}"
        )


def scale_features(df):
    """
    Standardize features.
    """

    try:

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(df)

        return scaled_data, scaler

    except Exception as e:
        raise ValueError(
            f"Feature scaling failed: {e}"
        )


def run_kmeans_clustering(
    df,
    n_clusters=4,
    random_state=42
):
    """
    Perform KMeans clustering.
    """

    try:

        numeric_df = prepare_clustering_data(df)

        scaled_data, scaler = scale_features(
            numeric_df
        )

        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

        clusters = model.fit_predict(
            scaled_data
        )

        result_df = numeric_df.copy()

        result_df["Cluster"] = clusters

        return (
            result_df,
            model,
            scaler
        )

    except Exception as e:
        raise ValueError(
            f"KMeans clustering failed: {e}"
        )


def create_cluster_profiles(
    clustered_df
):
    """
    Create cluster summary profiles.
    """

    try:

        profile = (
            clustered_df
            .groupby("Cluster")
            .mean()
            .round(2)
        )

        profile["Records"] = (
            clustered_df
            .groupby("Cluster")
            .size()
        )

        return profile

    except Exception as e:
        raise ValueError(
            f"Cluster profiling failed: {e}"
        )


def assign_cluster_labels(
    clustered_df
):
    """
    Generate human-readable labels.
    """

    try:

        profile = create_cluster_profiles(
            clustered_df
        )

        labels = {}

        for cluster_id in profile.index:

            row = profile.loc[
                cluster_id
            ]

            highest_feature = (
                row.drop(
                    labels=["Records"],
                    errors="ignore"
                )
                .idxmax()
            )

            if "NO2" in highest_feature:

                label = (
                    "Pollution Spike Event"
                )

            elif "speed" in highest_feature.lower():

                label = (
                    "High Wind Dispersion"
                )

            else:

                label = (
                    "Urban Heat Pattern"
                )

            labels[
                cluster_id
            ] = label

        clustered_df[
            "Cluster_Label"
        ] = clustered_df[
            "Cluster"
        ].map(labels)

        return clustered_df

    except Exception as e:
        raise ValueError(
            f"Cluster labeling failed: {e}"
        )


def calculate_cluster_centers(
    model,
    scaler,
    feature_names
):
    """
    Return cluster centers in
    original scale.
    """

    try:

        centers = scaler.inverse_transform(
            model.cluster_centers_
        )

        centers_df = pd.DataFrame(
            centers,
            columns=feature_names
        )

        centers_df.index.name = (
            "Cluster"
        )

        return centers_df

    except Exception as e:
        raise ValueError(
            f"Cluster center calculation failed: {e}"
        )


def run_isolation_forest(
    df,
    contamination=0.05,
    random_state=42
):
    """
    Detect anomalies.
    """

    try:

        numeric_df = prepare_clustering_data(
            df
        )

        model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )

        anomaly_labels = (
            model.fit_predict(
                numeric_df
            )
        )

        anomaly_scores = (
            model.decision_function(
                numeric_df
            )
        )

        result_df = numeric_df.copy()

        result_df[
            "Anomaly"
        ] = anomaly_labels

        result_df[
            "Anomaly_Score"
        ] = anomaly_scores

        return (
            result_df,
            model
        )

    except Exception as e:
        raise ValueError(
            f"Isolation Forest failed: {e}"
        )


def get_anomalies(
    anomaly_df
):
    """
    Return anomaly records only.
    """

    try:

        anomalies = anomaly_df[
            anomaly_df["Anomaly"] == -1
        ]

        return anomalies.sort_values(
            "Anomaly_Score"
        )

    except Exception as e:
        raise ValueError(
            f"Anomaly extraction failed: {e}"
        )


def calculate_cluster_distribution(
    clustered_df
):
    """
    Cluster percentages.
    """

    try:

        distribution = (
            clustered_df["Cluster"]
            .value_counts(
                normalize=True
            )
            .mul(100)
            .round(2)
            .reset_index()
        )

        distribution.columns = [
            "Cluster",
            "Percentage"
        ]

        return distribution

    except Exception as e:
        raise ValueError(
            f"Distribution calculation failed: {e}"
        )


def environmental_event_detector(
    df
):
    """
    Detect environmental events.
    """

    try:

        events = []

        for column in df.columns:

            if (
                df[column].dtype
                != "object"
            ):

                mean_val = (
                    df[column].mean()
                )

                std_val = (
                    df[column].std()
                )

                upper_limit = (
                    mean_val
                    + 2 * std_val
                )

                event_rows = df[
                    df[column]
                    > upper_limit
                ]

                for idx in event_rows.index:

                    events.append(
                        {
                            "Index": idx,
                            "Feature": column,
                            "Value":
                                event_rows.loc[
                                    idx,
                                    column
                                ],
                            "Threshold":
                                upper_limit
                        }
                    )

        return pd.DataFrame(events)

    except Exception as e:
        raise ValueError(
            f"Environmental event detection failed: {e}"
        )


def get_top_anomalies(
    anomaly_df,
    n=10
):
    """
    Return most unusual records.
    """

    try:

        anomalies = anomaly_df[
            anomaly_df["Anomaly"] == -1
        ]

        return anomalies.nsmallest(
            n,
            "Anomaly_Score"
        )

    except Exception as e:
        raise ValueError(
            f"Top anomaly extraction failed: {e}"
        )


def build_pattern_recognition_pipeline(
    df,
    n_clusters=4
):
    """
    Complete pattern recognition workflow.
    """

    try:

        clustered_df, model, scaler = (
            run_kmeans_clustering(
                df,
                n_clusters=n_clusters
            )
        )

        clustered_df = (
            assign_cluster_labels(
                clustered_df
            )
        )

        profile = (
            create_cluster_profiles(
                clustered_df
            )
        )

        centers = (
            calculate_cluster_centers(
                model,
                scaler,
                clustered_df
                .drop(
                    columns=[
                        "Cluster",
                        "Cluster_Label"
                    ],
                    errors="ignore"
                )
                .columns
            )
        )

        anomaly_df, anomaly_model = (
            run_isolation_forest(df)
        )

        anomalies = (
            get_anomalies(
                anomaly_df
            )
        )

        return {
            "clustered_data":
                clustered_df,
            "cluster_profiles":
                profile,
            "cluster_centers":
                centers,
            "anomaly_data":
                anomaly_df,
            "anomalies":
                anomalies,
            "kmeans_model":
                model,
            "isolation_model":
                anomaly_model
        }

    except Exception as e:
        raise ValueError(
            f"Pattern recognition pipeline failed: {e}"
        )

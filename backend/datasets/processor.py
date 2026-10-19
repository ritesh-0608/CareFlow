from __future__ import annotations
from io import BytesIO
from backend.storage import download_dataset
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from ml.features.hospital_features import create_hospital_features
from ml.features.resource_features import create_resource_features


MODEL_VERSION = "xgboost-v1"

XGB_CONFIG = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate validation/test metrics for a forecast model.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    non_zero = y_true != 0

    if non_zero.any():
        mape = (
            np.mean(
                np.abs(
                    (y_true[non_zero] - y_pred[non_zero])
                    / y_true[non_zero]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape) if not np.isnan(mape) else None,
    }


def chronological_split(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series dataset chronologically into:

    70% training
    15% validation
    15% test
    """

    data = (
        dataframe
        .sort_values("date")
        .reset_index(drop=True)
    )

    n = len(data)

    if n < 20:
        raise ValueError(
            "At least 20 usable observations are required "
            "for chronological model evaluation."
        )

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = data.iloc[:train_end].copy()
    validation = data.iloc[
        train_end:validation_end
    ].copy()
    test = data.iloc[validation_end:].copy()

    return train, validation, test


def build_xgboost_model() -> XGBRegressor:
    """
    Create the project's established XGBoost configuration.
    """

    return XGBRegressor(**XGB_CONFIG)


def prepare_hospital_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the established CareFlow hospital features.
    """

    required_columns = {
        "date",
        "admissions",
        "icu_demand",
        "emergency_visits",
        "avg_length_of_stay",
        "outpatient_visits",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Hospital dataset is missing required processing "
            f"columns: {sorted(missing)}"
        )

    data = dataframe.copy()

    return create_hospital_features(data)


def prepare_resource_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the established CareFlow resource features.
    """

    required_columns = {
        "date",
        "resource_category",
        "resource_name",
        "unit",
        "quantity_available",
        "quantity_used",
        "reference_capacity",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Resource dataset is missing required processing "
            f"columns: {sorted(missing)}"
        )

    data = dataframe.copy()

    return create_resource_features(data)


def train_hospital_model(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Train and evaluate one hospital forecasting model.

    The final test period remains separate from model selection.
    """

    if target_column not in {
        "admissions",
        "icu_demand",
    }:
        raise ValueError(
            f"Unsupported hospital target: {target_column}"
        )

    data = prepare_hospital_features(dataframe)

    data = data.dropna().copy()

    if len(data) < 20:
        raise ValueError(
            f"Not enough usable {target_column} observations "
            "after feature generation."
        )

    train, validation, test = chronological_split(data)

    if target_column == "admissions":
        feature_columns = [
            "day_of_week",
            "day_of_month",
            "month",
            "week_of_year",
            "admissions_lag_1",
            "admissions_lag_7",
            "admissions_lag_14",
            "admissions_lag_30",
            "admissions_rolling_mean_7",
            "admissions_rolling_mean_14",
            "admissions_rolling_mean_30",
        ]
    else:
        feature_columns = [
            "day_of_week",
            "day_of_month",
            "month",
            "week_of_year",
            "icu_demand_lag_1",
            "icu_demand_lag_7",
            "icu_demand_lag_14",
            "icu_demand_lag_30",
            "icu_demand_rolling_mean_7",
            "icu_demand_rolling_mean_14",
            "icu_demand_rolling_mean_30",
        ]

    X_train = train[feature_columns]
    y_train = train[target_column]

    X_validation = validation[feature_columns]
    y_validation = validation[target_column]

    X_test = test[feature_columns]
    y_test = test[target_column]

    model = build_xgboost_model()

    model.fit(
        X_train,
        y_train,
    )

    validation_predictions = model.predict(
        X_validation
    )

    test_predictions = model.predict(
        X_test
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    return {
        "model": model,
        "model_name": "XGBoost",
        "model_version": MODEL_VERSION,
        "target": target_column,
        "feature_columns": feature_columns,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_period_start": train["date"].min().date(),
        "training_period_end": train["date"].max().date(),
        "validation_period_start": validation["date"].min().date(),
        "validation_period_end": validation["date"].max().date(),
        "test_period_start": test["date"].min().date(),
        "test_period_end": test["date"].max().date(),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }


def train_resource_model(
    dataframe: pd.DataFrame,
    resource_name: str,
) -> dict[str, Any]:
    """
    Train and evaluate one resource-specific forecasting model.
    """

    data = dataframe.copy()

    data = data[
        data["resource_name"] == resource_name
    ].copy()

    if data.empty:
        raise ValueError(
            f"No records found for resource: {resource_name}"
        )

    data = prepare_resource_features(data)

    data = data.dropna().copy()

    if len(data) < 20:
        raise ValueError(
            f"Not enough usable observations for resource "
            f"'{resource_name}' after feature generation."
        )

    train, validation, test = chronological_split(data)

    excluded_columns = {
        "date",
        "resource_category",
        "resource_name",
        "unit",
        "quantity_used",
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded_columns
    ]

    X_train = train[feature_columns]
    y_train = train["quantity_used"]

    X_validation = validation[feature_columns]
    y_validation = validation["quantity_used"]

    X_test = test[feature_columns]
    y_test = test["quantity_used"]

    model = build_xgboost_model()

    model.fit(
        X_train,
        y_train,
    )

    validation_predictions = model.predict(
        X_validation
    )

    test_predictions = model.predict(
        X_test
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    return {
        "model": model,
        "model_name": "XGBoost",
        "model_version": MODEL_VERSION,
        "target": "quantity_used",
        "resource_name": resource_name,
        "feature_columns": feature_columns,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_period_start": train["date"].min().date(),
        "training_period_end": train["date"].max().date(),
        "validation_period_start": validation["date"].min().date(),
        "validation_period_end": validation["date"].max().date(),
        "test_period_start": test["date"].min().date(),
        "test_period_end": test["date"].max().date(),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
def save_uploaded_file(
    file_bytes: bytes,
    filename: str,
    user_id: int,
) -> str:
    """
    Save an uploaded dataset to local development storage.

    Files are isolated by user ID and stored outside PostgreSQL.
    """

    storage_root = (
        Path(__file__).resolve().parent.parent
        / "uploads"
        / "datasets"
    )

    user_directory = storage_root / str(user_id)
    user_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = user_directory / filename

    file_path.write_bytes(file_bytes)

    return str(file_path)

def read_uploaded_csv(
    file_path: str,
) -> pd.DataFrame:
    """
    Read a stored CSV dataset from Supabase Storage or a local file.
    """
    file_path = str(file_path)
    storage_prefix = "supabase://"

    if file_path.startswith(storage_prefix):
        storage_path = file_path[len(storage_prefix):]
        file_bytes = download_dataset(storage_path)
        return pd.read_csv(BytesIO(file_bytes))

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    return pd.read_csv(path)

def forecast_hospital(
    dataframe: pd.DataFrame,
    target_column: str,
    horizon: int,
) -> dict[str, Any]:
    """
    Generate a recursive hospital forecast using the
    established CareFlow XGBoost methodology.
    """

    if target_column not in {
        "admissions",
        "icu_demand",
    }:
        raise ValueError(
            f"Unsupported hospital target: {target_column}"
        )

    if horizon not in {7, 14, 30}:
        raise ValueError(
            "Hospital forecast horizon must be 7, 14, or 30 days."
        )

    data = prepare_hospital_features(
        dataframe
    )

    data = data.sort_values(
        "date"
    ).reset_index(drop=True)

    data = data.dropna(
        subset=[
            target_column,
        ]
    ).copy()

    if len(data) < 30:
        raise ValueError(
            f"At least 30 usable observations are required "
            f"for {target_column} forecasting."
        )

    training_result = train_hospital_model(
        dataframe,
        target_column,
    )

    model = training_result["model"]
    feature_columns = training_result["feature_columns"]

    history = data[target_column].astype(float).tolist()

    last_date = pd.Timestamp(
        data["date"].max()
    )

    predictions = []

    for step in range(1, horizon + 1):

        forecast_date = (
            last_date
            + pd.Timedelta(days=step)
        )

        history_series = pd.Series(
            history,
            dtype=float,
        )

        feature_row = {
            "day_of_week": forecast_date.dayofweek,
            "day_of_month": forecast_date.day,
            "month": forecast_date.month,
            "week_of_year": forecast_date.isocalendar().week,
            f"{target_column}_lag_1": history_series.iloc[-1],
            f"{target_column}_lag_7": history_series.iloc[-7],
            f"{target_column}_lag_14": history_series.iloc[-14],
            f"{target_column}_lag_30": history_series.iloc[-30],
            f"{target_column}_rolling_mean_7":
                history_series.iloc[-7:].mean(),
            f"{target_column}_rolling_mean_14":
                history_series.iloc[-14:].mean(),
            f"{target_column}_rolling_mean_30":
                history_series.iloc[-30:].mean(),
        }

        feature_row = pd.DataFrame(
            [feature_row]
        )

        feature_row = feature_row[
            feature_columns
        ]

        prediction = float(
            model.predict(
                feature_row
            )[0]
        )

        prediction = max(
            prediction,
            0.0,
        )

        predictions.append(
            {
                "date": forecast_date.date().isoformat(),
                "predicted_value": prediction,
            }
        )

        history.append(
            prediction
        )

    return {
        "forecast_type": target_column,
        "horizon": horizon,
        "model_name": training_result["model_name"],
        "model_version": training_result["model_version"],
        "predictions": predictions,
        "validation_metrics": training_result[
            "validation_metrics"
        ],
        "test_metrics": training_result[
            "test_metrics"
        ],
        "training_period_start": training_result[
            "training_period_start"
        ].isoformat(),
        "training_period_end": training_result[
            "training_period_end"
        ].isoformat(),
    }
def forecast_resource(dataframe, resource_name, horizon):
    if horizon not in {7, 14, 30}:
        raise ValueError("Horizon must be one of: 7, 14, 30.")

    data = prepare_resource_features(dataframe)

    data = data[
        data["resource_name"].astype(str).str.strip() == str(resource_name).strip()
    ].copy()

    data = data.sort_values("date").reset_index(drop=True)

    if data.empty:
        raise ValueError(f"No data found for resource: {resource_name}")

    data = data.dropna(subset=["quantity_used"]).copy()

    if len(data) < 30:
        raise ValueError(
            f"Insufficient historical data for resource '{resource_name}'. "
            "At least 30 observations are required."
        )

    training_result = train_resource_model(dataframe, resource_name)

    model = training_result["model"]
    feature_columns = training_result["feature_columns"]

    history = data["quantity_used"].astype(float).tolist()

    last_date = pd.Timestamp(data["date"].max())

    predictions = []

    for step in range(1, horizon + 1):
        forecast_date = last_date + pd.Timedelta(days=step)

        history_series = pd.Series(history, dtype=float)

        feature_row = {
            "day_of_week": forecast_date.dayofweek,
            "day_of_month": forecast_date.day,
            "month": forecast_date.month,
            "week_of_year": forecast_date.isocalendar().week,
            "quantity_available": data["quantity_available"].iloc[-1],
            "reference_capacity": data["reference_capacity"].iloc[-1],
            "quantity_used_lag_1": history_series.iloc[-1],
            "quantity_used_lag_7": history_series.iloc[-7],
            "quantity_used_lag_14": history_series.iloc[-14],
            "quantity_used_lag_30": history_series.iloc[-30],
            "quantity_used_rolling_mean_7": history_series.iloc[-7:].mean(),
            "quantity_used_rolling_mean_14": history_series.iloc[-14:].mean(),
            "quantity_used_rolling_mean_30": history_series.iloc[-30:].mean(),
        }

        feature_row = pd.DataFrame([feature_row])
        feature_row = feature_row[feature_columns]

        prediction = float(model.predict(feature_row)[0])
        prediction = max(prediction, 0.0)

        predictions.append(
            {
                "date": forecast_date.date().isoformat(),
                "predicted_value": prediction,
            }
        )

        history.append(prediction)

    return {
        "forecast_type": "resource",
        "resource_name": resource_name,
        "horizon": horizon,
        "model_name": training_result["model_name"],
        "model_version": training_result["model_version"],
        "predictions": predictions,
        "validation_metrics": training_result["validation_metrics"],
        "test_metrics": training_result["test_metrics"],
        "training_period_start": training_result[
            "training_period_start"
        ].isoformat(),
        "training_period_end": training_result[
            "training_period_end"
        ].isoformat(),
    }

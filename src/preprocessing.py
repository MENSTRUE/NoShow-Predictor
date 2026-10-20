from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "age",
    "under_12_years_old",
    "over_60_years_old",
    "patient_needs_companion",
    "appointment_hour",
    "appointment_minute",
    "appointment_dow",
    "appointment_day",
    "appointment_month_num",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "service_tenure_days",
    "specialty_missing",
    "age_missing",
    "tenure_missing",
]

CATEGORICAL_FEATURES = [
    "specialty",
    "appointment_shift",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


REQUIRED_INPUT_COLUMNS = {
    "appointment_date",
    "appointment_time",
    "specialty",
    "appointment_shift",
    "age",
    "under_12_years_old",
    "over_60_years_old",
    "patient_needs_companion",
    "entry_service_date",
}


def validate_input(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate the minimum raw fields required for inference.
    """

    if df is None or df.empty:
        return False, "Input data is empty."

    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)

    if missing:
        return (
            False,
            "Missing required columns: "
            + ", ".join(sorted(missing)),
        )

    appointment_date = pd.to_datetime(
        df["appointment_date"],
        dayfirst=True,
        errors="coerce",
    )

    if appointment_date.isna().any():
        invalid_count = int(
            appointment_date.isna().sum()
        )

        return (
            False,
            f"{invalid_count} row(s) contain invalid appointment_date values.",
        )

    return True, "Input data is valid."


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the exact feature engineering used in the training notebook.

    Important:
    - no_show / no_show_reason are NOT required;
    - ICD is NOT used;
    - gender, city and disability are NOT model features;
    - actual same-day weather is NOT used.
    """

    valid, message = validate_input(df)

    if not valid:
        raise ValueError(message)

    work = df.copy()

    # ---------------------------------------------------------
    # Appointment date
    # ---------------------------------------------------------

    work["appointment_date_dt"] = pd.to_datetime(
        work["appointment_date"],
        dayfirst=True,
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Appointment time
    # ---------------------------------------------------------

    time_parts = (
        work["appointment_time"]
        .astype(str)
        .str.extract(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
        )
    )

    work["appointment_hour"] = pd.to_numeric(
        time_parts["hour"],
        errors="coerce",
    )

    work["appointment_minute"] = pd.to_numeric(
        time_parts["minute"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Calendar features
    # ---------------------------------------------------------

    work["appointment_dow"] = (
        work["appointment_date_dt"].dt.dayofweek
    )

    work["appointment_day"] = (
        work["appointment_date_dt"].dt.day
    )

    work["appointment_month_num"] = (
        work["appointment_date_dt"].dt.month
    )

    work["is_weekend"] = (
        work["appointment_dow"]
        .isin([5, 6])
        .astype(int)
    )

    work["dow_sin"] = np.sin(
        2
        * np.pi
        * work["appointment_dow"]
        / 7
    )

    work["dow_cos"] = np.cos(
        2
        * np.pi
        * work["appointment_dow"]
        / 7
    )

    work["month_sin"] = np.sin(
        2
        * np.pi
        * work["appointment_month_num"]
        / 12
    )

    work["month_cos"] = np.cos(
        2
        * np.pi
        * work["appointment_month_num"]
        / 12
    )

    # ---------------------------------------------------------
    # Service tenure
    # ---------------------------------------------------------

    work["entry_service_dt"] = pd.to_datetime(
        work["entry_service_date"],
        dayfirst=True,
        errors="coerce",
    )

    work["service_tenure_days"] = (
        work["appointment_date_dt"]
        - work["entry_service_dt"]
    ).dt.days

    # Impossible negative tenure becomes missing.
    work.loc[
        work["service_tenure_days"] < 0,
        "service_tenure_days",
    ] = np.nan

    # ---------------------------------------------------------
    # Missingness indicators
    # ---------------------------------------------------------

    work["specialty_missing"] = (
        work["specialty"]
        .isna()
        .astype(int)
    )

    work["age_missing"] = (
        work["age"]
        .isna()
        .astype(int)
    )

    work["tenure_missing"] = (
        work["service_tenure_days"]
        .isna()
        .astype(int)
    )

    # ---------------------------------------------------------
    # Numeric conversion
    # ---------------------------------------------------------

    for column in NUMERIC_FEATURES:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # LightGBM categorical features
    # ---------------------------------------------------------

    for column in CATEGORICAL_FEATURES:
        work[column] = (
            work[column]
            .fillna("__MISSING__")
            .astype("category")
        )

    return work[MODEL_FEATURES].copy()
from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================
# MODEL FEATURE CONTRACT
# =========================================================

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


MODEL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


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


# =========================================================
# VALIDATION
# =========================================================

def validate_input(
    df: pd.DataFrame,
) -> tuple[bool, str]:

    if df is None:
        return (
            False,
            "Input data is missing.",
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        return (
            False,
            "Input must be a pandas DataFrame.",
        )

    if df.empty:
        return (
            False,
            "Input data is empty.",
        )

    missing_columns = (
        REQUIRED_INPUT_COLUMNS
        - set(df.columns)
    )

    if missing_columns:
        return (
            False,
            "Missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            ),
        )

    appointment_date = pd.to_datetime(
        df["appointment_date"],
        dayfirst=True,
        errors="coerce",
    )

    invalid_dates = int(
        appointment_date.isna().sum()
    )

    if invalid_dates > 0:
        return (
            False,
            f"{invalid_dates} row(s) contain "
            "invalid appointment_date values.",
        )

    return (
        True,
        "Input data is valid.",
    )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the original training feature engineering.

    IMPORTANT:
    Categorical columns remain as strings here.

    Encoding into LightGBM categorical integer codes is
    performed by the pure-Python model evaluator using
    metadata stored inside noshow_champion_model.json.
    """

    valid, message = validate_input(
        df
    )

    if not valid:
        raise ValueError(
            message
        )

    work = df.copy()


    # =====================================================
    # APPOINTMENT DATE
    # =====================================================

    work[
        "appointment_date_dt"
    ] = pd.to_datetime(
        work[
            "appointment_date"
        ],
        dayfirst=True,
        errors="coerce",
    )


    # =====================================================
    # APPOINTMENT TIME
    # =====================================================

    time_parts = (
        work[
            "appointment_time"
        ]
        .astype(str)
        .str.extract(
            r"(?P<hour>\d{1,2}):"
            r"(?P<minute>\d{2})"
        )
    )

    work[
        "appointment_hour"
    ] = pd.to_numeric(
        time_parts["hour"],
        errors="coerce",
    )

    work[
        "appointment_minute"
    ] = pd.to_numeric(
        time_parts["minute"],
        errors="coerce",
    )


    # =====================================================
    # CALENDAR FEATURES
    # =====================================================

    work[
        "appointment_dow"
    ] = (
        work[
            "appointment_date_dt"
        ]
        .dt.dayofweek
    )

    work[
        "appointment_day"
    ] = (
        work[
            "appointment_date_dt"
        ]
        .dt.day
    )

    work[
        "appointment_month_num"
    ] = (
        work[
            "appointment_date_dt"
        ]
        .dt.month
    )

    work[
        "is_weekend"
    ] = (
        work[
            "appointment_dow"
        ]
        .isin(
            [
                5,
                6,
            ]
        )
        .astype(int)
    )


    # =====================================================
    # CYCLICAL FEATURES
    # =====================================================

    work[
        "dow_sin"
    ] = np.sin(
        2
        * np.pi
        * work[
            "appointment_dow"
        ]
        / 7
    )

    work[
        "dow_cos"
    ] = np.cos(
        2
        * np.pi
        * work[
            "appointment_dow"
        ]
        / 7
    )

    work[
        "month_sin"
    ] = np.sin(
        2
        * np.pi
        * work[
            "appointment_month_num"
        ]
        / 12
    )

    work[
        "month_cos"
    ] = np.cos(
        2
        * np.pi
        * work[
            "appointment_month_num"
        ]
        / 12
    )


    # =====================================================
    # SERVICE TENURE
    # =====================================================

    work[
        "entry_service_dt"
    ] = pd.to_datetime(
        work[
            "entry_service_date"
        ],
        dayfirst=True,
        errors="coerce",
    )

    work[
        "service_tenure_days"
    ] = (
        work[
            "appointment_date_dt"
        ]
        - work[
            "entry_service_dt"
        ]
    ).dt.days

    work.loc[
        work[
            "service_tenure_days"
        ]
        < 0,
        "service_tenure_days",
    ] = np.nan


    # =====================================================
    # MISSINGNESS INDICATORS
    # =====================================================

    work[
        "specialty_missing"
    ] = (
        work[
            "specialty"
        ]
        .isna()
        .astype(int)
    )

    work[
        "age_missing"
    ] = (
        work[
            "age"
        ]
        .isna()
        .astype(int)
    )

    work[
        "tenure_missing"
    ] = (
        work[
            "service_tenure_days"
        ]
        .isna()
        .astype(int)
    )


    # =====================================================
    # NUMERIC NORMALIZATION
    # =====================================================

    for column in NUMERIC_FEATURES:
        work[
            column
        ] = pd.to_numeric(
            work[
                column
            ],
            errors="coerce",
        )


    # =====================================================
    # CATEGORICAL NORMALIZATION
    # =====================================================

    work[
        "specialty"
    ] = (
        work[
            "specialty"
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
    )

    work[
        "appointment_shift"
    ] = (
        work[
            "appointment_shift"
        ]
        .fillna(
            "__MISSING__"
        )
        .astype(str)
    )


    features = work[
        MODEL_FEATURES
    ].copy()


    if list(
        features.columns
    ) != MODEL_FEATURES:
        raise RuntimeError(
            "Inference feature order "
            "does not match model contract."
        )


    return features
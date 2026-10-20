from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference import (
    DEFAULT_CAPACITY_FRACTION,
    VALIDATION_THRESHOLD,
    NoShowPredictor,
)
from src.preprocessing import (
    MODEL_FEATURES,
    build_features,
    validate_input,
)


def make_sample_input(n: int = 20) -> pd.DataFrame:
    dates = pd.date_range(
        "2022-03-01",
        periods=n,
        freq="D",
    )

    return pd.DataFrame(
        {
            "appointment_date": dates.strftime("%d/%m/%Y"),
            "appointment_time": ["09:30"] * n,
            "specialty": ["General"] * n,
            "appointment_shift": ["Morning"] * n,
            "age": [30] * n,
            "under_12_years_old": [0] * n,
            "over_60_years_old": [0] * n,
            "patient_needs_companion": [0] * n,
            "entry_service_date": ["01/01/2020"] * n,
        }
    )


def test_validate_input_success():
    df = make_sample_input()

    valid, message = validate_input(df)

    assert valid is True
    assert message == "Input data is valid."


def test_validate_input_missing_column():
    df = make_sample_input().drop(
        columns=["appointment_date"]
    )

    valid, message = validate_input(df)

    assert valid is False
    assert "appointment_date" in message


def test_build_features_has_exact_model_columns():
    df = make_sample_input()

    features = build_features(df)

    assert list(features.columns) == MODEL_FEATURES
    assert len(features) == len(df)


def test_calendar_feature_engineering():
    df = make_sample_input(1)

    features = build_features(df)

    assert features.loc[0, "appointment_hour"] == 9
    assert features.loc[0, "appointment_minute"] == 30

    assert not np.isnan(
        features.loc[0, "dow_sin"]
    )

    assert not np.isnan(
        features.loc[0, "dow_cos"]
    )


def test_service_tenure_is_created():
    df = make_sample_input(1)

    features = build_features(df)

    assert features.loc[
        0,
        "service_tenure_days",
    ] > 0

    assert (
        features.loc[
            0,
            "tenure_missing",
        ]
        == 0
    )


def test_negative_tenure_becomes_missing():
    df = make_sample_input(1)

    df.loc[
        0,
        "entry_service_date",
    ] = "01/01/2030"

    features = build_features(df)

    assert np.isnan(
        features.loc[
            0,
            "service_tenure_days",
        ]
    )

    assert (
        features.loc[
            0,
            "tenure_missing",
        ]
        == 1
    )


def test_capacity_priority_top_10_percent():
    df = pd.DataFrame(
        {
            "no_show_risk_score": np.linspace(
                0.01,
                1.0,
                20,
            )
        }
    )

    result = (
        NoShowPredictor
        .add_capacity_priority(
            df,
            capacity_fraction=0.10,
        )
    )

    assert len(result) == 20

    # ceil(20 * 10%) = 2
    assert (
        result[
            "priority_top_capacity"
        ].sum()
        == 2
    )

    assert (
        result.iloc[0][
            "no_show_risk_score"
        ]
        >= result.iloc[-1][
            "no_show_risk_score"
        ]
    )

    assert result.iloc[0][
        "priority_rank"
    ] == 1


def test_capacity_fraction_validation():
    df = pd.DataFrame(
        {
            "no_show_risk_score": [
                0.1,
                0.2,
            ]
        }
    )

    with pytest.raises(ValueError):
        NoShowPredictor.add_capacity_priority(
            df,
            capacity_fraction=0,
        )

    with pytest.raises(ValueError):
        NoShowPredictor.add_capacity_priority(
            df,
            capacity_fraction=1.5,
        )


def test_validation_threshold_flag():
    df = pd.DataFrame(
        {
            "no_show_risk_score": [
                0.30,
                0.44,
                0.80,
            ]
        }
    )

    result = (
        NoShowPredictor
        .add_threshold_flag(
            df,
            threshold=VALIDATION_THRESHOLD,
        )
    )

    assert (
        result[
            "threshold_flag"
        ].tolist()
        == [
            False,
            True,
            True,
        ]
    )


def test_default_capacity_constant():
    assert (
        DEFAULT_CAPACITY_FRACTION
        == 0.10
    )


def test_model_artifact_can_load():
    predictor = NoShowPredictor(
        model_path=(
            "models/"
            "noshow_champion.joblib"
        )
    )

    assert predictor.model is not None
    assert hasattr(
        predictor.model,
        "predict_proba",
    )
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference import (
    DEFAULT_CAPACITY_FRACTION,
    VALIDATION_THRESHOLD,
    NoShowPredictor,
    PurePythonLightGBM,
)

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    build_features,
    validate_input,
)


# =========================================================
# TEST DATA
# =========================================================

def make_sample_input(
    n: int = 20,
) -> pd.DataFrame:

    dates = pd.date_range(
        "2022-01-01",
        periods=n,
        freq="D",
    )


    specialties = [
        "physiotherapy",
        "speech therapy",
        "psychotherapy",
        "occupational therapy",
    ]


    shifts = [
        "morning",
        "afternoon",
    ]


    rows = []


    for index in range(
        n
    ):

        rows.append(
            {
                "appointment_date": (
                    dates[index]
                    .strftime(
                        "%d/%m/%Y"
                    )
                ),

                "appointment_time": (
                    "09:30"
                    if index % 2 == 0
                    else "14:20"
                ),

                "specialty": (
                    specialties[
                        index
                        % len(
                            specialties
                        )
                    ]
                ),

                "appointment_shift": (
                    shifts[
                        index
                        % len(
                            shifts
                        )
                    ]
                ),

                "age": (
                    20
                    + index
                ),

                "under_12_years_old": 0,

                "over_60_years_old": (
                    1
                    if (
                        20
                        + index
                    )
                    > 60
                    else 0
                ),

                "patient_needs_companion": (
                    index
                    % 2
                ),

                "entry_service_date": (
                    "01/01/2020"
                ),
            }
        )


    return pd.DataFrame(
        rows
    )


# =========================================================
# VALIDATION
# =========================================================

def test_validate_input_success():

    df = make_sample_input()

    valid, message = (
        validate_input(
            df
        )
    )

    assert valid is True

    assert (
        message
        == "Input data is valid."
    )


def test_validate_input_missing_column():

    df = (
        make_sample_input()
        .drop(
            columns=[
                "appointment_date"
            ]
        )
    )


    valid, message = (
        validate_input(
            df
        )
    )


    assert valid is False

    assert (
        "appointment_date"
        in message
    )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def test_feature_contract():

    df = make_sample_input(
        5
    )


    features = (
        build_features(
            df
        )
    )


    assert list(
        features.columns
    ) == MODEL_FEATURES


    assert len(
        features
    ) == 5


def test_calendar_features():

    df = make_sample_input(
        1
    )


    features = (
        build_features(
            df
        )
    )


    assert (
        features.loc[
            0,
            "appointment_hour",
        ]
        == 9
    )


    assert (
        features.loc[
            0,
            "appointment_minute",
        ]
        == 30
    )


    assert np.isfinite(
        features.loc[
            0,
            "dow_sin",
        ]
    )


    assert np.isfinite(
        features.loc[
            0,
            "month_cos",
        ]
    )


def test_service_tenure():

    df = make_sample_input(
        1
    )


    features = (
        build_features(
            df
        )
    )


    assert (
        features.loc[
            0,
            "service_tenure_days",
        ]
        > 0
    )


    assert (
        features.loc[
            0,
            "tenure_missing",
        ]
        == 0
    )


def test_negative_tenure_is_missing():

    df = make_sample_input(
        1
    )


    df.loc[
        0,
        "entry_service_date",
    ] = "01/01/2030"


    features = (
        build_features(
            df
        )
    )


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


def test_categorical_features_remain_strings():

    df = make_sample_input(
        2
    )


    features = (
        build_features(
            df
        )
    )


    for column in (
        CATEGORICAL_FEATURES
    ):

        assert (
            features[
                column
            ]
            .map(
                lambda value:
                isinstance(
                    value,
                    str,
                )
            )
            .all()
        )


# =========================================================
# JSON MODEL
# =========================================================

def test_json_model_loads():

    model = (
        PurePythonLightGBM(
            "models/"
            "noshow_champion_model.json"
        )
    )


    assert (
        model.num_trees
        == 500
    )


    assert (
        model.feature_names
        == MODEL_FEATURES
    )


def test_original_categorical_metadata():

    model = (
        PurePythonLightGBM(
            "models/"
            "noshow_champion_model.json"
        )
    )


    assert (
        model.category_lists[
            "appointment_shift"
        ]
        == [
            "afternoon",
            "morning",
        ]
    )


    assert (
        "physiotherapy"
        in model.category_lists[
            "specialty"
        ]
    )


    assert (
        "__MISSING__"
        in model.category_lists[
            "specialty"
        ]
    )


def test_unknown_category_maps_to_negative_one():

    model = (
        PurePythonLightGBM(
            "models/"
            "noshow_champion_model.json"
        )
    )


    code = (
        model._encode_category(
            "specialty",
            "TOTALLY_NEW_SPECIALTY",
        )
    )


    assert (
        code
        == -1
    )


# =========================================================
# CAPACITY POLICY
# =========================================================

def test_capacity_priority_top_10_percent():

    df = pd.DataFrame(
        {
            "no_show_risk_score":
            np.linspace(
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


    assert len(
        result
    ) == 20


    assert int(
        result[
            "priority_top_capacity"
        ].sum()
    ) == 2


    assert (
        result.iloc[0][
            "no_show_risk_score"
        ]
        >= result.iloc[-1][
            "no_show_risk_score"
        ]
    )


def test_invalid_capacity():

    df = pd.DataFrame(
        {
            "no_show_risk_score": [
                0.1,
                0.2,
            ]
        }
    )


    with pytest.raises(
        ValueError
    ):
        (
            NoShowPredictor
            .add_capacity_priority(
                df,
                capacity_fraction=0,
            )
        )


# =========================================================
# HISTORICAL THRESHOLD
# =========================================================

def test_threshold_flag():

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
            threshold=(
                VALIDATION_THRESHOLD
            ),
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


# =========================================================
# CONSTANTS
# =========================================================

def test_constants():

    assert (
        DEFAULT_CAPACITY_FRACTION
        == 0.10
    )


    assert (
        VALIDATION_THRESHOLD
        == 0.44
    )


# =========================================================
# END-TO-END PURE PYTHON INFERENCE
# =========================================================

def test_pure_python_model_predicts():

    df = make_sample_input(
        5
    )


    predictor = (
        NoShowPredictor()
    )


    result = (
        predictor.predict(
            df,
            capacity_fraction=0.20,
        )
    )


    assert len(
        result
    ) == 5


    assert (
        "no_show_risk_score"
        in result.columns
    )


    assert np.isfinite(
        result[
            "no_show_risk_score"
        ]
    ).all()


    assert (
        result[
            "no_show_risk_score"
        ]
        .between(
            0,
            1,
        )
        .all()
    )


    assert (
        int(
            result[
                "priority_top_capacity"
            ].sum()
        )
        == 1
    )


    assert (
        "threshold_flag"
        in result.columns
    )
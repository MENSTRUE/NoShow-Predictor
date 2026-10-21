from __future__ import annotations

import json
import math
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    build_features,
    validate_input,
)


# =========================================================
# PATH / CONFIG
# =========================================================

ROOT = Path(
    __file__
).resolve().parents[1]


DEFAULT_MODEL_PATH = (
    ROOT
    / "models"
    / "noshow_champion_model.json"
)


VALIDATION_THRESHOLD = 0.44

DEFAULT_CAPACITY_FRACTION = 0.10


# =========================================================
# SUMMARY
# =========================================================

@dataclass
class InferenceSummary:
    rows: int
    priority_count: int

    mean_risk_score: float
    min_risk_score: float
    max_risk_score: float

    capacity_fraction: float

    threshold_flag_count: int


# =========================================================
# PURE PYTHON LIGHTGBM MODEL
# =========================================================

class PurePythonLightGBM:
    """
    Minimal pure-Python evaluator for the exported
    LightGBM JSON tree dump.

    This implementation is intentionally inference-only.

    It avoids:
        lightgbm.Booster.predict()
        sklearn predict_proba()
        joblib model execution

    This is useful here because native LightGBM prediction
    was producing a Windows access-violation in the local
    deployment environment.
    """

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "JSON model not found: "
                f"{self.model_path}"
            )

        with self.model_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.model: dict[str, Any] = (
                json.load(
                    file
                )
            )


        # =================================================
        # MODEL METADATA
        # =================================================

        self.feature_names = list(
            self.model.get(
                "feature_names",
                [],
            )
        )

        if not self.feature_names:
            raise ValueError(
                "JSON model does not contain "
                "'feature_names'."
            )

        if (
            self.feature_names
            != MODEL_FEATURES
        ):
            raise ValueError(
                "JSON model feature contract mismatch.\n"
                f"Model: {self.feature_names}\n"
                f"Code : {MODEL_FEATURES}"
            )


        self.tree_info = self.model.get(
            "tree_info",
            []
        )

        if not self.tree_info:
            raise ValueError(
                "JSON model contains no trees."
            )


        # =================================================
        # CATEGORICAL METADATA
        # =================================================

        pandas_categorical = (
            self.model.get(
                "pandas_categorical"
            )
            or []
        )

        if len(
            pandas_categorical
        ) != len(
            CATEGORICAL_FEATURES
        ):
            raise ValueError(
                "Unexpected categorical metadata. "
                f"Expected {len(CATEGORICAL_FEATURES)} "
                f"categorical features but found "
                f"{len(pandas_categorical)}."
            )


        self.category_lists: dict[
            str,
            list[str],
        ] = {}

        self.category_maps: dict[
            str,
            dict[str, int],
        ] = {}


        for (
            feature_name,
            categories,
        ) in zip(
            CATEGORICAL_FEATURES,
            pandas_categorical,
        ):

            normalized_categories = [
                str(value)
                for value
                in categories
            ]

            self.category_lists[
                feature_name
            ] = normalized_categories

            self.category_maps[
                feature_name
            ] = {
                value: index
                for index, value
                in enumerate(
                    normalized_categories
                )
            }


        # =================================================
        # OBJECTIVE
        # =================================================

        self.objective = str(
            self.model.get(
                "objective",
                "binary sigmoid:1",
            )
        )

        if not self.objective.startswith(
            "binary"
        ):
            raise ValueError(
                "This evaluator currently expects "
                "a binary LightGBM objective. "
                f"Received: {self.objective}"
            )


        self.sigmoid = (
            self._parse_sigmoid(
                self.objective
            )
        )


    # =====================================================
    # METADATA
    # =====================================================

    @property
    def num_trees(
        self,
    ) -> int:

        return len(
            self.tree_info
        )


    # =====================================================
    # OBJECTIVE
    # =====================================================

    @staticmethod
    def _parse_sigmoid(
        objective: str,
    ) -> float:

        match = re.search(
            r"sigmoid[:=]([0-9eE+\-.]+)",
            objective,
        )

        if match is None:
            return 1.0

        try:
            value = float(
                match.group(1)
            )

            if value <= 0:
                return 1.0

            return value

        except ValueError:
            return 1.0


    def _sigmoid(
        self,
        raw_score: float,
    ) -> float:

        z = (
            self.sigmoid
            * raw_score
        )

        # Numerically stable logistic sigmoid.
        if z >= 0:

            exp_negative = math.exp(
                -z
            )

            return (
                1.0
                / (
                    1.0
                    + exp_negative
                )
            )

        exp_positive = math.exp(
            z
        )

        return (
            exp_positive
            / (
                1.0
                + exp_positive
            )
        )


    # =====================================================
    # CATEGORICAL ENCODING
    # =====================================================

    def _encode_category(
        self,
        feature_name: str,
        value: Any,
    ) -> int:

        mapping = self.category_maps[
            feature_name
        ]

        if value is None:
            normalized = "__MISSING__"

        elif (
            isinstance(
                value,
                float,
            )
            and math.isnan(
                value
            )
        ):
            normalized = "__MISSING__"

        else:
            normalized = str(
                value
            )

        # This mirrors pandas categorical encoding:
        #
        # known category   -> 0, 1, 2, ...
        # unknown category -> -1
        return int(
            mapping.get(
                normalized,
                -1,
            )
        )


    # =====================================================
    # MISSING VALUE HANDLING
    # =====================================================

    @staticmethod
    def _is_nan(
        value: Any,
    ) -> bool:

        if value is None:
            return True

        try:
            return bool(
                pd.isna(
                    value
                )
            )

        except Exception:
            return False


    @staticmethod
    def _numeric_is_missing(
        value: Any,
        missing_type: str,
    ) -> bool:

        if PurePythonLightGBM._is_nan(
            value
        ):
            return True

        if missing_type == "Zero":

            try:
                return (
                    float(value)
                    == 0.0
                )

            except (
                TypeError,
                ValueError,
            ):
                return False

        return False


    # =====================================================
    # TREE SPLIT
    # =====================================================

    @staticmethod
    def _categorical_threshold_set(
        threshold: Any,
    ) -> set[int]:

        if threshold is None:
            return set()

        if isinstance(
            threshold,
            (
                int,
                np.integer,
            ),
        ):
            return {
                int(
                    threshold
                )
            }

        text = str(
            threshold
        ).strip()

        if not text:
            return set()

        output: set[int] = set()

        for item in text.split(
            "||"
        ):

            item = item.strip()

            if not item:
                continue

            try:
                output.add(
                    int(
                        item
                    )
                )

            except ValueError:
                continue

        return output


    def _go_left(
        self,
        node: dict[str, Any],
        row: pd.Series,
    ) -> bool:

        split_feature_index = int(
            node[
                "split_feature"
            ]
        )

        feature_name = (
            self.feature_names[
                split_feature_index
            ]
        )

        value = row[
            feature_name
        ]

        decision_type = str(
            node.get(
                "decision_type",
                "<=",
            )
        )

        default_left = bool(
            node.get(
                "default_left",
                False,
            )
        )

        missing_type = str(
            node.get(
                "missing_type",
                "None",
            )
        )


        # =================================================
        # CATEGORICAL SPLIT
        # =================================================

        if decision_type in {
            "==",
            "in",
        }:

            if feature_name not in (
                CATEGORICAL_FEATURES
            ):
                raise RuntimeError(
                    "Categorical split encountered "
                    "for unexpected feature: "
                    f"{feature_name}"
                )


            category_code = (
                self._encode_category(
                    feature_name,
                    value,
                )
            )


            # LightGBM treats negative category codes
            # as missing / invalid categories.
            if category_code < 0:
                return default_left


            accepted_codes = (
                self._categorical_threshold_set(
                    node.get(
                        "threshold"
                    )
                )
            )


            return (
                category_code
                in accepted_codes
            )


        # =================================================
        # NUMERIC SPLIT
        # =================================================

        if self._numeric_is_missing(
            value,
            missing_type,
        ):
            return default_left


        try:
            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return default_left


        threshold = float(
            node[
                "threshold"
            ]
        )


        if decision_type == "<=":

            return (
                numeric_value
                <= threshold
            )


        if decision_type == "<":

            return (
                numeric_value
                < threshold
            )


        if decision_type == ">":

            return (
                numeric_value
                > threshold
            )


        if decision_type == ">=":

            return (
                numeric_value
                >= threshold
            )


        raise RuntimeError(
            "Unsupported LightGBM decision type: "
            f"{decision_type}"
        )


    # =====================================================
    # TREE EVALUATION
    # =====================================================

    def _predict_tree(
        self,
        tree_structure: dict[str, Any],
        row: pd.Series,
    ) -> float:

        node = tree_structure


        while True:

            if "leaf_value" in node:

                return float(
                    node[
                        "leaf_value"
                    ]
                )


            go_left = self._go_left(
                node,
                row,
            )


            if go_left:

                node = node[
                    "left_child"
                ]

            else:

                node = node[
                    "right_child"
                ]


    # =====================================================
    # RAW SCORE
    # =====================================================

    def predict_raw_row(
        self,
        row: pd.Series,
    ) -> float:

        raw_score = 0.0


        for tree in self.tree_info:

            raw_score += (
                self._predict_tree(
                    tree[
                        "tree_structure"
                    ],
                    row,
                )
            )


        return float(
            raw_score
        )


    # =====================================================
    # PROBABILITY
    # =====================================================

    def predict_row(
        self,
        row: pd.Series,
    ) -> float:

        raw_score = (
            self.predict_raw_row(
                row
            )
        )

        return float(
            self._sigmoid(
                raw_score
            )
        )


    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:

        if list(
            features.columns
        ) != self.feature_names:

            raise ValueError(
                "Prediction feature order mismatch."
            )


        predictions = np.empty(
            len(features),
            dtype=np.float64,
        )


        for position, (
            _,
            row,
        ) in enumerate(
            features.iterrows()
        ):

            predictions[
                position
            ] = self.predict_row(
                row
            )


        return predictions


# =========================================================
# APPLICATION PREDICTOR
# =========================================================

class NoShowPredictor:

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:

        self.model_path = Path(
            model_path
        )


        self.model = (
            PurePythonLightGBM(
                model_path=(
                    self.model_path
                )
            )
        )


    # =====================================================
    # SCORE
    # =====================================================

    def predict_scores(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        valid, message = validate_input(
            df
        )

        if not valid:
            raise ValueError(
                message
            )


        features = build_features(
            df
        )


        predictions = (
            self.model.predict(
                features
            )
        )


        if len(
            predictions
        ) != len(
            df
        ):
            raise RuntimeError(
                "Model returned an unexpected "
                "number of predictions."
            )


        if not np.isfinite(
            predictions
        ).all():
            raise RuntimeError(
                "Model produced non-finite "
                "risk scores."
            )


        result = df.copy()


        result[
            "no_show_risk_score"
        ] = predictions


        return result


    # =====================================================
    # CAPACITY RANKING
    # =====================================================

    @staticmethod
    def add_capacity_priority(
        scored_df: pd.DataFrame,
        capacity_fraction: float = DEFAULT_CAPACITY_FRACTION,
    ) -> pd.DataFrame:

        if not (
            0
            < capacity_fraction
            <= 1
        ):
            raise ValueError(
                "capacity_fraction must be "
                "> 0 and <= 1."
            )


        if (
            "no_show_risk_score"
            not in scored_df.columns
        ):
            raise ValueError(
                "Column 'no_show_risk_score' "
                "is missing."
            )


        result = (
            scored_df
            .copy()
            .sort_values(
                "no_show_risk_score",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


        n_rows = len(
            result
        )


        if n_rows == 0:

            result[
                "priority_rank"
            ] = pd.Series(
                dtype=int
            )

            result[
                "priority_top_capacity"
            ] = pd.Series(
                dtype=bool
            )

            return result


        top_k = max(
            1,
            int(
                np.ceil(
                    n_rows
                    * capacity_fraction
                )
            ),
        )


        result[
            "priority_rank"
        ] = np.arange(
            1,
            n_rows + 1,
        )


        result[
            "priority_top_capacity"
        ] = False


        result.loc[
            result.index < top_k,
            "priority_top_capacity",
        ] = True


        return result


    # =====================================================
    # HISTORICAL THRESHOLD
    # =====================================================

    @staticmethod
    def add_threshold_flag(
        scored_df: pd.DataFrame,
        threshold: float = VALIDATION_THRESHOLD,
    ) -> pd.DataFrame:

        if not (
            0
            <= threshold
            <= 1
        ):
            raise ValueError(
                "threshold must be "
                "between 0 and 1."
            )


        result = scored_df.copy()


        result[
            "validation_threshold"
        ] = float(
            threshold
        )


        result[
            "threshold_flag"
        ] = (
            result[
                "no_show_risk_score"
            ]
            >= threshold
        )


        return result


    # =====================================================
    # COMPLETE INFERENCE
    # =====================================================

    def predict(
        self,
        df: pd.DataFrame,
        capacity_fraction: float = DEFAULT_CAPACITY_FRACTION,
        include_threshold_flag: bool = True,
    ) -> pd.DataFrame:

        result = (
            self.predict_scores(
                df
            )
        )


        result = (
            self.add_capacity_priority(
                result,
                capacity_fraction=(
                    capacity_fraction
                ),
            )
        )


        if include_threshold_flag:

            result = (
                self.add_threshold_flag(
                    result,
                    threshold=(
                        VALIDATION_THRESHOLD
                    ),
                )
            )


        return result


    # =====================================================
    # SUMMARY
    # =====================================================

    @staticmethod
    def summarize(
        result: pd.DataFrame,
        capacity_fraction: float = DEFAULT_CAPACITY_FRACTION,
    ) -> InferenceSummary:

        if result.empty:

            return InferenceSummary(
                rows=0,
                priority_count=0,
                mean_risk_score=0.0,
                min_risk_score=0.0,
                max_risk_score=0.0,
                capacity_fraction=float(
                    capacity_fraction
                ),
                threshold_flag_count=0,
            )


        scores = pd.to_numeric(
            result[
                "no_show_risk_score"
            ],
            errors="coerce",
        )


        priority_count = int(
            result[
                "priority_top_capacity"
            ].sum()
        )


        threshold_flag_count = 0


        if (
            "threshold_flag"
            in result.columns
        ):

            threshold_flag_count = int(
                result[
                    "threshold_flag"
                ].sum()
            )


        return InferenceSummary(
            rows=int(
                len(result)
            ),

            priority_count=(
                priority_count
            ),

            mean_risk_score=float(
                scores.mean()
            ),

            min_risk_score=float(
                scores.min()
            ),

            max_risk_score=float(
                scores.max()
            ),

            capacity_fraction=float(
                capacity_fraction
            ),

            threshold_flag_count=(
                threshold_flag_count
            ),
        )
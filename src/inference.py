from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import (
    MODEL_FEATURES,
    build_features,
    validate_input,
)


DEFAULT_MODEL_PATH = Path(
    "models/noshow_champion.joblib"
)

# Historical experiment threshold.
# IMPORTANT:
# This threshold was selected on validation predictions before
# the final champion was refit on Train + Validation.
VALIDATION_THRESHOLD = 0.44

DEFAULT_CAPACITY_FRACTION = 0.10


@dataclass
class InferenceSummary:
    rows: int
    priority_count: int
    mean_risk_score: float
    min_risk_score: float
    max_risk_score: float
    capacity_fraction: float
    threshold_flag_count: int


class NoShowPredictor:
    """
    No-show risk prioritization inference.

    Intended use:
        Prioritize reminder outreach.

    NOT intended for:
        - clinical decisions
        - diagnosis
        - appointment cancellation
        - punitive action
        - guaranteed no-show prediction
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ):
        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

    def predict_scores(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce a continuous model risk score for each appointment.

        This score is useful primarily for ranking appointments.
        It must not be interpreted as guaranteed real-world probability
        outside the evaluated data distribution.
        """

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

        if list(features.columns) != MODEL_FEATURES:
            raise RuntimeError(
                "Feature order mismatch."
            )

        probabilities = (
            self.model
            .predict_proba(features)[:, 1]
        )

        result = df.copy()

        result["no_show_risk_score"] = (
            np.asarray(
                probabilities,
                dtype=float,
            )
        )

        return result

    @staticmethod
    def add_capacity_priority(
        scored_df: pd.DataFrame,
        capacity_fraction: float = DEFAULT_CAPACITY_FRACTION,
    ) -> pd.DataFrame:
        """
        Mark the highest-risk appointments according to reminder capacity.

        Example:
            capacity_fraction=0.10
            -> prioritize the top 10% highest-risk appointments.
        """

        if not 0 < capacity_fraction <= 1:
            raise ValueError(
                "capacity_fraction must be "
                "greater than 0 and <= 1."
            )

        if "no_show_risk_score" not in scored_df.columns:
            raise ValueError(
                "Column no_show_risk_score is missing."
            )

        result = scored_df.copy()

        result = result.sort_values(
            "no_show_risk_score",
            ascending=False,
        ).reset_index(drop=True)

        n_rows = len(result)

        if n_rows == 0:
            result["priority_rank"] = []
            result["priority_top_capacity"] = []
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

        result["priority_rank"] = (
            np.arange(
                1,
                n_rows + 1,
            )
        )

        result[
            "priority_top_capacity"
        ] = False

        result.loc[
            : top_k - 1,
            "priority_top_capacity",
        ] = True

        return result

    @staticmethod
    def add_threshold_flag(
        scored_df: pd.DataFrame,
        threshold: float = VALIDATION_THRESHOLD,
    ) -> pd.DataFrame:
        """
        Add the historical validation-selected threshold flag.

        WARNING:
        The final model was refit on Train + Validation after threshold
        selection, so this threshold is preserved mainly for experiment
        traceability and comparison.

        Prefer capacity-based ranking for operational reminder prioritization.
        """

        if not 0 <= threshold <= 1:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        if "no_show_risk_score" not in scored_df.columns:
            raise ValueError(
                "Column no_show_risk_score is missing."
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

    def predict(
        self,
        df: pd.DataFrame,
        capacity_fraction: float = DEFAULT_CAPACITY_FRACTION,
        include_threshold_flag: bool = True,
    ) -> pd.DataFrame:
        """
        Main inference workflow.

        Default operational view:
            rank all appointments by risk score and flag the top capacity.

        Historical threshold:
            optionally included for audit/experiment comparison.
        """

        result = self.predict_scores(
            df
        )

        result = self.add_capacity_priority(
            result,
            capacity_fraction=capacity_fraction,
        )

        if include_threshold_flag:
            result = self.add_threshold_flag(
                result,
                threshold=VALIDATION_THRESHOLD,
            )

        return result

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
                capacity_fraction=capacity_fraction,
                threshold_flag_count=0,
            )

        priority_count = int(
            result[
                "priority_top_capacity"
            ].sum()
        )

        threshold_flag_count = (
            int(
                result[
                    "threshold_flag"
                ].sum()
            )
            if "threshold_flag"
            in result.columns
            else 0
        )

        return InferenceSummary(
            rows=len(result),
            priority_count=priority_count,
            mean_risk_score=float(
                result[
                    "no_show_risk_score"
                ].mean()
            ),
            min_risk_score=float(
                result[
                    "no_show_risk_score"
                ].min()
            ),
            max_risk_score=float(
                result[
                    "no_show_risk_score"
                ].max()
            ),
            capacity_fraction=float(
                capacity_fraction
            ),
            threshold_flag_count=threshold_flag_count,
        )
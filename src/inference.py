from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.preprocessing import (
    MODEL_FEATURES,
    build_features,
    validate_input,
)


# =========================================================
# CONFIG
# =========================================================

DEFAULT_MODEL_PATH = Path(
    "models/noshow_champion.txt"
)

# Historical validation-selected threshold.
#
# IMPORTANT:
# This threshold was selected using validation predictions
# before the champion was refit on Train + Validation.
#
# Therefore the application uses capacity-based ranking
# as its main operational view.
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
# PREDICTOR
# =========================================================

class NoShowPredictor:
    """
    NoShow Predictor inference interface.

    Intended use
    ------------
    Rank upcoming appointments so scheduling/service-operation
    staff can prioritize reminder outreach.

    Not intended for
    ----------------
    - diagnosis;
    - clinical prioritization;
    - patient blame;
    - punitive decisions;
    - automatic appointment cancellation;
    - guaranteed no-show prediction.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Native LightGBM model was not found: "
                f"{self.model_path}"
            )

        # Load the native LightGBM text model.
        #
        # This avoids relying on a Python pickle/joblib object
        # created in a different training environment.
        self.model = lgb.Booster(
            model_file=str(
                self.model_path
            )
        )

    # =====================================================
    # RAW SCORES
    # =====================================================

    def predict_scores(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate continuous no-show risk scores.

        The score is primarily intended for ranking.

        It must not be interpreted as certainty that an
        appointment will become a no-show.
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

        # Protect feature order.
        if list(features.columns) != MODEL_FEATURES:
            raise RuntimeError(
                "Feature order mismatch between inference "
                "pipeline and trained-model contract."
            )

        # LightGBM's native model stores pandas categorical
        # metadata from training. Passing the categorical
        # dataframe allows LightGBM to align those categories.
        predictions = self.model.predict(
            features,
            num_iteration=(
                self.model.best_iteration
                if self.model.best_iteration
                and self.model.best_iteration > 0
                else None
            ),
        )

        predictions = np.asarray(
            predictions,
            dtype=float,
        ).reshape(-1)

        if len(predictions) != len(df):
            raise RuntimeError(
                "Unexpected prediction count. "
                f"Input rows={len(df)}, "
                f"predictions={len(predictions)}."
            )

        if not np.isfinite(
            predictions
        ).all():
            raise RuntimeError(
                "Model produced non-finite risk scores."
            )

        result = df.copy()

        result[
            "no_show_risk_score"
        ] = predictions

        return result

    # =====================================================
    # CAPACITY-BASED PRIORITY
    # =====================================================

    @staticmethod
    def add_capacity_priority(
        scored_df: pd.DataFrame,
        capacity_fraction: float = DEFAULT_CAPACITY_FRACTION,
    ) -> pd.DataFrame:
        """
        Rank appointments from highest to lowest score.

        Example:
            capacity_fraction = 0.10

        means:
            prioritize the top 10% highest-scoring appointments.
        """

        if not (
            0
            < capacity_fraction
            <= 1
        ):
            raise ValueError(
                "capacity_fraction must be > 0 and <= 1."
            )

        if (
            "no_show_risk_score"
            not in scored_df.columns
        ):
            raise ValueError(
                "Column 'no_show_risk_score' is missing."
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
        """
        Add the historical validation-selected threshold.

        This is preserved primarily for experiment traceability.

        It is NOT the main operational reminder policy.
        """

        if not (
            0
            <= threshold
            <= 1
        ):
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        if (
            "no_show_risk_score"
            not in scored_df.columns
        ):
            raise ValueError(
                "Column 'no_show_risk_score' is missing."
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
        """
        Complete inference workflow:

        raw appointment data
            ↓
        model risk score
            ↓
        descending ranking
            ↓
        capacity-based reminder priority
            ↓
        optional historical threshold audit
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

        if (
            "priority_top_capacity"
            not in result.columns
        ):
            raise ValueError(
                "priority_top_capacity is missing."
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
            priority_count=priority_count,
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
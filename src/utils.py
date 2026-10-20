from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def ensure_directory(
    path: str | Path,
) -> Path:
    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def save_prediction_result(
    df: pd.DataFrame,
    output_dir: str | Path = "tests/results",
) -> Path:
    output_dir = ensure_directory(
        output_dir
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        output_dir
        / f"noshow_risk_{timestamp}.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    return output_path


def dataframe_to_csv_bytes(
    df: pd.DataFrame,
) -> bytes:
    return (
        df.to_csv(
            index=False
        )
        .encode("utf-8")
    )
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT ROOT
# =========================================================

ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from src.inference import (
    DEFAULT_CAPACITY_FRACTION,
    VALIDATION_THRESHOLD,
    NoShowPredictor,
)

from src.preprocessing import (
    validate_input,
)

from src.utils import (
    dataframe_to_csv_bytes,
    save_prediction_result,
)


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = (
    ROOT
    / "models"
    / "noshow_champion_model.json"
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NoShow Predictor",
    page_icon="📅",
    layout="wide",
)


# =========================================================
# MODEL LOADER
# =========================================================

@st.cache_resource
def load_predictor() -> NoShowPredictor:
    return NoShowPredictor(
        model_path=MODEL_PATH
    )


try:
    predictor = load_predictor()

except Exception as exc:
    st.error(
        "Could not load NoShow Predictor."
    )

    st.code(
        str(exc)
    )

    st.info(
        "Expected model:\n\n"
        "models/noshow_champion_model.json"
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "NoShow Predictor"
    )

    st.markdown(
        """
        **Champion Model**  
        LGBM-Balanced

        **Inference Runtime**  
        Pure-Python LightGBM tree evaluator

        **Task**  
        Pre-appointment no-show risk prioritization

        **Primary Use**  
        Reminder prioritization

        **Operational View**  
        Capacity-based Top-K ranking
        """
    )

    st.divider()

    st.caption(
        "Risk scores are primarily used to rank "
        "appointments for reminder outreach."
    )

    st.warning(
        "Risk score ≠ certainty of no-show."
    )

    st.warning(
        "Threshold 0.44 is preserved for historical "
        "experiment audit only."
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "📅 NoShow Predictor"
)

st.write(
    """
    Upload upcoming appointment data and generate a
    **no-show risk prioritization list** to help scheduling
    staff decide which appointments should receive
    reminder outreach first.
    """
)


# =========================================================
# ABOUT
# =========================================================

with st.expander(
    "ℹ️ About this MVP",
    expanded=False,
):

    st.markdown(
        """
        This project uses pre-appointment information to rank
        upcoming appointments according to model risk score.

        The main operational output is:

        **highest-risk appointments → reminder priority**

        The system does **not** determine whether an appointment
        will definitely become a no-show.

        It is also not intended for:

        - diagnosis;
        - treatment decisions;
        - clinical urgency;
        - patient punishment;
        - automatic appointment cancellation.
        """
    )


# =========================================================
# REQUIRED CSV
# =========================================================

with st.expander(
    "📄 Required CSV format",
    expanded=False,
):

    st.markdown(
        """
        Required columns:
        """
    )

    example = pd.DataFrame(
        {
            "appointment_date": [
                "09/09/2021",
                "10/09/2021",
                "11/09/2021",
            ],

            "appointment_time": [
                "13:20",
                "09:00",
                "15:30",
            ],

            "specialty": [
                "physiotherapy",
                "psychotherapy",
                "speech therapy",
            ],

            "appointment_shift": [
                "afternoon",
                "morning",
                "afternoon",
            ],

            "age": [
                30,
                45,
                67,
            ],

            "under_12_years_old": [
                0,
                0,
                0,
            ],

            "over_60_years_old": [
                0,
                0,
                1,
            ],

            "patient_needs_companion": [
                0,
                0,
                1,
            ],

            "entry_service_date": [
                "05/02/2020",
                "12/04/2019",
                "01/06/2021",
            ],
        }
    )

    st.dataframe(
        example,
        width="stretch",
        hide_index=True,
    )

    st.info(
        "Target columns such as `no_show` are not required "
        "during inference."
    )


# =========================================================
# 1. UPLOAD
# =========================================================

st.subheader(
    "1. Upload Appointment Data"
)


uploaded_file = st.file_uploader(
    "Upload appointment CSV",
    type=[
        "csv"
    ],
)


if uploaded_file is None:

    st.info(
        "Upload a CSV file to begin."
    )

    st.stop()


try:

    input_df = pd.read_csv(
        uploaded_file
    )

except Exception as exc:

    st.error(
        "Could not read the uploaded CSV."
    )

    st.code(
        str(exc)
    )

    st.stop()


# =========================================================
# 2. VALIDATE INPUT
# =========================================================

st.subheader(
    "2. Validate Input"
)


valid, validation_message = (
    validate_input(
        input_df
    )
)


if not valid:

    st.error(
        validation_message
    )

    st.stop()


st.success(
    "✅ Appointment data is valid."
)


appointment_dates = pd.to_datetime(
    input_df[
        "appointment_date"
    ],
    dayfirst=True,
    errors="coerce",
)


col1, col2, col3 = st.columns(
    3
)


with col1:

    st.metric(
        "Appointments",
        f"{len(input_df):,}",
    )


with col2:

    if not appointment_dates.isna().all():

        earliest = (
            appointment_dates
            .min()
            .strftime(
                "%Y-%m-%d"
            )
        )

    else:

        earliest = "-"


    st.metric(
        "Earliest Appointment",
        earliest,
    )


with col3:

    if not appointment_dates.isna().all():

        latest = (
            appointment_dates
            .max()
            .strftime(
                "%Y-%m-%d"
            )
        )

    else:

        latest = "-"


    st.metric(
        "Latest Appointment",
        latest,
    )


with st.expander(
    "Preview uploaded appointments",
):

    st.dataframe(
        input_df.head(
            100
        ),
        width="stretch",
        hide_index=True,
    )


# =========================================================
# 3. REMINDER CAPACITY
# =========================================================

st.subheader(
    "3. Set Reminder Capacity"
)


default_capacity_percent = int(
    DEFAULT_CAPACITY_FRACTION
    * 100
)


capacity_percent = st.slider(
    (
        "Percentage of appointments that can receive "
        "prioritized reminders"
    ),

    min_value=5,
    max_value=50,

    value=(
        default_capacity_percent
    ),

    step=5,
)


capacity_fraction = (
    capacity_percent
    / 100.0
)


estimated_priority_count = max(
    1,

    int(
        -(
            -len(input_df)
            * capacity_percent
            // 100
        )
    ),
)


st.info(
    f"Top {capacity_percent}% highest-scoring appointments "
    f"will be prioritized — approximately "
    f"**{estimated_priority_count} of {len(input_df)} appointments**."
)


# =========================================================
# 4. GENERATE
# =========================================================

st.subheader(
    "4. Generate Risk Prioritization"
)


generate = st.button(
    "🚀 Generate Reminder Priority List",
    type="primary",
    width="stretch",
)


if generate:

    try:

        with st.spinner(
            "Evaluating 500 LightGBM trees..."
        ):

            result_df = (
                predictor.predict(
                    input_df,

                    capacity_fraction=(
                        capacity_fraction
                    ),

                    include_threshold_flag=True,
                )
            )


        st.session_state[
            "noshow_result_df"
        ] = result_df


        st.session_state[
            "noshow_capacity_fraction"
        ] = capacity_fraction


    except Exception as exc:

        st.error(
            "Inference failed."
        )

        st.code(
            str(exc)
        )

        st.stop()


if (
    "noshow_result_df"
    not in st.session_state
):

    st.stop()


result_df = st.session_state[
    "noshow_result_df"
]


used_capacity_fraction = (
    st.session_state[
        "noshow_capacity_fraction"
    ]
)


summary = (
    predictor.summarize(
        result_df,

        capacity_fraction=(
            used_capacity_fraction
        ),
    )
)


# =========================================================
# SUCCESS
# =========================================================

st.success(
    "✅ Risk prioritization generated successfully."
)


# =========================================================
# SUMMARY
# =========================================================

st.subheader(
    "Reminder Priority Summary"
)


metric1, metric2, metric3, metric4 = (
    st.columns(
        4
    )
)


with metric1:

    st.metric(
        "Appointments",
        f"{summary.rows:,}",
    )


with metric2:

    st.metric(
        "Priority List",
        f"{summary.priority_count:,}",
    )


with metric3:

    st.metric(
        "Capacity",
        (
            f"{summary.capacity_fraction * 100:.0f}%"
        ),
    )


with metric4:

    st.metric(
        "Mean Model Score",
        (
            f"{summary.mean_risk_score:.3f}"
        ),
    )


st.caption(
    "The model score is primarily a ranking signal, "
    "not a guaranteed probability of missing an appointment."
)


# =========================================================
# PRIORITY TABLE
# =========================================================

st.divider()


st.subheader(
    (
        f"Top "
        f"{used_capacity_fraction * 100:.0f}% "
        f"Reminder Priority"
    )
)


priority_df = (
    result_df[
        result_df[
            "priority_top_capacity"
        ]
    ]
    .copy()
)


priority_columns = [
    "priority_rank",
    "no_show_risk_score",
]


optional_columns = [
    "appointment_date",
    "appointment_time",
    "specialty",
    "appointment_shift",
    "age",
]


for column in optional_columns:

    if (
        column
        in priority_df.columns
    ):

        priority_columns.append(
            column
        )


priority_view = (
    priority_df[
        priority_columns
    ]
    .copy()
)


priority_view = (
    priority_view.rename(
        columns={

            "priority_rank":
                "Rank",

            "no_show_risk_score":
                "Model Score",

            "appointment_date":
                "Appointment Date",

            "appointment_time":
                "Time",

            "specialty":
                "Specialty",

            "appointment_shift":
                "Shift",

            "age":
                "Age",
        }
    )
)


if (
    "Model Score"
    in priority_view.columns
):

    priority_view[
        "Model Score"
    ] = (
        priority_view[
            "Model Score"
        ]
        .round(
            4
        )
    )


st.dataframe(
    priority_view,
    width="stretch",
    hide_index=True,
)


# =========================================================
# SCORE RANGE
# =========================================================

score1, score2 = st.columns(
    2
)


with score1:

    st.metric(
        "Highest Score",
        f"{summary.max_risk_score:.4f}",
    )


with score2:

    st.metric(
        "Lowest Score",
        f"{summary.min_risk_score:.4f}",
    )


# =========================================================
# SCORE CHART
# =========================================================

st.subheader(
    "Ranked Model Scores"
)


score_chart = (
    result_df[
        [
            "priority_rank",
            "no_show_risk_score",
        ]
    ]
    .rename(
        columns={
            "priority_rank":
                "Rank",

            "no_show_risk_score":
                "Model Score",
        }
    )
    .set_index(
        "Rank"
    )
)


st.line_chart(
    score_chart,
    width="stretch",
)


# =========================================================
# ALL APPOINTMENTS
# =========================================================

st.subheader(
    "All Ranked Appointments"
)


all_columns = [
    "priority_rank",
    "no_show_risk_score",
    "priority_top_capacity",
]


for column in optional_columns:

    if (
        column
        in result_df.columns
    ):

        all_columns.append(
            column
        )


all_view = (
    result_df[
        all_columns
    ]
    .copy()
)


all_view = (
    all_view.rename(
        columns={

            "priority_rank":
                "Rank",

            "no_show_risk_score":
                "Model Score",

            "priority_top_capacity":
                "Reminder Priority",

            "appointment_date":
                "Appointment Date",

            "appointment_time":
                "Time",

            "specialty":
                "Specialty",

            "appointment_shift":
                "Shift",

            "age":
                "Age",
        }
    )
)


all_view[
    "Model Score"
] = (
    all_view[
        "Model Score"
    ]
    .round(
        4
    )
)


st.dataframe(
    all_view,
    width="stretch",
    hide_index=True,
)


# =========================================================
# APPOINTMENT INSPECTOR
# =========================================================

st.subheader(
    "Appointment Inspector"
)


selected_rank = st.selectbox(
    "Choose ranked appointment",
    result_df[
        "priority_rank"
    ].tolist(),
)


selected = (
    result_df[
        result_df[
            "priority_rank"
        ]
        == selected_rank
    ]
    .iloc[
        0
    ]
)


inspect1, inspect2, inspect3, inspect4 = (
    st.columns(
        4
    )
)


with inspect1:

    st.metric(
        "Rank",

        int(
            selected[
                "priority_rank"
            ]
        ),
    )


with inspect2:

    st.metric(
        "Model Score",

        (
            f"{float(selected['no_show_risk_score']):.4f}"
        ),
    )


with inspect3:

    st.metric(
        "Reminder Priority",

        (
            "Yes"

            if bool(
                selected[
                    "priority_top_capacity"
                ]
            )

            else "No"
        ),
    )


with inspect4:

    if (
        "appointment_date"
        in selected.index
    ):

        st.metric(
            "Appointment",

            str(
                selected[
                    "appointment_date"
                ]
            ),
        )

    else:

        st.metric(
            "Appointment",
            "-",
        )


if (
    "specialty"
    in selected.index
):

    st.caption(
        "Specialty: "
        f"{selected['specialty']}"
    )


# =========================================================
# HISTORICAL THRESHOLD
# =========================================================

st.divider()


st.subheader(
    "Historical Threshold Audit"
)


st.warning(
    f"""
    The original experiment selected threshold
    **{VALIDATION_THRESHOLD:.2f}**
    using validation data.

    The champion was later refit on
    Train + Validation.

    Because refitting changed the score distribution,
    this threshold is preserved mainly for
    **audit and reproducibility**.

    The operational UI therefore uses
    **Top-{used_capacity_fraction * 100:.0f}% ranking**
    for reminder prioritization.
    """
)


threshold_flag_count = int(
    result_df[
        "threshold_flag"
    ].sum()
)


threshold1, threshold2 = (
    st.columns(
        2
    )
)


with threshold1:

    st.metric(
        "Historical Threshold",
        f"{VALIDATION_THRESHOLD:.2f}",
    )


with threshold2:

    st.metric(
        "Rows ≥ Threshold",
        f"{threshold_flag_count:,}",
    )


with st.expander(
    "Why is threshold 0.44 not the default operational rule?"
):

    st.markdown(
        """
        The threshold was selected from the validation-stage
        prediction distribution.

        The champion was then refit using Train + Validation.

        After refitting, the score distribution changed.

        In the held-out test data, threshold 0.44 flagged a
        large share of appointments.

        For this MVP, the primary operational view is therefore
        **capacity-based ranking** rather than automatic binary
        classification.

        The threshold remains visible because it is part of the
        experiment history.
        """
    )


# =========================================================
# PERFORMANCE CONTEXT
# =========================================================

st.subheader(
    "Held-Out Test Context"
)


perf1, perf2, perf3, perf4 = (
    st.columns(
        4
    )
)


with perf1:

    st.metric(
        "PR-AUC",
        "0.182",
    )


with perf2:

    st.metric(
        "ROC-AUC",
        "0.561",
    )


with perf3:

    st.metric(
        "Top-10% Lift",
        "1.125×",
    )


with perf4:

    st.metric(
        "Test Base Rate",
        "16.07%",
    )


st.caption(
    """
    These held-out results show a modest ranking signal.
    The model should therefore be treated as operational
    decision support for reminder prioritization, not as a
    definitive no-show predictor.
    """
)


# =========================================================
# EXPORT
# =========================================================

st.divider()


st.subheader(
    "5. Export Result"
)


save_col, download_col = (
    st.columns(
        2
    )
)


with save_col:

    if st.button(
        "💾 Save Result",
        width="stretch",
    ):

        try:

            output_path = (
                save_prediction_result(
                    result_df,

                    ROOT
                    / "tests"
                    / "results",
                )
            )


            st.success(
                "Saved: "
                f"{output_path.relative_to(ROOT)}"
            )


        except Exception as exc:

            st.error(
                "Could not save result."
            )

            st.code(
                str(exc)
            )


with download_col:

    st.download_button(
        "⬇️ Download Priority List CSV",

        data=(
            dataframe_to_csv_bytes(
                result_df
            )
        ),

        file_name=(
            "noshow_priority_list.csv"
        ),

        mime="text/csv",

        width="stretch",
    )


# =========================================================
# RESPONSIBLE USE
# =========================================================

with st.expander(
    "⚠️ How to interpret this result",
):

    st.markdown(
        """
        ### Model score

        A higher score means the appointment is ranked higher
        by the model.

        The score is not a guarantee that the appointment will
        become a no-show.

        ### Capacity-based prioritization

        Example:

        - 100 upcoming appointments;
        - reminder capacity = 10%;
        - the 10 highest-scoring appointments are prioritized.

        ### Human-in-the-loop

        Scheduling or service-operations staff remain responsible
        for any action based on this list.

        The model must not automatically:

        - cancel appointments;
        - deny access to service;
        - change clinical priority;
        - punish patients;
        - make diagnoses.

        ### Current limitations

        The held-out ranking signal is modest.

        The dataset also lacks several potentially useful
        pre-appointment signals, such as:

        - booking creation timestamp / lead time;
        - stable patient identifier;
        - stable appointment identifier;
        - richer prior attendance history;
        - validated reminder-contact history.

        The source dataset comes from a medical-service setting.

        Transfer to salons, studios, or other service businesses
        has not been validated.
        """
    )
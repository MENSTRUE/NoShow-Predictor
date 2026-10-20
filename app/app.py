from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.inference import (
    DEFAULT_CAPACITY_FRACTION,
    VALIDATION_THRESHOLD,
    NoShowPredictor,
)
from src.preprocessing import validate_input
from src.utils import (
    dataframe_to_csv_bytes,
    save_prediction_result,
)


MODEL_PATH = (
    ROOT
    / "models"
    / "noshow_champion.joblib"
)


st.set_page_config(
    page_title="NoShow Predictor",
    page_icon="📅",
    layout="wide",
)


@st.cache_resource
def load_predictor():
    return NoShowPredictor(
        model_path=MODEL_PATH
    )


predictor = load_predictor()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("NoShow Predictor")

    st.markdown(
        """
        **Champion Model**  
        LGBM-Balanced

        **Task**  
        Pre-appointment no-show risk prioritization

        **Primary Use**  
        Reminder prioritization

        **Operational View**  
        Top 10% highest-risk appointments
        """
    )

    st.divider()

    st.caption(
        "Risk scores are used to rank reminder priority. "
        "They are not guaranteed no-show probabilities."
    )

    st.warning(
        "Threshold 0.44 is preserved as a historical "
        "validation threshold only."
    )


# =========================================================
# HEADER
# =========================================================

st.title("📅 NoShow Predictor")

st.write(
    """
    Upload upcoming appointment data and generate a
    **no-show risk prioritization list** to help scheduling
    staff decide which appointments should receive reminder
    outreach first.
    """
)


# =========================================================
# REQUIRED CSV FORMAT
# =========================================================

with st.expander(
    "📄 Required CSV format",
    expanded=False,
):
    st.markdown(
        """
        The current inference pipeline requires these columns:
        """
    )

    example = pd.DataFrame(
        {
            "appointment_date": [
                "01/11/2026",
                "02/11/2026",
                "03/11/2026",
            ],
            "appointment_time": [
                "09:30",
                "13:00",
                "16:30",
            ],
            "specialty": [
                "General",
                "General",
                "General",
            ],
            "appointment_shift": [
                "Morning",
                "Afternoon",
                "Evening",
            ],
            "age": [
                28,
                42,
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
                "01/01/2024",
                "15/03/2023",
                "20/07/2022",
            ],
        }
    )

    st.dataframe(
        example,
        width="stretch",
        hide_index=True,
    )

    st.info(
        "The app does not require target labels such as "
        "'no_show' because this is inference on future appointments."
    )


# =========================================================
# STEP 1 — UPLOAD
# =========================================================

st.subheader(
    "1. Upload Appointment Data"
)


uploaded_file = st.file_uploader(
    "Upload appointment CSV",
    type=["csv"],
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
        f"Could not read CSV: {exc}"
    )

    st.stop()


# =========================================================
# STEP 2 — VALIDATION
# =========================================================

st.subheader(
    "2. Validate Input"
)


valid, message = validate_input(
    input_df
)


if not valid:
    st.error(
        message
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


c1, c2, c3 = st.columns(3)


with c1:
    st.metric(
        "Appointments",
        len(
            input_df
        ),
    )


with c2:
    if not appointment_dates.isna().all():
        start_date = (
            appointment_dates.min()
            .strftime("%Y-%m-%d")
        )
    else:
        start_date = "-"

    st.metric(
        "Earliest Appointment",
        start_date,
    )


with c3:
    if not appointment_dates.isna().all():
        end_date = (
            appointment_dates.max()
            .strftime("%Y-%m-%d")
        )
    else:
        end_date = "-"

    st.metric(
        "Latest Appointment",
        end_date,
    )


with st.expander(
    "Preview uploaded appointments",
):
    st.dataframe(
        input_df.head(100),
        width="stretch",
        hide_index=True,
    )


# =========================================================
# STEP 3 — CAPACITY
# =========================================================

st.subheader(
    "3. Set Reminder Capacity"
)


capacity_percent = st.slider(
    "Percentage of appointments that can receive prioritized reminders",
    min_value=5,
    max_value=50,
    value=int(
        DEFAULT_CAPACITY_FRACTION
        * 100
    ),
    step=5,
)


capacity_fraction = (
    capacity_percent
    / 100.0
)


st.info(
    f"Top {capacity_percent}% highest-risk appointments "
    "will be marked as reminder priority."
)


# =========================================================
# STEP 4 — GENERATE
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
            "Scoring appointments..."
        ):
            result_df = predictor.predict(
                input_df,
                capacity_fraction=capacity_fraction,
                include_threshold_flag=True,
            )

        st.session_state[
            "noshow_result_df"
        ] = result_df

        st.session_state[
            "noshow_capacity_fraction"
        ] = capacity_fraction

    except Exception as exc:
        st.error(
            f"Inference failed: {exc}"
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


capacity_fraction = st.session_state[
    "noshow_capacity_fraction"
]


summary = predictor.summarize(
    result_df,
    capacity_fraction=capacity_fraction,
)


# =========================================================
# SUMMARY
# =========================================================

st.success(
    "✅ Risk prioritization generated successfully."
)


st.subheader(
    "Reminder Priority Summary"
)


m1, m2, m3, m4 = st.columns(
    4
)


with m1:
    st.metric(
        "Appointments",
        summary.rows,
    )


with m2:
    st.metric(
        "Priority List",
        summary.priority_count,
    )


with m3:
    st.metric(
        "Capacity",
        f"{summary.capacity_fraction * 100:.0f}%",
    )


with m4:
    st.metric(
        "Mean Risk Score",
        f"{summary.mean_risk_score:.3f}",
    )


# =========================================================
# PRIMARY PRIORITY LIST
# =========================================================

st.divider()

st.subheader(
    f"Top {capacity_fraction * 100:.0f}% Reminder Priority"
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
    if column in priority_df.columns:
        priority_columns.append(
            column
        )


priority_view = priority_df[
    priority_columns
].copy()


priority_view = priority_view.rename(
    columns={
        "priority_rank": "Rank",
        "no_show_risk_score": "Risk Score",
        "appointment_date": "Appointment Date",
        "appointment_time": "Time",
        "specialty": "Specialty",
        "appointment_shift": "Shift",
        "age": "Age",
    }
)


if "Risk Score" in priority_view.columns:
    priority_view[
        "Risk Score"
    ] = priority_view[
        "Risk Score"
    ].round(4)


st.dataframe(
    priority_view,
    width="stretch",
    hide_index=True,
)


# =========================================================
# RISK SCORE DISTRIBUTION
# =========================================================

st.subheader(
    "Risk Score Distribution"
)


chart_df = (
    result_df[
        [
            "priority_rank",
            "no_show_risk_score",
        ]
    ]
    .set_index(
        "priority_rank"
    )
)


st.line_chart(
    chart_df,
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
    if column in result_df.columns:
        all_columns.append(
            column
        )


all_view = result_df[
    all_columns
].copy()


all_view = all_view.rename(
    columns={
        "priority_rank": "Rank",
        "no_show_risk_score": "Risk Score",
        "priority_top_capacity": "Reminder Priority",
        "appointment_date": "Appointment Date",
        "appointment_time": "Time",
        "specialty": "Specialty",
        "appointment_shift": "Shift",
        "age": "Age",
    }
)


all_view[
    "Risk Score"
] = all_view[
    "Risk Score"
].round(4)


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


selected_row = result_df[
    result_df[
        "priority_rank"
    ]
    == selected_rank
].iloc[0]


a1, a2, a3, a4 = st.columns(
    4
)


with a1:
    st.metric(
        "Rank",
        int(
            selected_row[
                "priority_rank"
            ]
        ),
    )


with a2:
    st.metric(
        "Risk Score",
        f"{float(selected_row['no_show_risk_score']):.4f}",
    )


with a3:
    st.metric(
        "Reminder Priority",
        (
            "Yes"
            if bool(
                selected_row[
                    "priority_top_capacity"
                ]
            )
            else "No"
        ),
    )


with a4:
    if "appointment_date" in selected_row.index:
        st.metric(
            "Appointment",
            str(
                selected_row[
                    "appointment_date"
                ]
            ),
        )

    else:
        st.metric(
            "Appointment",
            "-",
        )


if "specialty" in selected_row.index:
    st.caption(
        f"Specialty: {selected_row['specialty']}"
    )


# =========================================================
# AUDIT-ONLY THRESHOLD
# =========================================================

st.divider()

st.subheader(
    "Historical Threshold Audit"
)


st.warning(
    f"""
    The experiment selected a threshold of **{VALIDATION_THRESHOLD:.2f}**
    on validation data.

    This threshold is shown for audit and experiment traceability.

    The current operational app prioritizes reminders using
    **Top-{capacity_fraction * 100:.0f}% ranking**, not automatic
    classification from this threshold.
    """
)


threshold_flag_count = int(
    result_df[
        "threshold_flag"
    ].sum()
)


t1, t2 = st.columns(
    2
)


with t1:
    st.metric(
        "Historical Threshold",
        f"{VALIDATION_THRESHOLD:.2f}",
    )


with t2:
    st.metric(
        "Appointments Above Threshold",
        threshold_flag_count,
    )


with st.expander(
    "Why is threshold 0.44 not the default operational rule?"
):
    st.markdown(
        """
        The threshold was selected using validation predictions.

        The final model was later refit on a larger training set,
        which can change the score distribution.

        In the held-out test data, applying 0.44 flagged a large
        fraction of appointments.

        For this MVP, reminder prioritization is therefore shown
        primarily as a **capacity-based ranking problem**.

        The threshold remains visible for reproducibility and audit,
        not as an automatic cancellation or punitive decision rule.
        """
    )


# =========================================================
# PERFORMANCE CONTEXT
# =========================================================

st.subheader(
    "Model Performance Context"
)


p1, p2, p3 = st.columns(
    3
)


with p1:
    st.metric(
        "Test PR-AUC",
        "0.182",
    )


with p2:
    st.metric(
        "Test ROC-AUC",
        "0.561",
    )


with p3:
    st.metric(
        "Top-10% Lift",
        "1.125×",
    )


st.caption(
    """
    These results indicate a modest ranking signal.
    The model should be used as reminder-prioritization support,
    not as a definitive prediction of who will miss an appointment.
    """
)


# =========================================================
# EXPORT
# =========================================================

st.divider()

st.subheader(
    "5. Export Result"
)


col_save, col_download = st.columns(
    2
)


with col_save:
    if st.button(
        "💾 Save Result",
        width="stretch",
    ):
        try:
            output_path = save_prediction_result(
                result_df,
                ROOT
                / "tests"
                / "results",
            )

            st.success(
                "Saved: "
                f"{output_path.relative_to(ROOT)}"
            )

        except Exception as exc:
            st.error(
                f"Could not save result: {exc}"
            )


with col_download:
    st.download_button(
        "⬇️ Download Priority List CSV",
        data=dataframe_to_csv_bytes(
            result_df
        ),
        file_name=(
            "noshow_priority_list.csv"
        ),
        mime="text/csv",
        width="stretch",
    )


# =========================================================
# INTERPRETATION
# =========================================================

with st.expander(
    "⚠️ How to interpret this result",
):
    st.markdown(
        """
        ### Risk score

        The model score is used to **rank appointments**.

        A higher score means the model places that appointment
        higher in the reminder-priority ordering.

        It does **not** mean the appointment is guaranteed to become
        a no-show.

        ### Reminder priority

        The current operational view uses the top fraction of
        appointments according to available reminder capacity.

        Example:

        - 100 appointments;
        - reminder capacity = 10%;
        - the 10 highest-scoring appointments become priority outreach.

        ### Human-in-the-loop

        The result should be reviewed by scheduling or service
        operations staff.

        It must not be used to:

        - cancel appointments automatically;
        - reduce service quality;
        - influence clinical priority;
        - punish patients;
        - make diagnostic decisions.

        ### Current limitations

        The model has only modest ranking performance.

        The available dataset also lacks several potentially useful
        pre-appointment signals such as:

        - booking creation timestamp / lead time;
        - validated reminder history;
        - richer attendance history;
        - stable appointment identifier;
        - stable patient identifier;
        - portable cross-business context.

        The dataset originates from a medical-service setting, so
        transfer to salons, studios, or other service businesses is
        not established.
        """
    )
# AGENTS.md — NoShow Predictor

This file defines the operational rules for coding agents, AI assistants, and contributors working in this repository.

The goal is not merely to keep the code running. The goal is to preserve experiment validity, claim boundaries, reproducibility, and the distinction between the original evaluated model and deployment adaptations.

## 1. Project Identity

Project: **NoShow Predictor**  
Portfolio milestone: **14%**  
Primary task: **pre-appointment no-show risk prioritization**  
Primary user: scheduling / service-operations staff  
Primary action: prioritize reminder outreach under limited capacity.

This is not a clinical model.

## 2. Source of Truth

When values disagree, use this precedence:

1. `reports/final_summary.json`
2. the specific machine-readable report for the metric or policy;
3. final notebook outputs in `notebooks/noshow-predictor.ipynb`;
4. current code behavior;
5. README prose;
6. screenshots, posts, or historical discussion.

Do not silently overwrite report-backed metrics with values inferred from app screenshots.

## 3. Champion Model

Champion:

```text
LGBM-Balanced
```

Selection metric:

```text
validation PR-AUC
```

Validation PR-AUC:

```text
0.1560918085
```

Historical validation-selected threshold:

```text
0.44
```

Official held-out test:

```text
PR-AUC: 0.1819318916
ROC-AUC: 0.5609038571
Brier: 0.2522836700
Precision @ 0.44: 0.1793893130
Recall @ 0.44: 0.7563218391
F1 @ 0.44: 0.2899955928
F2 @ 0.44: 0.4602686066
Predicted-positive rate @ 0.44: 0.6775027706
N: 2707
```

Top-10% held-out capacity:

```text
k: 271
captured no-shows: 49
precision: 0.1808118081
recall: 0.1126436782
lift vs random: 1.1251898036
base rate: 0.1606944958
```

Never claim this is a high-accuracy model. The ranking signal is modest.

## 4. Claim Boundaries

### Allowed

Agents may describe the system as:

- pre-appointment no-show risk prioritization;
- reminder-prioritization support;
- capacity-aware ranking;
- a temporally evaluated ML MVP;
- a human-in-the-loop operational aid.

### Forbidden / unsupported

Do not describe it as:

- diagnosis;
- clinical risk prediction;
- clinical urgency scoring;
- automated patient cancellation;
- patient reliability scoring;
- punishment or denial of service;
- universally calibrated no-show probability;
- proven transferable to salons, studios, or other domains.

Use:

```text
risk score
ranking score
reminder priority
```

Do not equate the score with certainty.

## 5. Dataset Contract

Canonical source:

```text
Medical Appointments No-Show
```

Canonical Kaggle path:

```text
/kaggle/input/datasets/wafabila/medical-appointments-no-show/Medical Appointments No-Show/medical-appointments-no-show-en.csv
```

Audited source facts:

```text
rows: 49,593
source columns: 26
date range: 2016-06-20 → 2022-02-25
unique dates: 1,001
no-shows: 4,832
shows: 44,761
overall no-show rate: 9.7433%
exact full-row duplicates: 2,921
missing cells: 134,394
```

Do not change these values unless the source dataset or canonical audit changes and the notebook is rerun.

## 6. Duplicate Policy

Do not automatically drop exact duplicate rows.

Reason:

- no stable patient ID;
- no stable appointment ID;
- identical records may represent separate appointments.

Any future deduplication must be justified with a stable identity key or a defensible event-level policy.

## 7. Leakage and Sensitive-Feature Policy

Never add the following to the champion feature set without a formal redesign:

```text
no_show_reason
icd
gender
city
disability
date_of_birth
same-day observed weather fields
```

Rules:

- `no_show_reason`: post-outcome leakage.
- `icd`: sensitive clinical diagnosis proxy.
- `gender`: audit only.
- `disability`: audit only.
- `city`: privacy/location proxy.
- `date_of_birth`: raw personal date; age is the allowed derived field.
- same-day weather: not guaranteed available at reminder time.

If a contributor proposes adding any of these, document the product-time availability, privacy rationale, and leakage analysis first.

## 8. Feature Contract

Model features must remain in this exact order:

```text
age
under_12_years_old
over_60_years_old
patient_needs_companion
appointment_hour
appointment_minute
appointment_dow
appointment_day
appointment_month_num
is_weekend
dow_sin
dow_cos
month_sin
month_cos
service_tenure_days
specialty_missing
age_missing
tenure_missing
specialty
appointment_shift
```

Raw inference input contract:

```text
appointment_date
appointment_time
specialty
appointment_shift
age
under_12_years_old
over_60_years_old
patient_needs_companion
entry_service_date
```

Do not duplicate feature logic independently in the Streamlit app. Use `src/preprocessing.py`.

## 9. Temporal Evaluation Contract

Configured split:

```text
Train <= 2020-12-31
Validation 2021-01-01 → 2021-09-30
Test 2021-10-01 → 2022-02-25
```

Observed:

```text
Train: 39,643 rows, 2016-06-20 → 2020-12-16
Validation: 7,243 rows, 2021-01-27 → 2021-09-24
Test: 2,707 rows, 2021-10-05 → 2022-02-25
```

Never replace the canonical test with a random split while still presenting the existing metrics as comparable.

## 10. Model Selection Rules

Champion selection must use validation data only.

Held-out test results must not be used to retroactively select a different champion.

If new models are introduced:

1. train on the permitted training window;
2. evaluate on validation;
3. choose using a declared validation metric;
4. lock the decision;
5. refit according to the documented protocol;
6. evaluate test once;
7. preserve the previous experiment for comparison.

## 11. Threshold Policy

`0.44` is a historical validation-selected threshold.

Do not use it as the default operational rule merely because it exists.

Why:

- final refit changed the score distribution;
- held-out test predicted-positive rate at 0.44 is ~67.75%.

Primary product behavior:

```text
capacity-based Top-K / Top-X% ranking
```

If a future threshold is promoted, it requires a new calibration/validation protocol and updated reports.

## 12. Model Artifact Policy

Current artifacts:

```text
models/noshow_champion.joblib
models/noshow_champion.txt
models/noshow_champion_model.json
```

Roles:

- `.joblib`: original serialized training object;
- `.txt`: native LightGBM export;
- `.json`: tree dump used by current local pure-Python runtime.

Do not delete historical model artifacts merely because only one is used by the app.

## 13. Windows Deployment Rule

Native LightGBM prediction caused a Windows access-violation in the local environment.

The current app must use:

```text
models/noshow_champion_model.json
```

through the pure-Python tree evaluator in `src/inference.py`.

Do not switch back to `Booster.predict()` in app code without reproducing and resolving the crash.

## 14. Pure-Python Evaluator Policy

`PurePythonLightGBM` exists to preserve the original champion tree structure while avoiding the native runtime failure.

If editing it:

- preserve exact feature order;
- preserve categorical metadata handling;
- preserve missing-value routing;
- preserve split decision semantics;
- preserve the binary sigmoid objective;
- add or update tests;
- perform parity testing against the native training environment whenever possible.

Do not claim bit-for-bit parity unless it has actually been measured.

## 15. App Policy

The Streamlit application should:

- validate raw inputs;
- show row/date summary;
- allow reminder-capacity selection;
- generate scores;
- rank descending;
- identify Top-K priority rows;
- expose threshold only as historical audit context;
- retain human-in-the-loop messaging;
- allow CSV export.

The app must not:

- display “cancel appointment” actions;
- claim certainty;
- present clinical advice;
- present the historical threshold as an automatic policy.

## 16. Repository Directory Contracts

```text
app/
    Streamlit product UI only.

assets/dataset/
    Dataset-audit visualizations.

assets/training/
    Validation/model-development visualizations.

assets/evaluation/
    Held-out evaluation visualizations.

models/
    Promoted/historical model artifacts.

notebooks/
    Canonical research/training workflow.

reports/
    Machine-readable experiment evidence.

src/
    Reusable inference, preprocessing, and utilities.

tests/
    Automated checks and generated local test outputs.
```

Do not scatter model outputs or reports into arbitrary directories.

## 17. Asset Policy

Current committed visual assets:

```text
assets/dataset/missingness.png
assets/dataset/monthly_no_show_rate.png
assets/dataset/target_distribution.png
assets/training/feature_importance_top20.png
assets/training/validation_capacity_tradeoff.png
assets/training/validation_threshold_search.png
assets/evaluation/test_confusion_matrix.png
assets/evaluation/test_pr_curve.png
assets/evaluation/test_roc_curve.png
```

README should render committed assets, not merely list filenames.

App screenshots should go to:

```text
assets/screenshots/
```

when added.

## 18. Reports Policy

Machine-readable reports are experiment evidence.

Do not casually edit metric CSV/JSON files by hand.

If results change:

1. rerun the responsible experiment;
2. regenerate the report;
3. inspect the diff;
4. update README only after the report is final.

## 19. Testing Policy

Run:

```bash
python -m pytest -q
```

before committing inference or preprocessing changes.

Tests should cover at minimum:

- schema validation;
- derived features;
- negative tenure;
- categorical behavior;
- model load;
- tree count;
- capacity ranking;
- threshold audit;
- end-to-end inference.

If a bug was discovered in production-like use, add a regression test before closing the change.

## 20. Dependency Policy

Current requirements intentionally retain training dependencies even though the app prediction path uses JSON evaluation.

Do not remove:

```text
scikit-learn
lightgbm
joblib
```

without considering notebook and artifact reproducibility.

`scikit-learn==1.6.1` is pinned because the historical serialized estimator was created with that version.

## 21. Privacy and Security

Do not log or commit real personally identifying appointment records.

Sample inference files should be synthetic or suitably de-identified.

Never add:

- names;
- phone numbers;
- addresses;
- national identifiers;
- raw medical diagnoses

to example data simply to make the app look realistic.

## 22. README Rules

README is a public source of truth.

When updating it:

- use actual audited numbers;
- keep model-selection discipline visible;
- render available assets;
- include limitations;
- distinguish threshold audit from operational policy;
- distinguish source dataset rights from MIT code license;
- update Before → Now → Next when project status changes.

## 23. Git Rules

Use conventional, descriptive commits.

Examples:

```text
docs: add full project source-of-truth documentation
test: add inference regression coverage
fix: preserve categorical routing in JSON evaluator
feat: add capacity-aware reminder export
```

Never force-push unless explicitly required and reviewed.

Before any destructive clean:

```bash
git clean -fdn
```

Only after reviewing the dry run may a destructive clean be considered.

## 24. Model Promotion Checklist

A candidate can replace the champion only if all are true:

- [ ] problem framing remains valid;
- [ ] feature availability is pre-appointment;
- [ ] no prohibited leakage feature is added;
- [ ] validation protocol is comparable;
- [ ] selection metric is declared in advance;
- [ ] candidate wins on validation according to the declared rule;
- [ ] test remains held out until selection is locked;
- [ ] capacity metrics are reported;
- [ ] failure analysis is regenerated;
- [ ] subgroup audit is regenerated;
- [ ] reports are updated;
- [ ] app behavior is reviewed;
- [ ] README/AGENTS are updated;
- [ ] tests pass.

## 25. Definition of Done

A change is done only when:

- code runs;
- tests pass;
- no claim boundary is violated;
- outputs are stored in the correct directory;
- report-backed metrics remain consistent;
- README is updated when public behavior changes;
- the app still supports human review;
- no unsupported performance claim is introduced.

## 26. Before → Now → Next

### Before

Basic binary no-show classification idea.

### Now

Capacity-aware, temporally evaluated, human-in-the-loop reminder prioritization MVP with a portable JSON-tree inference runtime.

### Next

Prioritize:

1. booking lead time;
2. stable appointment/patient identity;
3. attendance-history features;
4. reminder-channel history;
5. dedicated calibration data;
6. parity testing for JSON inference;
7. repeated temporal backtesting;
8. post-deployment drift monitoring.

## 27. Final Rule

If uncertain whether a proposed change makes the system look better but makes the evidence weaker, preserve the evidence.

This repository should prefer:

```text
transparent modest performance
```

over:

```text
inflated claims
```

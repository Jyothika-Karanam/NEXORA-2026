# NEXORA 2026 — Gateway Field-Visit Prioritization

## 1. Problem

The field team can visit only **15 gateways per week**, but more gateways may have an elevated risk of meter-read problems.

The goal is to rank gateways each week and identify the **15 gateways most deserving of field attention**.

The solution combines Data Science and Machine Learning to:

* Define what a problematic gateway means.
* Analyze historical meter-read and telemetry behavior.
* Estimate the probability of a future meter-read problem.
* Account for the different costs of unnecessary and missed visits.
* Rank gateways and select the top 15 each week.
* Compare the ML approach with the supplied 3-sigma baseline.

---

## 2. Data Used

The solution uses the challenge-provided datasets:

* `gateway_master.csv`
* `field_visits.csv`
* `meter_read_success.csv`
* `engineer_review_2026-02.xlsx`
* Monthly telemetry parquet files under `data/telemetry/`

The telemetry data covers August 2025 through March 2026.

The meter-read dataset contains weekly gateway-level measurements including:

* Week start
* Gateway ID
* Meters expected
* Meters read

Meter-read success is calculated as:

```text
meters_read / meters_expected
```

Gateway IDs are normalized before joining datasets so that formatting differences do not create false unmatched gateways.

---

## 3. Target Definition

A gateway is considered problematic when its **next-week meter-read success is below 80%**.

This target was selected after comparing different candidate thresholds.

The observed next-week problem rates based on current-week read success were:

| Current-week read success | Next-week problem rate |
| ------------------------- | ---------------------: |
| <50%                      |                100.00% |
| 50–70%                    |                 89.39% |
| 70–80%                    |                 40.60% |
| 80–90%                    |                  8.76% |
| 90–100%                   |                  1.19% |

This shows a strong relationship between current gateway performance and the following week's meter-read problems.

### Why field visits are not used as direct ground truth

Historical field visits were considered as a possible target, but they are operational decisions rather than perfect fault labels.

There were 642 historical visits:

* 223 — Fehler behoben
* 390 — Kein Fehler gefunden
* 29 — Kein Zugang

Therefore, **60.75% of visits ended with "Kein Fehler gefunden"**.

A field visit does not necessarily mean that the gateway had a confirmed physical fault, so visits were not used as the direct positive label.

---

## 4. Cost-Aware Decision

The challenge specifies:

* Unnecessary visit: **€380**
* Missed problematic gateway: **€600**

The independent probability break-even point for making a visit is:

```text
380 / (380 + 600) = 38.78%
```

However, the field team has a hard limit of **15 visits per week**.

Therefore, the final solution ranks gateways according to predicted risk and selects the **top 15 gateways**.

This converts the ML prediction into an operational field-prioritization decision.

---

## 5. Machine Learning Approach

A **Random Forest classifier** is used to predict whether the following week's meter-read success will be below 80%.

### Main input groups

The model uses information available before the prediction week, including:

* Previous meter-read success values
* Lagged read-success values
* Rolling read-success statistics
* Recent changes in read success
* Previous expected and read meter counts
* Historical telemetry aggregates
* Offline-duration behavior
* Connection-related telemetry
* Reboot-related telemetry
* Radio/network-quality telemetry
* Gateway characteristics

Telemetry is aggregated by gateway and Monday-based week.

Future telemetry is not used for a prediction week. For future predictions, the most recently available historical telemetry is used.

---

## 6. Model Validation

The historical evaluation uses a temporal split rather than randomly mixing past and future observations.

The final model was evaluated on future-like historical weeks from the later part of the available meter-read data.

Final validation results:

* ROC-AUC: **0.9312**
* Precision: **0.8254**
* Recall: **0.7536**
* Validation cost: **€111,600**

The final weekly evaluation selected exactly 15 gateways for each evaluated week.

---

## 7. ML vs 3-Sigma Baseline

The ML model was compared with the supplied 3-sigma baseline using the same operational costs:

* €380 for an unnecessary visit
* €600 for a missed problematic gateway
* Maximum 15 selected gateways per week

### Historical evaluation

| Approach         |  Total cost |
| ---------------- | ----------: |
| 3-sigma baseline |    €144,920 |
| Random Forest ML |    €111,600 |
| Cost reduction   | **€33,320** |
| Improvement      |   **23.0%** |

The Random Forest approach therefore achieved a **23.0% lower evaluated operational cost** than the supplied 3-sigma baseline on the historical evaluation period.

The comparison was performed on historical prediction weeks where the following week's actual outcome was available.

---

## 8. Feature Importance

The model indicates that several telemetry and historical-performance features contribute strongly to the predictions.

Examples of important features include:

* Mean offline duration
* Sum of offline duration
* Mean historical offline duration
* Sum of historical offline duration
* Previous-week meter-read success
* Connection-related importance measures

This provides an explanation of why a gateway may receive a high visit-priority score.

Feature importance should be interpreted as model-level importance, not as proof that an individual feature directly causes a gateway failure.

---

## 9. Final Prediction Output

The final submission contains:

```text
120 rows
15 gateways × 8 weeks
```

Prediction period:

```text
2026-02-02 to 2026-03-23
```

Output columns:

```text
week_start
rank
gateway_id
score
reason
```

The predictions were checked using the provided validator.

Validation result:

```text
predictions.csv: OK

15 ranked gateways for each of 8 weeks, 2026-02-02 to 2026-03-23
```

---

## 10. Project Files

The repository is organized with solution code, outputs, documentation, and the resume at the top level. The challenge data is kept under the top-level `data/` directory and is excluded from Git.

```text
NEXORA-2026/
│
├── README.md
├── DECISIONS.md
├── AI-USAGE.md
├── What-it-cannot-do.md
├── 23091A3255.pdf
│
├── baseline_3sigma.py
├── compare_baseline_cost.py
├── step1_data_check.py
├── step2_eda.py
├── step3_target_definition.py
├── step4_ml_model.py
├── validate_submission.py
│
├── predictions.csv
├── predictions_baseline.csv
├── baseline_cost_evaluation.csv
├── baseline_historical_cost_evaluation.csv
│
├── step3_target_definitions.csv
├── step3_read_success_bands.csv
├── step3_cost_analysis.csv
├── step3_gateway_sensitivity.csv
├── step4_cost_thresholds.csv
├── step4_feature_importance.csv
├── step4_model_comparison.csv
├── step4_model_summary.csv
├── step4_weekly_evaluation.csv
│
├── step3_next_week_risk_by_read_success.png
├── step3_cost_vs_threshold.png
├── step3_target_definition_sensitivity.png
├── step4_model_vs_baseline.png
├── step4_feature_importance.png
│
└── data/
    ├── gateway_master.csv
    ├── field_visits.csv
    ├── meter_read_success.csv
    ├── engineer_review_2026-02.xlsx
    ├── telemetry_sample_2025-08.csv
    │
    └── telemetry/
        ├── month=2025-08/
        ├── month=2025-09/
        ├── month=2025-10/
        ├── month=2025-11/
        ├── month=2025-12/
        ├── month=2026-01/
        ├── month=2026-02/
        └── month=2026-03/
```

The `data/` directory is kept local and excluded from Git using `.gitignore`, because the challenge data should not be made public.

---

## 11. How to Run

Run the commands from the root of the `NEXORA-2026` repository.

The solution uses the challenge data from the top-level `data/` directory.

### Step 1 — Data check

```bash
python step1_data_check.py --data .\data
```

### Step 2 — Exploratory analysis

```bash
python step2_eda.py --data .\data
```

### Step 3 — Target definition and cost analysis

```bash
python step3_target_definition.py --data .\data
```

### Step 4 — Train and evaluate ML model

```bash
python step4_ml_model.py --data .\data
```

### Step 5 — Compare with 3-sigma baseline

```bash
python compare_baseline_cost.py --data .\data
```

### Step 6 — Validate predictions

```bash
python validate_submission.py .\predictions.csv
```

The final prediction file must pass the provided validator before hand-in.

---

## 12. Limitations

The solution predicts the probability of a future meter-read problem. It does **not** guarantee that a selected gateway has a physical fault.

Important limitations include:

* Meter-read success is a proxy rather than perfect physical-fault ground truth.
* The 15-visit limit forces prioritization.
* Future conditions may differ from historical conditions.
* Network, hardware, firmware, antenna, or configuration changes can reduce model accuracy.
* The model cannot diagnose the exact physical cause of a problem.
* A gateway below rank 15 may still have a genuine problem but cannot be selected because of the weekly visit limit.

See `What-it-cannot-do.md` for the detailed limitations and proposed two-week improvements.

---

## 13. Key Takeaway

The solution turns historical gateway data into a **cost-aware weekly field-prioritization system**.

Instead of attempting to guarantee which gateways will fail, it answers the operational question:

> **Which 15 gateways should the field team prioritize this week?**

The final Random Forest model achieved a **23.0% lower evaluated cost than the supplied 3-sigma baseline**, while keeping the 15-gateway weekly limit and using information available before each prediction week.

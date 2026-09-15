import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# STEP 3 — TARGET DEFINITION AND COST-BASED DECISION
# NEXORA 2026
# ============================================================

print("=" * 70)
print("STEP 3 — TARGET DEFINITION AND COST-BASED DECISION")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

METER_FILE = "meter_read_success.csv"
VISIT_FILE = "field_visits.csv"

meter = pd.read_csv(METER_FILE, encoding="latin1")
visits = pd.read_csv(VISIT_FILE, encoding="latin1")

print("\nLoaded files:")
print(f"Meter-read success: {meter.shape}")
print(f"Field visits:      {visits.shape}")


# ------------------------------------------------------------
# 2. NORMALIZE GATEWAY IDS
# ------------------------------------------------------------

def normalize_gateway_id(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(":", "", regex=False)
        .str.replace("-", "", regex=False)
    )


meter["gateway_norm"] = normalize_gateway_id(meter["gateway_id"])
visits["gateway_norm"] = normalize_gateway_id(visits["gateway_id"])


# ------------------------------------------------------------
# 3. PREPARE METER-READ DATA
# ------------------------------------------------------------

meter["week_start"] = pd.to_datetime(
    meter["week_start"],
    errors="coerce"
)

meter["meters_expected"] = pd.to_numeric(
    meter["meters_expected"],
    errors="coerce"
)

meter["meters_read"] = pd.to_numeric(
    meter["meters_read"],
    errors="coerce"
)

meter["read_success"] = np.where(
    meter["meters_expected"] > 0,
    meter["meters_read"] / meter["meters_expected"],
    np.nan
)

meter = meter.sort_values(
    ["gateway_norm", "week_start"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 4. CREATE FUTURE TARGET
# ------------------------------------------------------------
# IMPORTANT:
# We only use the NEXT week's result as the target.
#
# At prediction time, the next week's result is unknown.
# Therefore it is a valid future outcome and does not leak
# into the features.
# ------------------------------------------------------------

meter["next_week_success"] = (
    meter.groupby("gateway_norm")["read_success"]
    .shift(-1)
)

meter["change_next_week"] = (
    meter["next_week_success"] -
    meter["read_success"]
)

model_eval = meter[
    meter["next_week_success"].notna()
].copy()

print("\nTemporal target dataset:")
print(f"Rows with a known following week: {len(model_eval)}")
print(
    f"Gateways: {model_eval['gateway_norm'].nunique()}"
)
print(
    f"Period: {model_eval['week_start'].min().date()} "
    f"to {model_eval['week_start'].max().date()}"
)


# ------------------------------------------------------------
# 5. TEST DIFFERENT DEFINITIONS OF "NEEDS A VISIT"
# ------------------------------------------------------------

target_thresholds = [0.70, 0.80, 0.90]

definition_results = []

for threshold in target_thresholds:

    target = (
        model_eval["next_week_success"] < threshold
    )

    definition_results.append({
        "definition": f"Next-week read success < {threshold:.0%}",
        "threshold": threshold,
        "problem_weeks": int(target.sum()),
        "total_weeks": len(model_eval),
        "problem_rate": target.mean()
    })

definition_df = pd.DataFrame(definition_results)

print("\n" + "=" * 70)
print("A. CANDIDATE TARGET DEFINITIONS")
print("=" * 70)

print(
    definition_df.to_string(
        index=False,
        formatters={
            "problem_rate": "{:.2%}".format
        }
    )
)


# ------------------------------------------------------------
# 6. HISTORICAL FIELD VISITS — WHY THIS IS NOT THE DIRECT TARGET
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("B. HISTORICAL FIELD VISITS")
print("=" * 70)

visit_outcomes = (
    visits["outcome"]
    .value_counts()
)

print("\nVisit outcomes:")
print(visit_outcomes.to_string())

print(
    "\nHistorical visits are NOT treated as the ground-truth "
    "target because many visits ended without a fault being found."
)

no_error_count = (
    visits["outcome"]
    .eq("Kein Fehler gefunden")
    .sum()
)

no_error_rate = no_error_count / len(visits)

print(
    f"\nVisits ending with 'Kein Fehler gefunden': "
    f"{no_error_count}/{len(visits)} "
    f"({no_error_rate:.2%})"
)


# ------------------------------------------------------------
# 7. CURRENT READ SUCCESS AS A SIMPLE EARLY-WARNING SIGNAL
# ------------------------------------------------------------
# We test whether today's/current week's read success
# gives useful information about next week's deterioration.
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("C. CURRENT READ SUCCESS VS NEXT-WEEK PROBLEM")
print("=" * 70)

PRIMARY_TARGET = 0.80

model_eval["next_week_problem"] = (
    model_eval["next_week_success"] < PRIMARY_TARGET
)


current_bands = pd.cut(
    model_eval["read_success"],
    bins=[-np.inf, 0.50, 0.70, 0.80, 0.90, 1.00, np.inf],
    labels=[
        "<50%",
        "50–70%",
        "70–80%",
        "80–90%",
        "90–100%",
        ">100%"
    ],
    include_lowest=True
)

band_summary = (
    model_eval
    .groupby(
        current_bands,
        observed=False
    )
    .agg(
        gateways=("gateway_norm", "nunique"),
        observations=("next_week_problem", "size"),
        next_week_problem_rate=("next_week_problem", "mean"),
        current_read_success=("read_success", "mean")
    )
    .reset_index()
)

print(
    band_summary.to_string(
        index=False,
        formatters={
            "next_week_problem_rate": "{:.2%}".format,
            "current_read_success": "{:.2%}".format
        }
    )
)


# ------------------------------------------------------------
# 8. COST MODEL
# ------------------------------------------------------------

UNNECESSARY_VISIT_COST = 380
MISSED_PROBLEM_COST = 600

print("\n" + "=" * 70)
print("D. COST MODEL")
print("=" * 70)

print(
    f"Unnecessary visit cost: €{UNNECESSARY_VISIT_COST}"
)

print(
    f"Missed problem cost for one week: "
    f"€{MISSED_PROBLEM_COST}"
)

# If probability of a problem is p:
#
# Expected cost of NOT visiting = p * 600
# Expected cost of visiting = (1-p) * 380
#
# Visit when:
# p * 600 > (1-p) * 380
#
# p > 380 / (600 + 380)
#
# This is the probability at which the two actions have
# equal expected cost.

decision_probability_threshold = (
    UNNECESSARY_VISIT_COST /
    (UNNECESSARY_VISIT_COST + MISSED_PROBLEM_COST)
)

print(
    f"\nCost-based probability break-even: "
    f"{decision_probability_threshold:.2%}"
)

print(
    "Interpretation: if the estimated probability of a "
    "next-week problem is above this level, visiting has "
    "lower expected cost than leaving the gateway alone."
)


# ------------------------------------------------------------
# 9. COST OF SIMPLE CURRENT-READ-SUCCESS RULES
# ------------------------------------------------------------

action_thresholds = [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

cost_results = []

for threshold in action_thresholds:

    actual = model_eval["next_week_problem"]

    # Simple rule:
    # visit if current read success is below threshold
    predicted_visit = (
        model_eval["read_success"] < threshold
    )

    false_alarm = (
        predicted_visit & ~actual
    )

    missed_problem = (
        ~predicted_visit & actual
    )

    correct_visit = (
        predicted_visit & actual
    )

    correct_no_visit = (
        ~predicted_visit & ~actual
    )

    unnecessary_visits = int(false_alarm.sum())
    missed = int(missed_problem.sum())

    total_cost = (
        unnecessary_visits * UNNECESSARY_VISIT_COST
        +
        missed * MISSED_PROBLEM_COST
    )

    cost_results.append({
        "current_read_success_threshold": threshold,
        "visits_selected": int(predicted_visit.sum()),
        "problem_cases": int(actual.sum()),
        "correct_visits": int(correct_visit.sum()),
        "unnecessary_visits": unnecessary_visits,
        "missed_problems": missed,
        "total_cost_eur": total_cost,
        "average_cost_per_week_gateway": (
            total_cost / len(model_eval)
        )
    })

cost_df = pd.DataFrame(cost_results)

print("\nSimple threshold cost comparison:")

print(
    cost_df.to_string(
        index=False,
        formatters={
            "current_read_success_threshold":
                "{:.0%}".format,
            "average_cost_per_week_gateway":
                "€{:.2f}".format
        }
    )
)


# ------------------------------------------------------------
# 10. BEST SIMPLE THRESHOLD
# ------------------------------------------------------------

best_row = cost_df.loc[
    cost_df["total_cost_eur"].idxmin()
]

print("\n" + "=" * 70)
print("E. BEST SIMPLE THRESHOLD ON HISTORICAL DATA")
print("=" * 70)

print(
    f"Lowest-cost current read-success threshold: "
    f"{best_row['current_read_success_threshold']:.0%}"
)

print(
    f"Gateways/weeks selected for visit: "
    f"{int(best_row['visits_selected'])}"
)

print(
    f"Unnecessary visits: "
    f"{int(best_row['unnecessary_visits'])}"
)

print(
    f"Missed problems: "
    f"{int(best_row['missed_problems'])}"
)

print(
    f"Estimated total cost: "
    f"€{best_row['total_cost_eur']:,.0f}"
)


# ------------------------------------------------------------
# 11. TEMPORAL STABILITY CHECK
# ------------------------------------------------------------
# Split the historical period into earlier and later data.
# This checks whether the relationship is stable over time.
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("F. TEMPORAL STABILITY CHECK")
print("=" * 70)

date_cut = model_eval["week_start"].quantile(0.70)

early = model_eval[
    model_eval["week_start"] <= date_cut
].copy()

late = model_eval[
    model_eval["week_start"] > date_cut
].copy()

print(
    f"Early period: {early['week_start'].min().date()} "
    f"to {early['week_start'].max().date()}"
)

print(
    f"Later period: {late['week_start'].min().date()} "
    f"to {late['week_start'].max().date()}"
)

for name, subset in [
    ("Early", early),
    ("Later", late)
]:

    rate = subset["next_week_problem"].mean()

    print(
        f"{name} next-week problem rate: {rate:.2%}"
    )


# ------------------------------------------------------------
# 12. GATEWAY-LEVEL SENSITIVITY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("G. GATEWAY-LEVEL SENSITIVITY")
print("=" * 70)

gateway_target = (
    model_eval
    .groupby("gateway_norm")
    .agg(
        weeks_observed=("next_week_problem", "size"),
        problem_rate=("next_week_problem", "mean"),
        avg_current_success=("read_success", "mean")
    )
    .reset_index()
)

print(
    f"Gateways with future target observations: "
    f"{len(gateway_target)}"
)

print(
    "\nProblem-rate distribution across gateways:"
)

print(
    gateway_target["problem_rate"]
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# 13. SAVE RESULTS
# ------------------------------------------------------------

definition_df.to_csv(
    "step3_target_definitions.csv",
    index=False
)

band_summary.to_csv(
    "step3_read_success_bands.csv",
    index=False
)

cost_df.to_csv(
    "step3_cost_analysis.csv",
    index=False
)

gateway_target.to_csv(
    "step3_gateway_sensitivity.csv",
    index=False
)


# ------------------------------------------------------------
# 14. CHART 1 — NEXT-WEEK PROBLEM RATE BY CURRENT SUCCESS
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plot_df = band_summary[
    band_summary["current_read_success"].notna()
].copy()

plt.bar(
    plot_df["read_success"].astype(str),
    plot_df["next_week_problem_rate"] * 100
)

plt.xlabel("Current-week meter-read success")
plt.ylabel("Next-week problem rate (%)")
plt.title(
    "Next-week operational risk increases as current "
    "meter-read success deteriorates"
)

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "step3_next_week_risk_by_read_success.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# 15. CHART 2 — COST BY CURRENT SUCCESS THRESHOLD
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    cost_df["current_read_success_threshold"] * 100,
    cost_df["total_cost_eur"],
    marker="o"
)

plt.xlabel(
    "Current-week read-success threshold for recommending visit (%)"
)

plt.ylabel("Estimated total cost (€)")

plt.title(
    "Cost sensitivity to the visit decision threshold"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "step3_cost_vs_threshold.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# 16. CHART 3 — TARGET DEFINITION SENSITIVITY
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    definition_df["threshold"].astype(str),
    definition_df["problem_rate"] * 100
)

plt.xlabel("Definition threshold")
plt.ylabel("Next-week problem rate (%)")

plt.title(
    "Sensitivity of 'needs a visit' definition"
)

plt.tight_layout()

plt.savefig(
    "step3_target_definition_sensitivity.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# 17. FINAL CONCLUSION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 CONCLUSION")
print("=" * 70)

print(
    "\nPrimary target definition:"
)

print(
    "A gateway is considered to need a visit when its "
    "next-week meter-read success falls below 80%."
)

print(
    "\nWhy this definition:"
)

print(
    "• It represents future operational deterioration."
)

print(
    "• It can be evaluated using historical meter-read outcomes."
)

print(
    "• It avoids treating every historical field visit as a fault."
)

print(
    "• It creates a future-week target that ML can predict."
)

print(
    "\nImportant limitation:"
)

print(
    "Meter-read failure is a proxy for operational need, not "
    "a perfect ground-truth label for a physical gateway fault."
)

print(
    "Historical field visits contain routine checks and visits "
    "where no error was found, so they are better treated as "
    "supporting evidence rather than the target itself."
)

print(
    "\nSensitivity:"
)

print(
    "The analysis tests 70%, 80%, and 90% as possible definitions "
    "of a meaningful deterioration. The final report should "
    "present this range rather than claiming that one threshold "
    "is universally correct."
)

print(
    "\nCost decision:"
)

print(
    f"With €380 for an unnecessary visit and €600 for one week "
    f"of missed problems, the probability break-even is "
    f"{decision_probability_threshold:.2%}."
)

print(
    "\nFiles created:"
)

for filename in [
    "step3_target_definitions.csv",
    "step3_read_success_bands.csv",
    "step3_cost_analysis.csv",
    "step3_gateway_sensitivity.csv",
    "step3_next_week_risk_by_read_success.png",
    "step3_cost_vs_threshold.png",
    "step3_target_definition_sensitivity.png"
]:
    print(f"  ✓ {filename}")

print("\n" + "=" * 70)
print("STEP 3 COMPLETE")
print("=" * 70)
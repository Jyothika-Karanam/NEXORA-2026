
import os
import numpy as np
import pandas as pd

# ============================================================
# NEXORA 2026
# Historical 3-Sigma Baseline vs ML
#
# IMPORTANT:
# The ML model predicts NEXT-WEEK read-success failure.
# Therefore, this script also evaluates the 3-sigma baseline
# against the NEXT week's actual outcome.
# ============================================================

DATA_DIR = "."

METER_FILE = os.path.join(DATA_DIR, "meter_read_success.csv")
TELEMETRY_DIR = os.path.join(DATA_DIR, "telemetry")

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "baseline_historical_cost_evaluation.csv"
)

# Same six historical test weeks used by the ML model.
# Baseline is scored using information available BEFORE each week.
TEST_WEEKS = pd.to_datetime([
    "2025-12-15",
    "2025-12-22",
    "2025-12-29",
    "2026-01-05",
    "2026-01-12",
    "2026-01-19",
])

PROBLEM_THRESHOLD = 0.80

UNNECESSARY_VISIT_COST = 380
MISSED_PROBLEM_COST = 600

METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
]


# ============================================================
# Helpers
# ============================================================

def normalize_gateway_id(series):
    """
    Normalize gateway IDs so telemetry and meter-read files
    use the same key format.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(":", "", regex=False)
        .str.replace("-", "", regex=False)
    )



def load_telemetry():
    """
    Load all telemetry parquet partitions.
    """
    files = []

    for root, _, filenames in os.walk(TELEMETRY_DIR):
        for filename in filenames:
            if filename.endswith(".parquet"):
                files.append(os.path.join(root, filename))

    files = sorted(files)

    if not files:
        raise FileNotFoundError(
            f"No parquet telemetry files found in {TELEMETRY_DIR}"
        )

    frames = []

    for file in files:
        df = pd.read_parquet(file)
        frames.append(df)

    telemetry = pd.concat(frames, ignore_index=True)

    telemetry["gateway_id"] = normalize_gateway_id(
        telemetry["gateway_id"]
    )

    telemetry["ts_utc"] = pd.to_datetime(
        telemetry["ts_utc"],
        utc=True,
        errors="coerce"
    )

    telemetry = telemetry.dropna(
        subset=["gateway_id", "ts_utc"]
    )

    return telemetry


def calculate_week_start(telemetry):
    """
    Convert UTC timestamps to Europe/Berlin local time and
    assign each row to the Monday of its local calendar week.
    """

    local_time = telemetry["ts_utc"].dt.tz_convert(
        "Europe/Berlin"
    )

    local_date = local_time.dt.normalize()

    telemetry["week_start"] = (
        local_date
        - pd.to_timedelta(
            local_date.dt.weekday,
            unit="D"
        )
    )

    telemetry["week_start"] = (
        telemetry["week_start"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    return telemetry


# ============================================================
# 3-Sigma scoring
# ============================================================

def score_baseline_for_week(telemetry, score_week):
    """
    Reproduce the provided baseline logic.

    For each gateway:
      - use the trailing 28 days strictly before score_week
      - calculate gateway mean/std for each metric
      - look at the most recent 7 days before score_week
      - count hourly observations above mean + 3*std
      - rank gateways by flagged-hour count
      - select top 15
    """

    score_week = pd.Timestamp(score_week)

    window_start = score_week - pd.Timedelta(days=28)
    recent_start = score_week - pd.Timedelta(days=7)

    history = telemetry[
        (telemetry["ts_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
         >= window_start)
        &
        (telemetry["ts_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
         < score_week)
    ].copy()

    recent = telemetry[
        (telemetry["ts_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
         >= recent_start)
        &
        (telemetry["ts_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
         < score_week)
    ].copy()

    if history.empty or recent.empty:
        return pd.DataFrame(
            columns=[
                "gateway_id",
                "flagged_hours"
            ]
        )

    # --------------------------------------------------------
    # Gateway-specific historical mean/std
    # --------------------------------------------------------

    stats = (
        history
        .groupby("gateway_id")[METRICS]
        .agg(["mean", "std"])
    )

    flagged = pd.DataFrame(
        index=recent.index
    )

    flagged["gateway_id"] = recent["gateway_id"].values

    total_flags = np.zeros(len(recent), dtype=int)

    for metric in METRICS:

        mean_values = recent["gateway_id"].map(
            stats[(metric, "mean")]
        )

        std_values = recent["gateway_id"].map(
            stats[(metric, "std")]
        )

        # If std is missing, treat it as zero.
        std_values = std_values.fillna(0)

        threshold = mean_values + 3 * std_values

        metric_values = recent[metric]

        metric_flags = (
            metric_values > threshold
        ).fillna(False)

        total_flags += metric_flags.astype(int).to_numpy()

    flagged["flagged_hours"] = total_flags

    scores = (
        flagged
        .groupby("gateway_id")["flagged_hours"]
        .sum()
        .reset_index()
        .sort_values(
            ["flagged_hours", "gateway_id"],
            ascending=[False, True]
        )
    )

    return scores


# ============================================================
# Actual next-week outcome
# ============================================================

def load_meter_data():
    meter = pd.read_csv(
        METER_FILE,
        encoding="latin1"
    )

    meter["gateway_id"] = normalize_gateway_id(
        meter["gateway_id"]
    )

    meter["week_start"] = pd.to_datetime(
        meter["week_start"],
        errors="coerce"
    ).dt.normalize()

    meter["read_success"] = (
        meter["meters_read"]
        / meter["meters_expected"]
    )

    meter["read_success"] = meter["read_success"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    meter["problem"] = (
        meter["read_success"] < PROBLEM_THRESHOLD
    )

    return meter


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 70)
    print("NEXORA 2026 - FAIR 3-SIGMA BASELINE COMPARISON")
    print("=" * 70)

    print("\nLoading meter-read data...")
    meter = load_meter_data()

    print(
        f"Meter rows: {len(meter)}"
    )

    print("\nLoading telemetry...")
    telemetry = load_telemetry()
    telemetry = calculate_week_start(telemetry)

    print(
        f"Telemetry rows: {len(telemetry)}"
    )

    results = []

    for score_week in TEST_WEEKS:

        # ----------------------------------------------------
        # IMPORTANT:
        # Baseline ranks gateways using data available before
        # score_week.
        #
        # Actual outcome is measured in score_week + 7 days.
        # This matches the ML target.
        # ----------------------------------------------------

        actual_week = (
            score_week + pd.Timedelta(days=7)
        )

        scores = score_baseline_for_week(
            telemetry,
            score_week
        )

        if scores.empty:
            print(
                f"{score_week.date()} | "
                "No baseline scores available"
            )
            continue

        selected = scores.head(15).copy()

        selected_ids = set(
            selected["gateway_id"]
        )

        actual = meter[
            meter["week_start"] == actual_week
        ].copy()

        actual_problem_ids = set(
            actual.loc[
                actual["problem"],
                "gateway_id"
            ]
        )

        # ----------------------------------------------------
        # Cost calculation
        # ----------------------------------------------------

        unnecessary = len(
            selected_ids - actual_problem_ids
        )

        missed = len(
            actual_problem_ids - selected_ids
        )

        cost = (
            unnecessary * UNNECESSARY_VISIT_COST
            +
            missed * MISSED_PROBLEM_COST
        )

        results.append({
            "score_week": score_week.date(),
            "actual_week": actual_week.date(),
            "selected": len(selected_ids),
            "actual_problems": len(actual_problem_ids),
            "unnecessary_visits": unnecessary,
            "missed_problems": missed,
            "cost_eur": cost,
        })

        print(
            f"{score_week.date()} -> "
            f"{actual_week.date()} | "
            f"selected={len(selected_ids)} | "
            f"actual={len(actual_problem_ids)} | "
            f"unnecessary={unnecessary} | "
            f"missed={missed} | "
            f"cost=€{cost:,}"
        )

    # ========================================================
    # Summary
    # ========================================================

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("\nNo evaluation results were generated.")
        return

    total_unnecessary = int(
        results_df["unnecessary_visits"].sum()
    )

    total_missed = int(
        results_df["missed_problems"].sum()
    )

    total_cost = int(
        results_df["cost_eur"].sum()
    )

    ML_COST = 111600

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Baseline unnecessary visits: "
        f"{total_unnecessary}"
    )

    print(
        f"Baseline missed problems: "
        f"{total_missed}"
    )

    print(
        f"Total 3-sigma baseline cost: "
        f"€{total_cost:,}"
    )

    print(
        f"V3 ML validation cost: "
        f"€{ML_COST:,}"
    )

    difference = total_cost - ML_COST

    print(
        f"ML cost advantage: "
        f"€{difference:,}"
    )

    if total_cost > ML_COST:

        percentage = (
            difference / total_cost
        ) * 100

        print(
            f"ML improvement vs baseline: "
            f"{percentage:.1f}%"
        )

        print(
            "\nRESULT: ML BEATS THE 3-SIGMA BASELINE"
        )

    elif total_cost < ML_COST:

        percentage = (
            abs(difference) / total_cost
        ) * 100

        print(
            f"Baseline improvement vs ML: "
            f"{percentage:.1f}%"
        )

        print(
            "\nRESULT: 3-SIGMA BASELINE BEATS ML"
        )

    else:

        print(
            "\nRESULT: ML AND BASELINE HAVE THE SAME COST"
        )

    # ========================================================
    # Save results
    # ========================================================

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()


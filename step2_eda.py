
import pandas as pd
import numpy as np
import os
import argparse

print("=" * 70)
print("NEXORA 2026 - STEP 2 EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# COMMAND-LINE DATA PATH
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    default="data",
    help="Path to the challenge data directory"
)
args = parser.parse_args()

DATA_DIR = args.data


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("\n1. LOADING DATA")
print("-" * 70)

gateway_master = pd.read_csv(
    os.path.join(DATA_DIR, "gateway_master.csv"),
    encoding="latin1"
)

field_visits = pd.read_csv(
    os.path.join(DATA_DIR, "field_visits.csv"),
    encoding="latin1"
)

meter_read_success = pd.read_csv(
    os.path.join(DATA_DIR, "meter_read_success.csv"),
    encoding="latin1"
)

engineer_review = pd.read_excel(
    os.path.join(DATA_DIR, "engineer_review_2026-02.xlsx")
)

print("\n--- GATEWAY ID DIAGNOSTIC ---")

print("Gateway master:")
print(gateway_master["gateway_id"].head(10).tolist())
print(gateway_master["gateway_id"].dtype)

print("\nField visits:")
print(field_visits["gateway_id"].head(10).tolist())
print(field_visits["gateway_id"].dtype)

print("\nMeter read success:")
print(meter_read_success["gateway_id"].head(10).tolist())
print(meter_read_success["gateway_id"].dtype)

print("\nEngineer review:")
print(engineer_review["gateway_id"].head(10).tolist())
print(engineer_review["gateway_id"].dtype)

print("\nCommon IDs:")
print(
    len(
        set(gateway_master["gateway_id"])
        & set(meter_read_success["gateway_id"])
    )
)

print(
    "Master ∩ Visits:",
    len(
        set(gateway_master["gateway_id"])
        & set(field_visits["gateway_id"])
    )
)

# Standardize gateway IDs before any merge
def normalize_gateway_id(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(":", "", regex=False)
        .str.replace("-", "", regex=False)
    )

gateway_master["gateway_id"] = normalize_gateway_id(
    gateway_master["gateway_id"]
)

field_visits["gateway_id"] = normalize_gateway_id(
    field_visits["gateway_id"]
)

meter_read_success["gateway_id"] = normalize_gateway_id(
    meter_read_success["gateway_id"]
)

engineer_review["gateway_id"] = normalize_gateway_id(
    engineer_review["gateway_id"]
)

# Convert dates
field_visits["requested_on"] = pd.to_datetime(
    field_visits["requested_on"], errors="coerce"
)

field_visits["visited_on"] = pd.to_datetime(
    field_visits["visited_on"], errors="coerce"
)

meter_read_success["week_start"] = pd.to_datetime(
    meter_read_success["week_start"], errors="coerce"
)

engineer_review["reviewed_on"] = pd.to_datetime(
    engineer_review["reviewed_on"], errors="coerce"
)

print(f"Gateway master: {gateway_master.shape}")
print(f"Field visits: {field_visits.shape}")
print(f"Meter read success: {meter_read_success.shape}")
print(f"Engineer review: {engineer_review.shape}")


# ---------------------------------------------------------
# 2. METER READ SUCCESS ANALYSIS
# ---------------------------------------------------------
print("\n2. METER READ SUCCESS ANALYSIS")
print("-" * 70)

meter_read_success["read_success_rate"] = np.where(
    meter_read_success["meters_expected"] > 0,
    meter_read_success["meters_read"] /
    meter_read_success["meters_expected"],
    np.nan
)

print(
    "Overall mean read success:",
    round(meter_read_success["read_success_rate"].mean(), 4)
)

print(
    "Median read success:",
    round(meter_read_success["read_success_rate"].median(), 4)
)

print("\nRead success distribution:")

bins = [-0.01, 0.50, 0.70, 0.80, 0.90, 1.01]

labels = [
    "<50%",
    "50-70%",
    "70-80%",
    "80-90%",
    "90-100%"
]

meter_read_success["success_band"] = pd.cut(
    meter_read_success["read_success_rate"],
    bins=bins,
    labels=labels
)

print(
    meter_read_success["success_band"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# 3. FIELD VISIT ANALYSIS
# ---------------------------------------------------------
print("\n3. FIELD VISIT ANALYSIS")
print("-" * 70)

print("Total visits:", len(field_visits))
print("Unique gateways visited:", field_visits["gateway_id"].nunique())

print("\nVisit outcomes:")
print(field_visits["outcome"].value_counts())

print("\nVisit reasons:")
print(field_visits["reason_reported"].value_counts())

# Count visits per gateway
visit_counts = (
    field_visits.groupby("gateway_id")
    .size()
    .reset_index(name="visit_count")
)

print("\nVisit count per gateway:")
print(visit_counts["visit_count"].describe().round(2))


# ---------------------------------------------------------
# 4. VISIT OUTCOME QUALITY
# ---------------------------------------------------------
print("\n4. VISIT OUTCOME ANALYSIS")
print("-" * 70)

outcome_summary = (
    field_visits.groupby("outcome")
    .agg(
        visits=("visit_id", "count"),
        avg_technician_hours=("technician_hours", "mean")
    )
    .reset_index()
)

outcome_summary["avg_technician_hours"] = (
    outcome_summary["avg_technician_hours"].round(2)
)

print(outcome_summary.to_string(index=False))


# ---------------------------------------------------------
# 5. METER SUCCESS VS VISITS
# ---------------------------------------------------------
print("\n5. METER SUCCESS VS FIELD VISITS")
print("-" * 70)

gateway_meter = (
    meter_read_success.groupby("gateway_id")
    .agg(
        avg_read_success=("read_success_rate", "mean"),
        min_read_success=("read_success_rate", "min"),
        weeks_observed=("week_start", "nunique")
    )
    .reset_index()
)

gateway_analysis = gateway_master[["gateway_id"]].merge(
    gateway_meter,
    on="gateway_id",
    how="left"
)

gateway_analysis = gateway_analysis.merge(
    visit_counts,
    on="gateway_id",
    how="left"
)

gateway_analysis["visit_count"] = (
    gateway_analysis["visit_count"].fillna(0)
)

print("\nGateway-level read success vs visit count:")

print(
    gateway_analysis[
        ["avg_read_success", "visit_count"]
    ].describe().round(3)
)

# Create broad success groups
gateway_analysis["success_group"] = pd.cut(
    gateway_analysis["avg_read_success"],
    bins=[-0.01, 0.70, 0.80, 0.90, 1.01],
    labels=["<70%", "70-80%", "80-90%", "90-100%"]
)

success_visit_summary = (
    gateway_analysis
    .groupby("success_group", observed=False)
    .agg(
        gateways=("gateway_id", "count"),
        avg_visits=("visit_count", "mean"),
        median_visits=("visit_count", "median")
    )
    .reset_index()
)

print("\nVisit frequency by average meter-read success:")
print(success_visit_summary.to_string(index=False))


# ---------------------------------------------------------
# 6. ENGINEER REVIEW ANALYSIS
# ---------------------------------------------------------
print("\n6. ENGINEER REVIEW ANALYSIS")
print("-" * 70)

print("Engineer review categories:")
print(engineer_review["Kategorie"].value_counts())

review_counts = (
    engineer_review
    .groupby("Kategorie")["gateway_id"]
    .nunique()
)

print("\nUnique gateways by engineer category:")
print(review_counts)

# Merge review with gateway analysis
review_analysis = gateway_analysis.merge(
    engineer_review[["gateway_id", "Kategorie"]],
    on="gateway_id",
    how="left"
)

print("\nMeter-read success by engineer category:")

review_summary = (
    review_analysis
    .dropna(subset=["Kategorie"])
    .groupby("Kategorie")
    .agg(
        gateways=("gateway_id", "count"),
        avg_read_success=("avg_read_success", "mean"),
        median_read_success=("avg_read_success", "median"),
        avg_visits=("visit_count", "mean")
    )
    .reset_index()
)

review_summary = review_summary.round(3)

print(review_summary.to_string(index=False))


# ---------------------------------------------------------
# 7. VISIT SUCCESS BY REASON
# ---------------------------------------------------------
print("\n7. VISIT REASON VS OUTCOME")
print("-" * 70)

reason_outcome = pd.crosstab(
    field_visits["reason_reported"],
    field_visits["outcome"]
)

print(reason_outcome.to_string())


# ---------------------------------------------------------
# 8. TELEMETRY SAMPLE ANALYSIS
# ---------------------------------------------------------
print("\n8. TELEMETRY ANALYSIS")
print("-" * 70)

telemetry_path = os.path.join(
    DATA_DIR,
    "telemetry",
    "month=2026-01",
    "part-0.parquet"
)

print("Loading:")
print(telemetry_path)

telemetry = pd.read_parquet(telemetry_path)

# Standardize telemetry gateway IDs too
telemetry["gateway_id"] = normalize_gateway_id(
    telemetry["gateway_id"]
)

print("Telemetry shape:", telemetry.shape)

# Important numeric columns
telemetry_features = [
    "rx_nr_pkts",
    "rx_crc_bad",
    "tx_success",
    "tx_busy",
    "avg_load1",
    "avg_memfree",
    "avg_uptime",
    "reboot_cnt",
    "offline_duration_sec",
    "online_duration_mins",
    "no_conn_importance",
    "reboot_importance",
    "rssi_good",
    "rssi_normal",
    "rssi_bad",
    "rscp_rsrp_good",
    "rscp_rsrp_normal",
    "rscp_rsrp_bad",
    "ecio_rsrq_good",
    "ecio_rsrq_normal",
    "ecio_rsrq_bad"
]

available_features = [
    col for col in telemetry_features
    if col in telemetry.columns
]

telemetry_summary = telemetry[available_features].describe().T

print("\nTelemetry feature summary:")
print(
    telemetry_summary[
        ["mean", "std", "min", "50%", "max"]
    ].round(3).to_string()
)


# ---------------------------------------------------------
# 9. TELEMETRY GATEWAY-LEVEL AGGREGATION
# ---------------------------------------------------------
print("\n9. TELEMETRY GATEWAY-LEVEL SIGNALS")
print("-" * 70)

telemetry_gateway = (
    telemetry.groupby("gateway_id")
    .agg(
        avg_reboots=("reboot_cnt", "mean"),
        total_reboots=("reboot_cnt", "sum"),
        avg_offline_seconds=("offline_duration_sec", "mean"),
        total_offline_seconds=("offline_duration_sec", "sum"),
        avg_online_minutes=("online_duration_mins", "mean"),
        avg_load=("avg_load1", "mean"),
        avg_memfree=("avg_memfree", "mean"),
        avg_uptime=("avg_uptime", "mean"),
        avg_reboot_importance=("reboot_importance", "mean"),
        avg_no_conn_importance=("no_conn_importance", "mean"),
        avg_rssi_bad=("rssi_bad", "mean"),
        avg_rscp_rsrp_bad=("rscp_rsrp_bad", "mean"),
        avg_ecio_rsrq_bad=("ecio_rsrq_bad", "mean")
    )
    .reset_index()
)

print("Telemetry gateways:", len(telemetry_gateway))

print("\nTelemetry gateway-level summary:")
print(
    telemetry_gateway.describe().round(3).to_string()
)


# ---------------------------------------------------------
# 10. TELEMETRY VS FIELD VISITS
# ---------------------------------------------------------
print("\n10. TELEMETRY VS FIELD VISITS")
print("-" * 70)

telemetry_visit = telemetry_gateway.merge(
    visit_counts,
    on="gateway_id",
    how="left"
)

telemetry_visit["visit_count"] = (
    telemetry_visit["visit_count"].fillna(0)
)

telemetry_visit["visited"] = (
    telemetry_visit["visit_count"] > 0
).astype(int)

print("\nAverage telemetry signals by visited/not visited:")

telemetry_comparison = (
    telemetry_visit
    .groupby("visited")
    .agg(
        gateways=("gateway_id", "count"),
        avg_reboots=("avg_reboots", "mean"),
        avg_offline_seconds=("avg_offline_seconds", "mean"),
        avg_load=("avg_load", "mean"),
        avg_memfree=("avg_memfree", "mean"),
        avg_reboot_importance=("avg_reboot_importance", "mean"),
        avg_no_conn_importance=("avg_no_conn_importance", "mean"),
        avg_rssi_bad=("avg_rssi_bad", "mean"),
        avg_rscp_rsrp_bad=("avg_rscp_rsrp_bad", "mean"),
        avg_ecio_rsrq_bad=("avg_ecio_rsrq_bad", "mean"),
        avg_visit_count=("visit_count", "mean")
    )
    .reset_index()
)

print(telemetry_comparison.round(3).to_string(index=False))


# ---------------------------------------------------------
# 11. SIMPLE CORRELATION CHECK
# ---------------------------------------------------------
print("\n11. SIMPLE CORRELATION CHECK")
print("-" * 70)

correlation_columns = [
    "avg_reboots",
    "avg_offline_seconds",
    "avg_online_minutes",
    "avg_load",
    "avg_memfree",
    "avg_uptime",
    "avg_reboot_importance",
    "avg_no_conn_importance",
    "avg_rssi_bad",
    "avg_rscp_rsrp_bad",
    "avg_ecio_rsrq_bad",
    "visit_count"
]

correlation_data = telemetry_visit[correlation_columns].corr(
    numeric_only=True
)

visit_correlations = (
    correlation_data["visit_count"]
    .drop("visit_count")
    .sort_values(key=abs, ascending=False)
)

print("\nCorrelation with field visit count:")
print(visit_correlations.round(3).to_string())


# ---------------------------------------------------------
# 12. KEY FINDINGS
# ---------------------------------------------------------
print("\n12. INITIAL FINDINGS")
print("-" * 70)

print("""
These are exploratory findings only.

We are NOT yet defining:
- the final DS target,
- the final visit threshold,
- the final ML model,
- or the final prediction scores.

The purpose of this step is to identify which signals appear useful
and which relationships need further testing.
""")

print("=" * 70)
print("STEP 2 EDA COMPLETED")
print("=" * 70)


import os
import argparse
import pandas as pd


# ---------------------------------------------------------
# DATA DIRECTORY
# ---------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    default="data",
    help="Path to the challenge data directory"
)
args = parser.parse_args()

DATA_DIR = args.data


print("=" * 60)
print("NEXORA 2026 - STEP 1 DATA CHECK")
print("=" * 60)


# ---------------------------------------------------------
# 1. CHECK FILES
# ---------------------------------------------------------

print("\n1. CHECKING FILES")

required_files = [
    "gateway_master.csv",
    "field_visits.csv",
    "meter_read_success.csv",
    "engineer_review_2026-02.xlsx",
    "telemetry"
]

for file_name in required_files:
    file_path = os.path.join(DATA_DIR, file_name)

    if os.path.exists(file_path):
        print(f"[OK] {file_name}")
    else:
        print(f"[MISSING] {file_name}")


# ---------------------------------------------------------
# 2. LOAD MAIN DATASETS
# ---------------------------------------------------------

print("\n2. LOADING DATASETS")

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

print(
    f"Gateway master: {gateway_master.shape[0]} rows, "
    f"{gateway_master.shape[1]} columns"
)

print(
    f"Field visits: {field_visits.shape[0]} rows, "
    f"{field_visits.shape[1]} columns"
)

print(
    f"Meter read success: {meter_read_success.shape[0]} rows, "
    f"{meter_read_success.shape[1]} columns"
)

print(
    f"Engineer review: {engineer_review.shape[0]} rows, "
    f"{engineer_review.shape[1]} columns"
)


# ---------------------------------------------------------
# 3. SHOW COLUMNS
# ---------------------------------------------------------

# Columns are checked internally but not printed individually.
# This keeps terminal output concise.


# ---------------------------------------------------------
# 4. GATEWAY COUNTS
# ---------------------------------------------------------

print("\n4. GATEWAY COUNTS")

print(
    "Gateway master:",
    gateway_master["gateway_id"].nunique()
)

print(
    "Field visit:",
    field_visits["gateway_id"].nunique()
)

print(
    "Meter read:",
    meter_read_success["gateway_id"].nunique()
)

print(
    "Engineer review:",
    engineer_review["gateway_id"].nunique()
)


# ---------------------------------------------------------
# 5. MISSING VALUES
# ---------------------------------------------------------

print("\n5. MISSING VALUE CHECK")

gateway_missing = gateway_master.isna().sum().sum()
field_missing = field_visits.isna().sum().sum()
meter_missing = meter_read_success.isna().sum().sum()
review_missing = engineer_review.isna().sum().sum()

print(f"Gateway master missing values: {gateway_missing}")
print(f"Field visits missing values: {field_missing}")
print(f"Meter read missing values: {meter_missing}")
print(f"Engineer review missing values: {review_missing}")


# ---------------------------------------------------------
# 6. DUPLICATES
# ---------------------------------------------------------

print("\n6. DUPLICATE CHECK")

gateway_duplicates = gateway_master.duplicated().sum()
field_duplicates = field_visits.duplicated().sum()
meter_duplicates = meter_read_success.duplicated().sum()
review_duplicates = engineer_review.duplicated().sum()

print(f"Gateway master: {gateway_duplicates}")
print(f"Field visits: {field_duplicates}")
print(f"Meter read: {meter_duplicates}")
print(f"Engineer review: {review_duplicates}")


# ---------------------------------------------------------
# 7. FIELD VISIT INFORMATION
# ---------------------------------------------------------

print("\n7. FIELD VISIT SUMMARY")

if "reason_reported" in field_visits.columns:
    visit_reason_count = field_visits["reason_reported"].nunique()
    print(f"Unique visit reasons: {visit_reason_count}")

if "outcome" in field_visits.columns:
    visit_outcome_count = field_visits["outcome"].nunique()
    print(f"Unique visit outcomes: {visit_outcome_count}")


# ---------------------------------------------------------
# 8. FIELD VISIT DATE RANGE
# ---------------------------------------------------------

print("\n8. FIELD VISIT DATES")

if "requested_on" in field_visits.columns:
    field_visits["requested_on"] = pd.to_datetime(
        field_visits["requested_on"],
        errors="coerce"
    )

    print(
        f"Requested: {field_visits['requested_on'].min().date()} "
        f"to {field_visits['requested_on'].max().date()}"
    )

if "visited_on" in field_visits.columns:
    field_visits["visited_on"] = pd.to_datetime(
        field_visits["visited_on"],
        errors="coerce"
    )

    print(
        f"Visited: {field_visits['visited_on'].min().date()} "
        f"to {field_visits['visited_on'].max().date()}"
    )


# ---------------------------------------------------------
# 9. METER READ SUCCESS
# ---------------------------------------------------------

print("\n9. METER READ SUCCESS")

meter_read_success["meters_expected"] = pd.to_numeric(
    meter_read_success["meters_expected"],
    errors="coerce"
)

meter_read_success["meters_read"] = pd.to_numeric(
    meter_read_success["meters_read"],
    errors="coerce"
)

meter_read_success["read_success_rate"] = (
    meter_read_success["meters_read"]
    / meter_read_success["meters_expected"].replace(0, pd.NA)
)

print(
    f"Average: "
    f"{meter_read_success['read_success_rate'].mean():.4f}"
)

print(
    f"Minimum: "
    f"{meter_read_success['read_success_rate'].min():.4f}"
)

print(
    f"Maximum: "
    f"{meter_read_success['read_success_rate'].max():.4f}"
)


# ---------------------------------------------------------
# 10. ENGINEER REVIEW
# ---------------------------------------------------------

print("\n10. ENGINEER REVIEW")

if "Kategorie" in engineer_review.columns:
    print(
        "Categories:",
        engineer_review["Kategorie"].nunique()
    )


# ---------------------------------------------------------
# 11. CHECK GATEWAY ID COVERAGE
# ---------------------------------------------------------

print("\n11. GATEWAY ID COVERAGE")

master_ids = set(gateway_master["gateway_id"].dropna())

visit_ids = set(field_visits["gateway_id"].dropna())
meter_ids = set(meter_read_success["gateway_id"].dropna())
review_ids = set(engineer_review["gateway_id"].dropna())

print(
    "Field visit IDs not in master:",
    len(visit_ids - master_ids)
)

print(
    "Meter-read IDs not in master:",
    len(meter_ids - master_ids)
)

print(
    "Engineer-review IDs not in master:",
    len(review_ids - master_ids)
)


# ---------------------------------------------------------
# 12. TELEMETRY CHECK
# ---------------------------------------------------------

print("\n12. TELEMETRY CHECK")

telemetry_path = os.path.join(
    DATA_DIR,
    "telemetry",
    "month=2026-01",
    "part-0.parquet"
)

if os.path.exists(telemetry_path):

    telemetry = pd.read_parquet(telemetry_path)

    print(
        f"January telemetry: "
        f"{telemetry.shape[0]} rows, "
        f"{telemetry.shape[1]} columns"
    )

    important_columns = [
        "gateway_id",
        "ts_utc",
        "DateDt",
        "hour",
        "rx_nr_pkts",
        "rx_crc_bad",
        "tx_success",
        "tx_busy",
        "avg_load1",
        "avg_memfree",
        "avg_uptime",
        "reboot_cnt",
        "offline_duration_sec",
        "online_duration_mins"
    ]

    missing_important = [
        column
        for column in important_columns
        if column not in telemetry.columns
    ]

    if missing_important:
        print(
            "Missing important telemetry columns:",
            len(missing_important)
        )
    else:
        print("Important telemetry columns: OK")

    if "DateDt" in telemetry.columns:
        dates = pd.to_datetime(
            telemetry["DateDt"],
            errors="coerce"
        )

        print(
            f"Telemetry dates: "
            f"{dates.min().date()} "
            f"to {dates.max().date()}"
        )

    print(
        "Telemetry unique gateways:",
        telemetry["gateway_id"].nunique()
    )

else:
    print("[MISSING] Telemetry January 2026 partition")


# ---------------------------------------------------------
# 13. FINAL MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1 DATA CHECK COMPLETED")
print("=" * 60)
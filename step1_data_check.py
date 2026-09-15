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


print("=" * 70)
print("NEXORA 2026 - STEP 1 DATA CHECK")
print("=" * 70)


# ---------------------------------------------------------
# 1. CHECK FILES
# ---------------------------------------------------------

print("\n1. CHECKING FILES")
print("-" * 70)

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

print("gateway_master:", gateway_master.shape)
print("field_visits:", field_visits.shape)
print("meter_read_success:", meter_read_success.shape)
print("engineer_review:", engineer_review.shape)


# ---------------------------------------------------------
# 3. SHOW COLUMNS
# ---------------------------------------------------------

print("\n3. COLUMNS")
print("-" * 70)

print("\nGateway Master:")
print(list(gateway_master.columns))

print("\nField Visits:")
print(list(field_visits.columns))

print("\nMeter Read Success:")
print(list(meter_read_success.columns))

print("\nEngineer Review:")
print(list(engineer_review.columns))


# ---------------------------------------------------------
# 4. GATEWAY COUNTS
# ---------------------------------------------------------

print("\n4. GATEWAY COUNTS")
print("-" * 70)

print(
    "Gateway master unique gateways:",
    gateway_master["gateway_id"].nunique()
)

print(
    "Field visit unique gateways:",
    field_visits["gateway_id"].nunique()
)

print(
    "Meter read unique gateways:",
    meter_read_success["gateway_id"].nunique()
)

print(
    "Engineer review unique gateways:",
    engineer_review["gateway_id"].nunique()
)


# ---------------------------------------------------------
# 5. MISSING VALUES
# ---------------------------------------------------------

print("\n5. MISSING VALUES")
print("-" * 70)

print("\nGateway Master missing values:")
print(gateway_master.isna().sum())

print("\nField Visits missing values:")
print(field_visits.isna().sum())

print("\nMeter Read Success missing values:")
print(meter_read_success.isna().sum())

print("\nEngineer Review missing values:")
print(engineer_review.isna().sum())


# ---------------------------------------------------------
# 6. DUPLICATES
# ---------------------------------------------------------

print("\n6. DUPLICATE CHECK")
print("-" * 70)

print(
    "Gateway master duplicate rows:",
    gateway_master.duplicated().sum()
)

print(
    "Field visit duplicate rows:",
    field_visits.duplicated().sum()
)

print(
    "Meter read duplicate rows:",
    meter_read_success.duplicated().sum()
)

print(
    "Engineer review duplicate rows:",
    engineer_review.duplicated().sum()
)


# ---------------------------------------------------------
# 7. FIELD VISIT INFORMATION
# ---------------------------------------------------------

print("\n7. FIELD VISIT INFORMATION")
print("-" * 70)

if "reason_reported" in field_visits.columns:
    print("\nVisit reasons:")
    print(field_visits["reason_reported"].value_counts(dropna=False))

if "outcome" in field_visits.columns:
    print("\nVisit outcomes:")
    print(field_visits["outcome"].value_counts(dropna=False))


# ---------------------------------------------------------
# 8. FIELD VISIT DATE RANGE
# ---------------------------------------------------------

print("\n8. FIELD VISIT DATES")
print("-" * 70)

if "requested_on" in field_visits.columns:
    field_visits["requested_on"] = pd.to_datetime(
        field_visits["requested_on"],
        errors="coerce"
    )

    print("Requested date:")
    print("Minimum:", field_visits["requested_on"].min())
    print("Maximum:", field_visits["requested_on"].max())

if "visited_on" in field_visits.columns:
    field_visits["visited_on"] = pd.to_datetime(
        field_visits["visited_on"],
        errors="coerce"
    )

    print("\nVisited date:")
    print("Minimum:", field_visits["visited_on"].min())
    print("Maximum:", field_visits["visited_on"].max())


# ---------------------------------------------------------
# 9. METER READ SUCCESS
# ---------------------------------------------------------

print("\n9. METER READ SUCCESS")
print("-" * 70)

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
    "Overall average read success rate:",
    meter_read_success["read_success_rate"].mean()
)

print(
    "Minimum read success rate:",
    meter_read_success["read_success_rate"].min()
)

print(
    "Maximum read success rate:",
    meter_read_success["read_success_rate"].max()
)


# ---------------------------------------------------------
# 10. ENGINEER REVIEW
# ---------------------------------------------------------

print("\n10. ENGINEER REVIEW")
print("-" * 70)

if "Kategorie" in engineer_review.columns:
    print("\nEngineer categories:")
    print(engineer_review["Kategorie"].value_counts(dropna=False))


# ---------------------------------------------------------
# 11. CHECK GATEWAY ID COVERAGE
# ---------------------------------------------------------

print("\n11. GATEWAY ID COVERAGE")
print("-" * 70)

master_ids = set(gateway_master["gateway_id"].dropna())

visit_ids = set(field_visits["gateway_id"].dropna())
meter_ids = set(meter_read_success["gateway_id"].dropna())
review_ids = set(engineer_review["gateway_id"].dropna())

print(
    "Field visit gateways not in gateway master:",
    len(visit_ids - master_ids)
)

print(
    "Meter-read gateways not in gateway master:",
    len(meter_ids - master_ids)
)

print(
    "Engineer-review gateways not in gateway master:",
    len(review_ids - master_ids)
)


# ---------------------------------------------------------
# 12. TELEMETRY CHECK
# ---------------------------------------------------------

print("\n12. TELEMETRY CHECK")
print("-" * 70)

telemetry_path = os.path.join(
    DATA_DIR,
    "telemetry",
    "month=2026-01",
    "part-0.parquet"
)

if os.path.exists(telemetry_path):

    print("Loading one telemetry month only:")
    print(telemetry_path)

    telemetry = pd.read_parquet(telemetry_path)

    print("Telemetry shape:", telemetry.shape)

    print("\nFirst telemetry columns:")
    print(list(telemetry.columns))

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

    print("\nImportant telemetry columns:")

    for column in important_columns:
        if column in telemetry.columns:
            print(f"[OK] {column}")
        else:
            print(f"[MISSING] {column}")

    print("\nTelemetry date range:")

    if "DateDt" in telemetry.columns:
        dates = pd.to_datetime(
            telemetry["DateDt"],
            errors="coerce"
        )

        print("Minimum:", dates.min())
        print("Maximum:", dates.max())

    print(
        "\nTelemetry unique gateways:",
        telemetry["gateway_id"].nunique()
    )

else:
    print("[MISSING] Telemetry January 2026 partition")


# ---------------------------------------------------------
# 13. FINAL MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 1 DATA CHECK COMPLETED")
print("=" * 70)

print(
    "\nIMPORTANT:"
    "\nThis step only checks and understands the available data."
    "\nWe are NOT defining the ML target yet."
    "\nWe are NOT changing predictions.csv."
)
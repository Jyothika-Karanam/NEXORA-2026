import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")




# ============================================================
# NEXORA 2026 - STEP 4 MACHINE LEARNING MODEL V3
# ============================================================

print("=" * 75)
print("NEXORA 2026 - STEP 4 MACHINE LEARNING MODEL V3")
print("=" * 75)

UNNECESSARY_VISIT_COST = 380
MISSED_PROBLEM_COST = 600
MAX_VISITS = 15

PREDICTION_WEEKS = pd.date_range(
    "2026-02-02",
    "2026-03-23",
    freq="7D"
)

TARGET_THRESHOLD = 0.80


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def normalize_gateway_id(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(":", "", regex=False)
        .str.replace("-", "", regex=False)
    )


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def calculate_cost(y_true, probabilities, threshold):
    """
    Cost-aware evaluation.

    The top 15 gateways by predicted probability are selected.
    """

    temp = pd.DataFrame({
        "actual": np.asarray(y_true),
        "probability": np.asarray(probabilities)
    }).reset_index(drop=True)

    selected = (
        temp.sort_values(
            "probability",
            ascending=False
        )
        .head(MAX_VISITS)
        .index
    )

    visit_mask = temp.index.isin(selected)

    unnecessary = (
        visit_mask &
        (temp["actual"] == 0)
    ).sum()

    missed = (
        (~visit_mask) &
        (temp["actual"] == 1)
    ).sum()

    cost = (
        unnecessary * UNNECESSARY_VISIT_COST
        + missed * MISSED_PROBLEM_COST
    )

    return cost, unnecessary, missed


def make_reason(row):
    reasons = []

    if row.get("lag1_success", 1) < 0.80:
        reasons.append(
            "low recent meter-read success"
        )

    if row.get("rolling_4w_mean", 1) < 0.80:
        reasons.append(
            "weak 4-week read-success trend"
        )

    if row.get(
        "tele_offline_duration_sec_mean",
        0
    ) > 1000:
        reasons.append(
            "high offline duration"
        )

    if row.get(
        "tele_no_conn_importance_mean",
        0
    ) > 5000:
        reasons.append(
            "high no-connection signal"
        )

    if row.get(
        "tele_avg_load1_mean",
        0
    ) > 1:
        reasons.append(
            "high gateway load"
        )

    if row.get(
        "tele_rssi_bad_mean",
        0
    ) > 1:
        reasons.append(
            "poor radio quality"
        )

    if not reasons:
        reasons.append(
            "highest predicted next-week failure risk"
        )

    return "; ".join(reasons[:2])


# ============================================================
# 2. LOAD METER DATA
# ============================================================

print("\n1. LOADING METER DATA")
print("-" * 75)

meter_file = os.path.join(
    DATA_DIR,
    "meter_read_success.csv"
)

meter = pd.read_csv(
    meter_file,
    encoding="latin1"
)
meter["gateway_id"] = normalize_gateway_id(
    meter["gateway_id"]
)

meter["week_start"] = pd.to_datetime(
    meter["week_start"],
    errors="coerce"
).dt.normalize()

meter["meters_expected"] = safe_numeric(
    meter["meters_expected"]
)

meter["meters_read"] = safe_numeric(
    meter["meters_read"]
)

meter["read_success"] = np.where(
    meter["meters_expected"] > 0,
    meter["meters_read"] / meter["meters_expected"],
    np.nan
)

meter = (
    meter
    .sort_values(
        ["gateway_id", "week_start"]
    )
    .drop_duplicates(
        ["gateway_id", "week_start"]
    )
)

print("Rows:", len(meter))
print(
    "Gateways:",
    meter["gateway_id"].nunique()
)
print(
    "Weeks:",
    meter["week_start"].min().date(),
    "to",
    meter["week_start"].max().date()
)


# ============================================================
# 3. CREATE SUPERVISED LEARNING TARGET
# ============================================================

print("\n2. BUILDING TEMPORAL TARGET")
print("-" * 75)

meter["target_next_week"] = (
    meter
    .groupby("gateway_id")["read_success"]
    .shift(-1)
)

meter["next_week_start"] = (
    meter
    .groupby("gateway_id")["week_start"]
    .shift(-1)
)

meter["problem_next_week"] = (
    meter["target_next_week"] < TARGET_THRESHOLD
).astype(int)

model_base = meter[
    meter["target_next_week"].notna()
].copy()

print(
    "Training rows:",
    len(model_base)
)

print(
    "Positive target rate:",
    round(
        model_base["problem_next_week"].mean(),
        4
    )
)


# ============================================================
# 4. METER HISTORY FEATURES
# ============================================================

print("\n3. CREATING METER HISTORY FEATURES")
print("-" * 75)

group = meter.groupby(
    "gateway_id"
)["read_success"]

for lag in [1, 2, 3, 4]:

    model_base[
        f"lag{lag}_success"
    ] = (
        group.shift(lag)
        .loc[model_base.index]
        .values
    )


model_base["rolling_4w_mean"] = (
    group.transform(
        lambda x:
        x.shift(1)
        .rolling(
            4,
            min_periods=1
        )
        .mean()
    )
    .loc[model_base.index]
    .values
)


model_base["rolling_4w_std"] = (
    group.transform(
        lambda x:
        x.shift(1)
        .rolling(
            4,
            min_periods=2
        )
        .std()
    )
    .loc[model_base.index]
    .values
)


model_base["success_change_1w"] = (
    model_base["lag1_success"]
    - model_base["lag2_success"]
)


model_base["success_change_2w"] = (
    model_base["lag1_success"]
    - model_base["lag3_success"]
)


# ============================================================
# 5. LOAD GATEWAY MASTER
# ============================================================

print("\n4. LOADING GATEWAY MASTER")
print("-" * 75)

gateway_master_file = os.path.join(`r`n    DATA_DIR,`r`n    "gateway_master.csv"`r`n)`r`n`r`ngateway_master = pd.read_csv(`r`n    gateway_master_file,`r`n    encoding="latin1"`r`n)

gateway_master["gateway_id"] = (
    normalize_gateway_id(
        gateway_master["gateway_id"]
    )
)

print(
    "Gateway master rows:",
    len(gateway_master)
)


# ============================================================
# 6. LOAD TELEMETRY
# ============================================================

print("\n5. LOADING TELEMETRY")
print("-" * 75)

telemetry_files = sorted(
    glob.glob(
        os.path.join(
            "telemetry",
            "month=*",
            "*.parquet"
        )
    )
)

print(
    "Telemetry partitions:",
    len(telemetry_files)
)

if len(telemetry_files) == 0:
    raise FileNotFoundError(
        "No telemetry parquet files were found."
    )


telemetry_frames = []

for file in telemetry_files:

    temp = pd.read_parquet(file)

    if "gateway_id" not in temp.columns:
        continue

    temp["gateway_id"] = (
        normalize_gateway_id(
            temp["gateway_id"]
        )
    )

    if "ts_utc" in temp.columns:

        temp["ts_utc"] = pd.to_datetime(
            temp["ts_utc"],
            errors="coerce",
            utc=True
        )

    telemetry_frames.append(temp)


if len(telemetry_frames) == 0:
    raise ValueError(
        "Telemetry files were found, but no usable "
        "gateway_id data was loaded."
    )


telemetry = pd.concat(
    telemetry_frames,
    ignore_index=True
)

print(
    "Telemetry rows:",
    len(telemetry)
)

print(
    "Telemetry gateways:",
    telemetry["gateway_id"].nunique()
)

print(
    "Telemetry timestamp:",
    telemetry["ts_utc"].min(),
    "to",
    telemetry["ts_utc"].max()
)


# ============================================================
# 7. WEEKLY TELEMETRY
# ============================================================

print("\n6. BUILDING WEEKLY TELEMETRY FEATURES")
print("-" * 75)

if "ts_utc" not in telemetry.columns:
    raise ValueError(
        "Telemetry does not contain ts_utc."
    )


# ------------------------------------------------------------
# FIX:
#
# Meter week_start values are Mondays.
#
# Instead of using:
#     to_period("W-MON").start_time
#
# which gives Tuesday as the start of a week ending Monday,
# explicitly calculate the Monday for each local timestamp.
#
# This makes telemetry week_start compatible with
# meter_read_success.week_start.
# ------------------------------------------------------------

local_time = (
    telemetry["ts_utc"]
    .dt.tz_convert("Europe/Berlin")
)

local_date = local_time.dt.normalize()

telemetry["week_start"] = (
    local_date
    - pd.to_timedelta(
        local_date.dt.weekday,
        unit="D"
    )
)

# Remove timezone so it exactly matches meter week_start.
telemetry["week_start"] = (
    telemetry["week_start"]
    .dt.tz_localize(None)
    .dt.normalize()
)


print(
    "Telemetry week range:",
    telemetry["week_start"].min().date(),
    "to",
    telemetry["week_start"].max().date()
)


numeric_candidates = [
    "rx_nr_pkts",
    "rx_crc_bad",
    "tx_success",
    "tx_busy",
    "tx_override",
    "number_of_messages",
    "avg_idletime",
    "avg_load1",
    "load1_bigger1",
    "load1_bigger2",
    "avg_memfree",
    "avg_uptime",
    "avg_activeproccess",
    "avg_totalproccess",
    "reboot_cnt",
    "reboot_duration_sec",
    "r_cnt_power_cycle",
    "r_cnt_reboot",
    "r_cnt_unknown",
    "avg_reboot_duration",
    "reboot_importance",
    "disconnection_cnt",
    "offline_duration_sec",
    "avg_offline_duration",
    "online_duration_mins",
    "no_conn_importance",
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

telemetry_features = [
    c
    for c in numeric_candidates
    if c in telemetry.columns
]

print(
    "Telemetry numeric features:",
    len(telemetry_features)
)


for c in telemetry_features:

    telemetry[c] = safe_numeric(
        telemetry[c]
    )


weekly_telemetry = (
    telemetry
    .groupby(
        [
            "gateway_id",
            "week_start"
        ]
    )[telemetry_features]
    .agg(
        ["mean", "max", "sum"]
    )
)

weekly_telemetry.columns = [
    "tele_" + col[0] + "_" + col[1]
    for col in weekly_telemetry.columns
]

weekly_telemetry = (
    weekly_telemetry
    .reset_index()
)

print(
    "Weekly telemetry rows:",
    len(weekly_telemetry)
)

print(
    "Weekly telemetry gateways:",
    weekly_telemetry[
        "gateway_id"
    ].nunique()
)


# ============================================================
# 8. ALIGN TELEMETRY WITHOUT FUTURE LEAKAGE
# ============================================================

print("\n7. ALIGNING TELEMETRY WITHOUT FUTURE LEAKAGE")
print("-" * 75)

tele_columns = [
    c
    for c in weekly_telemetry.columns
    if c.startswith("tele_")
]


# ------------------------------------------------------------
# For a training row with week_start W:
#
#   meter data at W
#       ->
#   predict whether next week's read success is < 80%
#
# Therefore telemetry from W is allowed.
#
# For example:
#
#   Jan 26 telemetry
#       ->
#   predict Feb 2 meter-read problem
#
# This does not use future information.
# ------------------------------------------------------------

model_base = model_base.merge(
    weekly_telemetry,
    on=[
        "gateway_id",
        "week_start"
    ],
    how="left"
)


rows_with_telemetry = (
    model_base[tele_columns]
    .notna()
    .any(axis=1)
    .sum()
)

print(
    "Rows with telemetry:",
    rows_with_telemetry,
    "/",
    len(model_base)
)

print(
    "Telemetry coverage:",
    round(
        rows_with_telemetry / len(model_base),
        4
    )
)


# Extra diagnostic:
training_keys = model_base[
    ["gateway_id", "week_start"]
].drop_duplicates()

telemetry_keys = weekly_telemetry[
    ["gateway_id", "week_start"]
].drop_duplicates()

matching_keys = training_keys.merge(
    telemetry_keys,
    on=[
        "gateway_id",
        "week_start"
    ],
    how="inner"
)

print(
    "Matching gateway-week keys:",
    len(matching_keys)
)


if rows_with_telemetry == 0:

    raise RuntimeError(
        "Telemetry alignment failed: zero model rows "
        "have telemetry. Check gateway IDs and week_start."
    )


# ============================================================
# 9. MERGE GATEWAY STATIC FEATURES
# ============================================================

print("\n8. ADDING GATEWAY FEATURES")
print("-" * 75)

static_candidates = [
    "tenant",
    "site_type",
    "region",
    "hw_model",
    "antenna_type",
    "n_meters_installed"
]

static_features = [
    c
    for c in static_candidates
    if c in gateway_master.columns
]

model_base = model_base.merge(
    gateway_master[
        ["gateway_id"] + static_features
    ],
    on="gateway_id",
    how="left"
)

print(
    "Static features:",
    static_features
)


# ============================================================
# 10. FEATURE LIST
# ============================================================

meter_features = [
    "lag1_success",
    "lag2_success",
    "lag3_success",
    "lag4_success",
    "rolling_4w_mean",
    "rolling_4w_std",
    "success_change_1w",
    "success_change_2w"
]

tele_features = [
    c
    for c in model_base.columns
    if c.startswith("tele_")
]

feature_columns = (
    meter_features
    + tele_features
    + static_features
)

feature_columns = [
    c
    for c in feature_columns
    if c in model_base.columns
]

print(
    "Raw feature count:",
    len(feature_columns)
)


# ============================================================
# 11. PREPARE MODEL DATA
# ============================================================

model_data = model_base[
    [
        "gateway_id",
        "week_start",
        "problem_next_week"
    ]
    + feature_columns
].copy()


categorical_features = [
    c
    for c in static_features
    if model_data[c].dtype == "object"
]

numeric_features = [
    c
    for c in feature_columns
    if c not in categorical_features
]


for c in numeric_features:

    model_data[c] = safe_numeric(
        model_data[c]
    )

    model_data[c] = model_data[c].fillna(
        model_data[c].median()
    )


for c in categorical_features:

    model_data[c] = (
        model_data[c]
        .fillna("UNKNOWN")
        .astype(str)
    )


# ============================================================
# 12. ONE-HOT ENCODING
# ============================================================

if categorical_features:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    encoded = encoder.fit_transform(
        model_data[
            categorical_features
        ]
    )

    encoded_names = (
        encoder.get_feature_names_out(
            categorical_features
        )
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_names,
        index=model_data.index
    )

    X_numeric = (
        model_data[
            numeric_features
        ]
        .reset_index(drop=True)
    )

    X = pd.concat(
        [
            X_numeric,
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

else:

    encoder = None
    encoded_names = []

    X = model_data[
        numeric_features
    ].reset_index(drop=True)


y = (
    model_data[
        "problem_next_week"
    ]
    .astype(int)
    .reset_index(drop=True)
)

metadata = (
    model_data[
        ["gateway_id", "week_start"]
    ]
    .reset_index(drop=True)
)


print(
    "Final model rows:",
    len(X)
)

print(
    "Encoded feature count:",
    X.shape[1]
)


# ============================================================
# 13. TEMPORAL TRAIN / TEST SPLIT
# ============================================================

print("\n9. TEMPORAL TRAIN / TEST SPLIT")
print("-" * 75)

unique_weeks = sorted(
    model_data[
        "week_start"
    ].unique()
)

test_weeks = unique_weeks[-6:]

train_mask = ~metadata[
    "week_start"
].isin(test_weeks)

test_mask = metadata[
    "week_start"
].isin(test_weeks)


X_train = X.loc[
    train_mask
]

X_test = X.loc[
    test_mask
]

y_train = y.loc[
    train_mask
]

y_test = y.loc[
    test_mask
]


print(
    "Training:",
    len(X_train),
    "rows"
)

print(
    "Testing:",
    len(X_test),
    "rows"
)

print(
    "Train positive rate:",
    round(
        y_train.mean(),
        4
    )
)

print(
    "Test positive rate:",
    round(
        y_test.mean(),
        4
    )
)


# ============================================================
# 14. TRAIN RANDOM FOREST
# ============================================================

print("\n10. TRAINING RANDOM FOREST")
print("-" * 75)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

test_prob = model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= 0.5
).astype(int)


roc_auc = roc_auc_score(
    y_test,
    test_prob
)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)


print(
    "ROC-AUC:",
    round(roc_auc, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)


# ============================================================
# 15. COST-AWARE THRESHOLD SEARCH
# ============================================================

print("\n11. COST-AWARE THRESHOLD SEARCH")
print("-" * 75)

threshold_results = []

for threshold in np.arange(
    0.20,
    0.81,
    0.02
):

    cost, unnecessary, missed = calculate_cost(
        y_test,
        test_prob,
        threshold
    )

    threshold_results.append({
        "threshold": round(
            threshold,
            2
        ),
        "cost": cost,
        "unnecessary": unnecessary,
        "missed": missed
    })


threshold_df = pd.DataFrame(
    threshold_results
)

threshold_df.to_csv(
    "step4_cost_thresholds.csv",
    index=False
)

best_threshold_row = (
    threshold_df
    .sort_values("cost")
    .iloc[0]
)

BEST_THRESHOLD = float(
    best_threshold_row[
        "threshold"
    ]
)

print(
    "Best cost threshold:",
    BEST_THRESHOLD
)

print(
    "Validation cost:",
    int(
        best_threshold_row[
            "cost"
        ]
    )
)


# ============================================================
# 16. WEEKLY TEMPORAL EVALUATION
# ============================================================

print("\n12. WEEKLY COST EVALUATION")
print("-" * 75)

weekly_results = []

test_meta = metadata.loc[
    test_mask
].copy()

test_meta["actual"] = (
    y_test.values
)

test_meta["probability"] = (
    test_prob
)

for week in sorted(
    test_meta[
        "week_start"
    ].unique()
):

    week_data = test_meta[
        test_meta[
            "week_start"
        ] == week
    ].copy()

    selected = (
        week_data
        .sort_values(
            "probability",
            ascending=False
        )
        .head(MAX_VISITS)
    )

    unnecessary = (
        selected[
            "actual"
        ] == 0
    ).sum()

    missed = (
        (
            ~week_data.index.isin(
                selected.index
            )
        )
        &
        (
            week_data[
                "actual"
            ] == 1
        )
    ).sum()

    cost = (
        unnecessary *
        UNNECESSARY_VISIT_COST
        +
        missed *
        MISSED_PROBLEM_COST
    )

    weekly_results.append({
        "week_start": week,
        "selected": len(selected),
        "actual_problems": int(
            week_data[
                "actual"
            ].sum()
        ),
        "unnecessary": int(
            unnecessary
        ),
        "missed": int(
            missed
        ),
        "cost": int(
            cost
        )
    })


weekly_eval = pd.DataFrame(
    weekly_results
)

weekly_eval.to_csv(
    "step4_weekly_evaluation.csv",
    index=False
)

print(
    weekly_eval.to_string(
        index=False
    )
)

print(
    "\nTotal ML validation cost:",
    int(
        weekly_eval[
            "cost"
        ].sum()
    )
)


# ============================================================
# 17. UNSEEN GATEWAY TEST
# ============================================================

print("\n13. UNSEEN GATEWAY TEST")
print("-" * 75)

all_gateways = set(
    model_data[
        "gateway_id"
    ].unique()
)

train_gateways = set(
    metadata.loc[
        train_mask,
        "gateway_id"
    ].unique()
)

unseen_gateways = (
    all_gateways -
    train_gateways
)

unseen_mask = metadata[
    "gateway_id"
].isin(
    unseen_gateways
)


if unseen_mask.sum() > 0:

    unseen_prob = model.predict_proba(
        X.loc[unseen_mask]
    )[:, 1]

    unseen_y = y.loc[
        unseen_mask
    ]

    print(
        "Unseen gateways:",
        len(unseen_gateways)
    )

    print(
        "Unseen gateway rows:",
        unseen_mask.sum()
    )

    if unseen_y.nunique() > 1:

        unseen_auc = roc_auc_score(
            unseen_y,
            unseen_prob
        )

        print(
            "Unseen gateway ROC-AUC:",
            round(
                unseen_auc,
                4
            )
        )

    else:

        print(
            "Unseen gateway ROC-AUC cannot "
            "be calculated because only one "
            "class is present."
        )

else:

    print(
        "No completely unseen gateways "
        "available for this split."
    )


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

print("\n14. FEATURE IMPORTANCE")
print("-" * 75)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance":
        model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)

importance_df.to_csv(
    "step4_feature_importance.csv",
    index=False
)

print(
    importance_df
    .head(20)
    .to_string(
        index=False
    )
)


plt.figure(
    figsize=(10, 7)
)

top_features = (
    importance_df
    .head(15)
    .sort_values(
        "importance"
    )
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title(
    "Top ML Feature Importance"
)

plt.xlabel(
    "Random Forest Importance"
)

plt.tight_layout()

plt.savefig(
    "step4_feature_importance.png",
    dpi=150
)

plt.close()


# ============================================================
# 19. FINAL 8-WEEK PREDICTION
# ============================================================

print("\n15. GENERATING FINAL 8-WEEK PREDICTIONS")
print("-" * 75)


# ------------------------------------------------------------
# Initial history comes from the latest observed meter data.
# ------------------------------------------------------------

latest_meter = (
    meter
    .sort_values(
        "week_start"
    )
    .copy()
)


history = {}

for gateway_id, group_df in (
    latest_meter
    .groupby("gateway_id")
):

    group_df = (
        group_df
        .sort_values(
            "week_start"
        )
    )

    history[gateway_id] = list(
        group_df[
            "read_success"
        ]
        .dropna()
        .tail(4)
    )


candidate_gateways = sorted(
    gateway_master[
        "gateway_id"
    ]
    .dropna()
    .unique()
)


final_predictions = []


for prediction_week in PREDICTION_WEEKS:

    feature_week = (
        prediction_week
        - pd.Timedelta(
            days=7
        )
    )

    rows = []

    for gateway_id in candidate_gateways:

        hist = history.get(
            gateway_id,
            []
        )

        if len(hist) == 0:

            lag1 = meter[
                "read_success"
            ].median()

            lag2 = lag1
            lag3 = lag1
            lag4 = lag1

        else:

            lag1 = hist[-1]

            lag2 = (
                hist[-2]
                if len(hist) >= 2
                else lag1
            )

            lag3 = (
                hist[-3]
                if len(hist) >= 3
                else lag2
            )

            lag4 = (
                hist[-4]
                if len(hist) >= 4
                else lag3
            )


        history_values = [
            lag4,
            lag3,
            lag2,
            lag1
        ]

        valid_history = [
            x
            for x in history_values
            if pd.notna(x)
        ]


        if len(valid_history) == 0:

            rolling_mean = (
                meter[
                    "read_success"
                ].median()
            )

            rolling_std = 0

        else:

            rolling_mean = np.mean(
                valid_history
            )

            rolling_std = (
                np.std(
                    valid_history
                )
                if len(valid_history) >= 2
                else 0
            )


        # ----------------------------------------------------
        # Get telemetry from latest completed week.
        # ----------------------------------------------------

        tele_row = weekly_telemetry[
            (
                weekly_telemetry[
                    "gateway_id"
                ] == gateway_id
            )
            &
            (
                weekly_telemetry[
                    "week_start"
                ] == feature_week
            )
        ]


        row = {
            "gateway_id":
                gateway_id,

            "week_start":
                feature_week,

            "lag1_success":
                lag1,

            "lag2_success":
                lag2,

            "lag3_success":
                lag3,

            "lag4_success":
                lag4,

            "rolling_4w_mean":
                rolling_mean,

            "rolling_4w_std":
                rolling_std,

            "success_change_1w":
                lag1 - lag2,

            "success_change_2w":
                lag1 - lag3
        }


        # ----------------------------------------------------
        # Add telemetry features.
        # ----------------------------------------------------

        if len(tele_row) > 0:

            tele_values = (
                tele_row.iloc[0]
            )

            for col in tele_features:

                row[col] = tele_values.get(
                    col,
                    np.nan
                )

        else:

            for col in tele_features:

                row[col] = model_base[
                    col
                ].median()


        # ----------------------------------------------------
        # Add static gateway features.
        # ----------------------------------------------------

        master_row = gateway_master[
            gateway_master[
                "gateway_id"
            ] == gateway_id
        ]


        if len(master_row) > 0:

            master_row = (
                master_row.iloc[0]
            )

            for col in static_features:

                row[col] = (
                    master_row.get(
                        col,
                        "UNKNOWN"
                    )
                )

        else:

            for col in static_features:

                row[col] = "UNKNOWN"


        rows.append(row)


    future_features = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Numeric cleanup
    # --------------------------------------------------------

    for col in numeric_features:

        if col in future_features.columns:

            future_features[col] = (
                safe_numeric(
                    future_features[col]
                )
            )

            future_features[col] = (
                future_features[col]
                .fillna(
                    model_base[
                        col
                    ].median()
                )
            )


    # --------------------------------------------------------
    # Categorical cleanup
    # --------------------------------------------------------

    for col in categorical_features:

        if col in future_features.columns:

            future_features[col] = (
                future_features[col]
                .fillna("UNKNOWN")
                .astype(str)
            )


    # --------------------------------------------------------
    # Encode using the same encoder.
    # --------------------------------------------------------

    if encoder is not None:

        encoded_future = (
            encoder.transform(
                future_features[
                    categorical_features
                ]
            )
        )

        encoded_future_df = (
            pd.DataFrame(
                encoded_future,
                columns=encoded_names
            )
        )

        future_X_numeric = (
            future_features[
                numeric_features
            ]
            .reset_index(drop=True)
        )

        future_X = pd.concat(
            [
                future_X_numeric,
                encoded_future_df.reset_index(
                    drop=True
                )
            ],
            axis=1
        )

    else:

        future_X = (
            future_features[
                numeric_features
            ]
            .reset_index(drop=True)
        )


    # Exact training feature order.
    future_X = future_X[
        X.columns
    ]


    probabilities = (
        model.predict_proba(
            future_X
        )[:, 1]
    )


    future_features[
        "predicted_probability"
    ] = probabilities


    # --------------------------------------------------------
    # Expected value of a visit:
    #
    # probability * €600 - €380
    # --------------------------------------------------------

    future_features[
        "expected_visit_value"
    ] = (
        future_features[
            "predicted_probability"
        ]
        * MISSED_PROBLEM_COST
        - UNNECESSARY_VISIT_COST
    )


    # --------------------------------------------------------
    # Select highest-risk 15 gateways.
    # --------------------------------------------------------

    selected = (
        future_features
        .sort_values(
            [
                "predicted_probability",
                "expected_visit_value"
            ],
            ascending=False
        )
        .head(
            MAX_VISITS
        )
        .copy()
    )


    selected = (
        selected
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Create final prediction rows.
    # --------------------------------------------------------

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        final_predictions.append({
            "week_start":
                prediction_week.strftime(
                    "%Y-%m-%d"
                ),

            "rank":
                rank,

            "gateway_id":
                row[
                    "gateway_id"
                ],

            "score":
                round(
                    float(
                        row[
                            "predicted_probability"
                        ]
                    ),
                    6
                ),

            "reason":
                make_reason(row)
        })


    # --------------------------------------------------------
    # Recursive meter-history update.
    #
    # Future actual meter success is unknown.
    # We estimate it as:
    #
    # predicted success = 1 - probability of problem
    #
    # This allows the 8-week forecast to continue without
    # using unavailable future labels.
    # --------------------------------------------------------

    for _, row in (
        future_features.iterrows()
    ):

        gateway_id = row[
            "gateway_id"
        ]

        predicted_success = (
            1.0
            -
            float(
                row[
                    "predicted_probability"
                ]
            )
        )

        if gateway_id not in history:

            history[gateway_id] = []

        history[gateway_id].append(
            predicted_success
        )

        history[gateway_id] = (
            history[gateway_id][-4:]
        )


# ============================================================
# 20. SAVE FINAL PREDICTIONS
# ============================================================

predictions = pd.DataFrame(
    final_predictions
)

predictions.to_csv(
    "predictions.csv",
    index=False
)

print(
    "\nPrediction rows:",
    len(predictions)
)

print(
    "\nPrediction count by week:"
)

print(
    predictions[
        "week_start"
    ]
    .value_counts()
    .sort_index()
)

print(
    "\nFinal prediction sample:"
)

print(
    predictions
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# 21. FINAL VALIDATION CHECKS
# ============================================================

print("\n16. FINAL VALIDATION CHECKS")
print("-" * 75)

expected_weeks = [
    x.strftime(
        "%Y-%m-%d"
    )
    for x in PREDICTION_WEEKS
]

actual_weeks = sorted(
    predictions[
        "week_start"
    ].unique()
)


print(
    "Expected weeks:",
    len(expected_weeks)
)

print(
    "Actual weeks:",
    len(actual_weeks)
)

print(
    "Total rows:",
    len(predictions)
)


assert len(predictions) == 120, (
    f"Expected 120 rows, got "
    f"{len(predictions)}"
)


assert actual_weeks == expected_weeks, (
    "Prediction weeks are incorrect."
)


for week in expected_weeks:

    count = (
        predictions[
            predictions[
                "week_start"
            ] == week
        ]
        .shape[0]
    )

    assert count == 15, (
        f"{week}: expected 15 rows, "
        f"got {count}"
    )


assert predictions[
    "rank"
].between(
    1,
    15
).all()


assert predictions[
    [
        "week_start",
        "rank",
        "gateway_id",
        "score",
        "reason"
    ]
].notna().all().all()


print(
    "PASS: 8 weeks × 15 gateways = 120 rows"
)

print(
    "PASS: ranks 1-15 for every week"
)

print(
    "PASS: required columns present"
)

print(
    "PASS: no missing prediction values"
)


# ============================================================
# 22. SAVE MODEL SUMMARY
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "target",
        "target_threshold",
        "roc_auc",
        "precision",
        "recall",
        "best_cost_threshold",
        "validation_cost",
        "telemetry_rows_aligned",
        "telemetry_coverage",
        "final_prediction_rows",
        "prediction_weeks"
    ],

    "value": [
        "next-week read success < 80%",
        TARGET_THRESHOLD,
        round(
            roc_auc,
            4
        ),
        round(
            precision,
            4
        ),
        round(
            recall,
            4
        ),
        BEST_THRESHOLD,
        int(
            threshold_df[
                "cost"
            ].min()
        ),
        int(
            rows_with_telemetry
        ),
        round(
            rows_with_telemetry /
            len(model_base),
            4
        ),
        len(predictions),
        len(PREDICTION_WEEKS)
    ]
})


summary.to_csv(
    "step4_model_summary.csv",
    index=False
)


print(
    "\n" + "=" * 75
)

print(
    "STEP 4 V3 COMPLETE"
)

print(
    "=" * 75
)

print(
    "predictions.csv now contains exactly",
    len(predictions),
    "rows."
)

print(
    "Telemetry was explicitly aligned "
    "to Monday week_start values."
)

print(
    "Next step: run validate_submission.py"
)

print(
    "=" * 75
)


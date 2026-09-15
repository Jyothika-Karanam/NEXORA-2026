# NEXORA 2026 — What It Cannot Do

## Current limitations

### 1. Meter-read success is a proxy, not perfect fault ground truth

The solution defines a problematic gateway using **next-week meter-read success below 80%**. A failed meter read does not always mean that the physical gateway is faulty.

Historical field visits demonstrate this limitation: a large proportion of visits ended with **"Kein Fehler gefunden"**.

### 2. The model cannot guarantee that a selected gateway will have a physical fault

The model predicts the probability of a future meter-read problem. It does not directly diagnose hardware failure, wiring problems, communication faults, or other physical causes.

The prediction should therefore be interpreted as a **visit-priority score**, not a guaranteed fault diagnosis.

### 3. The 15-gateway limit forces prioritization

There may be more than 15 gateways with elevated risk in a week. The system must therefore rank gateways and select only the highest-priority 15.

A gateway ranked below 15 may still have a genuine problem but will not be selected because of the operational limit.

### 4. Future predictions depend on historical patterns

The model learns from historical telemetry and meter-read behavior. It cannot know about operational conditions that have not appeared in the available historical data.

This means prediction quality may decrease when future conditions are substantially different from the training period.

### 5. Network and operational changes can reduce accuracy

Changes such as:

* Network operator changes
* Network outages
* Hardware replacements
* Firmware changes
* Antenna changes
* Changes in the number of installed meters
* Major changes in gateway configuration

may cause future gateway behavior to differ from historical patterns.

The model includes historical telemetry and gateway information, but it cannot fully anticipate these changes before they occur.

## What two additional weeks of work would improve

### 1. Test robustness against temporal and network changes

I would evaluate the model across additional future-like time periods and specifically test how its performance changes after network, hardware, firmware, or configuration changes.

I would also investigate probability calibration and threshold selection so that the visit ranking remains cost-aware under changing operating conditions.

### 2. Add stronger operational ground truth

I would investigate whether field-visit outcomes, engineer reviews, maintenance records, and gateway configuration changes can be combined into a more reliable fault label.

This could reduce the dependence on meter-read success as a proxy for gateway problems and help distinguish communication problems from genuine physical faults.

## What the solution should not claim

The solution should not claim that it can:

* Guarantee which gateways will fail.
* Diagnose the exact physical cause of a problem.
* Eliminate unnecessary visits.
* Detect every problematic gateway.
* Remain equally accurate after major network or operational changes.

The intended output is a **cost-aware ranking of the 15 gateways most deserving of field attention each week**, rather than a guaranteed fault diagnosis.

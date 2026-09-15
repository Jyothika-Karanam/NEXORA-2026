# NEXORA 2026 — AI Usage

## How AI was used

AI assistance was used during the development of the NEXORA 2026 solution for:

* Understanding the challenge requirements and data dictionary.
* Planning the exploratory data analysis.
* Designing and refining the target definition.
* Developing Python scripts for data checking, EDA, target analysis, model training, and evaluation.
* Debugging data-processing and date/week-alignment issues.
* Reviewing model evaluation logic and operational cost calculations.
* Improving documentation and explaining technical decisions clearly.

AI was used as a development and reasoning assistant. The final implementation was run and tested locally against the challenge data, and the generated predictions were checked using the provided validator.

## What AI got wrong and how it was caught

One important issue was found in the machine-learning implementation.

An earlier version of the ML pipeline attempted to combine telemetry with weekly meter-read data, but the telemetry week calculation did not align with the Monday-based week used by the meter-read data. As a result, the telemetry features were not actually being matched to the meter-read observations.

This was identified by checking the number of rows that successfully received telemetry features. The telemetry match count was effectively zero in the affected version.

The week-alignment logic was then corrected by converting telemetry timestamps to Europe/Berlin local time and explicitly calculating the Monday start of each week. After the correction, almost all training rows received telemetry information:

**6,926 of 6,927 training rows matched telemetry.**

The model was then retrained and evaluated again.

A second validation issue was also identified while comparing the ML model with the supplied 3-sigma baseline. The initial comparison attempted to evaluate future prediction weeks for which the actual meter-read outcome was not yet available. The evaluation was corrected to score historical test weeks against the following week's known outcome.

The corrected comparison showed:

* 3-sigma baseline cost: **€144,920**
* ML validation cost: **€111,600**
* Cost reduction: **€33,320**
* Improvement: **23.0%**

These checks were important because model code can execute successfully while still producing an incorrect evaluation if time alignment or target alignment is wrong.

## Human verification

The final solution was not accepted solely because AI-generated code executed successfully.

The implementation was manually inspected and tested by:

1. Checking dataset shapes, missing values, duplicates, and gateway IDs.
2. Verifying gateway ID normalization and joins.
3. Checking temporal alignment between telemetry and meter-read data.
4. Checking that future information was not used when constructing historical features.
5. Comparing the ML model against the supplied 3-sigma baseline using the same operational cost function.
6. Running the provided submission validator.

The final submission passed the validator with:

**15 ranked gateways for each of 8 weeks, covering 2026-02-02 to 2026-03-23.**

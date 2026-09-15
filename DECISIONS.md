# NEXORA 2026 — Key Decisions

## 1. What counts as a gateway that "needs a visit"?

### Decision

We define a gateway as needing attention when its **next-week meter-read success is below 80%**.

This definition is used as the main prediction target for the Data Science + Machine Learning solution.

The reason for using this definition is that meter-read success directly represents whether the gateway is successfully supporting meter reads. The data also provides enough weekly history to evaluate whether current gateway behavior predicts a problem in the following week.

The analysis showed a clear relationship between current-week read success and next-week problems:

* Current success below 50% → 100.00% next-week problem rate
* 50–70% → 89.39%
* 70–80% → 40.60%
* 80–90% → 8.76%
* 90–100% → 1.19%

This provides evidence that lower current read success is strongly associated with a future meter-read problem.

### Alternative considered: Field visit records

We considered defining a gateway as problematic when it received a field visit.

We rejected this because a field visit is an operational action rather than direct ground truth for a gateway fault. Of the 642 historical visits, 390 ended with **"Kein Fehler gefunden"**, meaning that 60.75% of visits did not identify a confirmed error.

Therefore, using every field visit as a positive fault label could teach the model to predict previous human decisions rather than actual gateway problems.

### Alternative considered: More aggressive read-success thresholds

We also considered thresholds of 70% and 90%.

We rejected 70% as the primary definition because it identifies a narrower set of severe problems and may miss gateways with meaningful degradation between 70% and 80%.

We rejected 90% as the primary definition because it marks a much larger proportion of gateway-weeks as problematic and would make the target less specific.

The 80% threshold provides a practical middle ground for the prediction task.

---

## 2. How should the model decide whether a gateway is worth visiting?

### Decision

We use a **cost-aware ranking approach** rather than treating the prediction as a simple fault/no-fault diagnosis.

The operational costs are:

* Unnecessary visit: **€380**
* Missed problematic gateway: **€600**

The probability at which a visit becomes economically justified is:

**€380 / (€380 + €600) = 38.78%**

Therefore, a predicted probability above approximately 38.78% can justify a visit when considered independently.

However, the field team can visit only **15 gateways per week**. Therefore, the final operational decision is to rank gateways by predicted risk and select the top 15.

### Alternative considered: Fixed current read-success threshold

We tested simple rules based only on current-week meter-read success.

The lowest historical cost among the tested thresholds occurred at a **75% current read-success threshold**, with a cost of €194,400 over the evaluation period.

We did not use this as the final solution because it does not use the additional historical telemetry, lagged behavior, gateway characteristics, or machine-learning probability estimates.

The 75% rule is retained as a useful simple benchmark.

---

## 3. Which Part 2 area did we choose?

### Decision

We chose **Data Science + Machine Learning**.

The solution combines both areas into one operational pipeline:

1. Explore the available gateway, visit, meter-read, and telemetry data.
2. Define a measurable future problem.
3. Quantify the operational cost of unnecessary and missed visits.
4. Build a machine-learning model to estimate future problem risk.
5. Rank gateways by risk.
6. Select the highest-priority 15 gateways for each week.
7. Compare the model against the supplied 3-sigma baseline.

### Why this area?

Data Science provides the target definition, exploratory analysis, cost analysis, and evidence for the decision.

Machine Learning then uses historical meter-read behavior, telemetry, and gateway information to improve the ranking of future problem gateways.

This gives the field team an actionable output rather than only a descriptive analysis.

### Alternative considered: Data Engineering

Data Engineering would be useful for building a production-quality pipeline, automated ingestion, monitoring, and scalable data processing.

We did not select it as the main Part 2 focus because the challenge's main decision is which 15 gateways should receive field attention each week. Data Science + Machine Learning directly addresses that decision.

### Alternative considered: Dashboard / Visualization

A dashboard can help an operations manager understand the results, but visualization alone does not solve the gateway-prioritization problem.

We therefore use charts to explain the model and decision rather than treating visualization as the main Part 2 solution.

---

## 4. Which machine-learning approach did we choose?

### Decision

We use a **Random Forest classifier** to predict whether the following week's meter-read success will be below 80%.

The model uses historical information available before the prediction week, including:

* Previous weeks' meter-read success
* Lagged read-success values
* Rolling read-success statistics
* Changes in recent read success
* Historical telemetry aggregates
* Offline-duration behavior
* Connection-related telemetry
* Reboot-related telemetry
* Radio/network-quality telemetry
* Gateway characteristics

Telemetry is aligned to the correct Monday-based week and only information available before the prediction week is used.

### Why Random Forest?

Random Forest can model nonlinear relationships between gateway behavior and future problems without requiring strong assumptions about the form of those relationships.

It also provides feature-importance information that can be explained to an operations manager.

The final model achieved a validation cost of **€111,600** on the selected historical test period.

### Alternative considered: Simple threshold model

A simple read-success threshold is easy to understand and useful as a benchmark.

However, it uses much less information than the machine-learning model and cannot combine multiple historical and telemetry signals effectively.

### Alternative considered: More complex model

More complex models could potentially improve predictive performance, but the challenge places strong importance on explainability and honest validation.

Random Forest provides a reasonable balance between predictive ability, feature importance, and operational explainability.

---

## 5. How do we know the ML model improves on the baseline?

### Decision

We evaluate the ML model using the same operational cost function as the supplied 3-sigma baseline:

* Unnecessary visit = €380
* Missed problem = €600
* Maximum selected gateways = 15 per week

The comparison is performed on historical future-like weeks where the actual following week's outcome is known.

### Result

| Approach               | Total cost |
| ---------------------- | ---------: |
| 3-sigma baseline       |   €144,920 |
| ML model               |   €111,600 |
| Improvement            |    €33,320 |
| Improvement percentage |      23.0% |

The ML model therefore reduced the evaluated operational cost by **€33,320**, or approximately **23.0%**, compared with the supplied 3-sigma baseline.

### Why this comparison is important

The model is not judged only by classification accuracy.

A prediction can have good accuracy but still produce expensive field decisions. The cost-based evaluation directly measures whether the model makes better use of the limited 15-visit capacity.

### Alternative considered: Accuracy as the main metric

We rejected accuracy as the primary evaluation metric because the operational consequences of false positives and false negatives are different.

A missed problematic gateway costs €600, while an unnecessary visit costs €380.

Therefore, total operational cost is more relevant to the actual field-team decision.

---

## Summary of the five decisions

| Decision                | Choice                                       |
| ----------------------- | -------------------------------------------- |
| Definition of a problem | Next-week meter-read success < 80%           |
| Visit decision          | Cost-aware risk ranking with top 15 selected |
| Part 2 focus            | Data Science + Machine Learning              |
| ML approach             | Random Forest classifier                     |
| Main evaluation         | Operational cost vs 3-sigma baseline         |

The overall design prioritizes **cost-aware field prioritization**, not guaranteed fault diagnosis. The model's output should be interpreted as a ranking of gateways most deserving of field attention.

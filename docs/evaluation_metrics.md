# Evaluation Metrics — What AmpOps Measures and Why

Status: current as of 2026-08-13 (branch `sachin/feature-fastapi`). This document is the
rationale record for the metric set. The metrics themselves are defined in exactly one
place — `src/ampops/training/bakeoff.py::evaluate` — and named in
`src/ampops/config.py::PRIMARY_METRIC`.

---

## 1. The metric set

| Metric | Role | Unit | Computed in |
|---|---|---|---|
| **MAPE** | **Primary / headline** — the number the project reports | dimensionless (fraction, e.g. `0.0302` = 3.02%) | `bakeoff.evaluate` via `sklearn.metrics.mean_absolute_percentage_error` |
| **RMSE** | Secondary — spike sensitivity, *and* the actual AutoML ranking metric (§4) | MW | `bakeoff.evaluate` via `sqrt(mean_squared_error)` |
| **MAE** | Tertiary — plain-language error size, and the RMSE/MAE ratio diagnostic | MW | `bakeoff.evaluate`, `np.mean(np.abs(...))` |

All three are returned together by a single function, so validation scoring, sealed-test
scoring, and any future backtest are guaranteed to compute them identically. There is no
second implementation anywhere in the repo — that was deliberate, and it is the reason
`tests/test_training.py::test_evaluate_returns_headline_metrics` can assert the contract
in one place.

---

## 2. The task these metrics have to score

Metric choice only makes sense against the prediction task, so stating it first:

- **Target:** `COMED_MW`, ComEd hourly regional electricity demand — a continuous,
  strictly positive regression target.
- **Horizon:** day-ahead. `config.HORIZON_HOURS = 24`; no feature may reference load
  newer than `t-24` (enforced by `tests/test_features.py::test_no_feature_sees_load_newer_than_horizon`).
- **Observed scale (actual data, not assumptions):** train mean 11,458.6 MW over 56,529
  rows; sealed test mean 11,298.1 MW over 8,591 rows. Full observed range across the
  dataset is **7,237–23,753 MW**.
- **Consumer of the forecast:** grid operations — day-ahead unit commitment, where the
  cost of being wrong is asymmetric and concentrated in peaks.

Three properties of that task drive everything below: the target **never approaches
zero**, the target's **absolute scale is meaningful** (MW is a unit an operator acts on),
and the **failure mode that matters is the peak**, not the average hour.

---

## 3. Why each metric was chosen

### 3.1 MAPE as the headline

**Chosen because it is scale-free and directly interpretable.** "The forecaster is off by
3% on average" is a sentence a non-technical stakeholder can act on without knowing what
a megawatt is or how large ComEd's load happens to be. An RMSE of 519 MW carries no
meaning to that reader without a second number to divide it by.

**It is safe here specifically because the denominator is safe.** MAPE's standard
objection is that it explodes or is undefined when actuals approach zero. Regional
electricity demand has a hard floor — the minimum ever observed in this dataset is
7,237 MW, roughly 64% of the mean. There is no near-zero hour, no zero hour, and
structurally cannot be one for a metropolitan load zone. The usual reason to reject MAPE
does not apply to this target, which is why it was chosen over sMAPE or MASE (§5).

**Its known bias is tolerable and, arguably, correctly aligned.** MAPE penalizes
over-forecasting less than under-forecasting for equal absolute error, because the
denominator is the actual. For a grid operator that skew is not obviously wrong:
under-forecasting demand is the more expensive direction operationally (it is the one
that leaves capacity uncommitted). This was not the reason MAPE was selected, but it
means the bias does not cut against the use case.

Recorded in `docs/AmpOps_Project_Context.md` §2.3 as a confirmed decision, and closed as
checklist item "Confirm MAPE/RMSE as the metric pair" — the metric pair predates the
model and was not chosen after seeing results.

### 3.2 RMSE as the secondary check

**Chosen because it is the metric that notices peaks.** RMSE squares errors before
averaging, so a handful of large misses move it far more than a uniform drizzle of small
ones. Peak hours — summer afternoons, cold snaps — are exactly where a demand forecast
earns or loses its value, and they are a small minority of rows. A model could improve
MAPE by getting the 3am hours slightly better while degrading the 4pm July hours, and
MAPE alone would report that as progress. RMSE is the guard against that trade.

**It stays in MW, which is the unit of the decision.** Reserve margins, unit commitment,
and capacity are all denominated in megawatts. RMSE gives a number the operations side can
compare to a real quantity; MAPE cannot.

### 3.3 MAE as the third metric

MAE is the "typical hour" error in MW — no squaring, so no peak amplification. It earns
its place for two reasons:

1. **Plain-language framing.** "Typically off by ~350 MW" is the most direct statement of
   error magnitude available.
2. **The RMSE/MAE ratio is a free diagnostic.** For the registered champion on the sealed
   test set: 519.3 / 351.3 = **1.48**. A ratio at 1.0 would mean errors are uniform;
   the further above 1.0, the more the total error is concentrated in a few large misses.
   1.48 says the residuals are meaningfully tail-heavy — the model's remaining error is
   the hard hours, not a flat offset. That is a property no single metric reports.

---

## 4. Selection metric ≠ headline metric (an honest caveat)

**MAPE is the headline, but RMSE is what actually ranks candidate models.**

`src/ampops/training/automl.py` configures the search with `sort_metric="RMSE"`, because
**MAPE is not a native H2O AutoML sort metric** — H2O's regression sort options do not
include it. So:

- H2O ranks its own leaderboard and picks `aml.leader` by **RMSE**.
- MAPE is then computed *post hoc* on the leader by `bakeoff.evaluate` and reported as
  the headline.

The practical consequence: the number the project leads with is not the number that chose
the model. In practice the two agree closely on this data (both penalize the same
residuals, differing in weighting), and RMSE is arguably the *better* selection criterion
for a peak-sensitive task — so this was accepted rather than worked around. It is
documented here rather than smoothed over because anyone reading `PRIMARY_METRIC = "mape"`
would otherwise reasonably assume MAPE drove selection. Closing the gap properly would
require a custom H2O metric or a post-leaderboard re-rank on MAPE; neither is implemented.

---

## 5. Metrics deliberately not used

| Not used | Why not |
|---|---|
| **R²** | Load has enormous, trivially-predictable daily and seasonal structure. R² against that variance runs high (>0.9) for almost any competent model and would flatter a mediocre forecaster. It does not discriminate between candidates here. |
| **MSE (raw)** | Same ranking as RMSE but in MW², a unit with no operational meaning. RMSE is MSE made readable; there is no reason to report both. |
| **sMAPE** | Exists to fix MAPE's near-zero denominator problem. That problem does not occur on this target (§3.1), so sMAPE would trade MAPE's direct interpretability for a defense against a risk that isn't present. |
| **MASE** | Would require committing to a naive baseline as the scaling denominator inside the metric. The baseline comparison is genuinely useful (§7) but is clearer stated as an explicit side-by-side than folded into a scale factor. |
| **Pinball loss / interval coverage** | These score *probabilistic* forecasts. The champion is a point forecaster with no predictive interval, so there is nothing for them to score. If uncertainty quantification is ever added, this is the metric family to add with it. |
| **Peak-hour-only MAPE** | Genuinely valuable for this use case and **not currently implemented** — listed as a gap in §8, not as a rejection. |

---

## 6. Where each metric is evaluated

Two evaluation surfaces, with strictly different jobs. The separation is the point:

| Surface | Span | Rows | What it decides | Enforced by |
|---|---|---|---|---|
| **Validation tail** | last 3 months of `train.parquet` (`bakeoff.VALIDATION_MONTHS = 3`) | ~2,200 | **Selects** the AutoML leader. Logged to MLflow as `mape` / `rmse` / `mae`. | `H2OAutoML(nfolds=0, validation_frame=...)` — chronological carve-out, never k-fold |
| **Sealed test** | final **12 months** (`config.TEST_MONTHS = 12`), 2017-08-03 → 2018-08-02 | 8,591 | **Decides nothing.** Reporting only. Logged as `test_mape` / `test_rmse` / `test_mae`. | `evaluate_on_test` runs *after* `register_champion`, in a separate Airflow task |

Two design rules make these numbers mean what they claim:

1. **No k-fold, ever.** `nfolds=0` with an explicit `validation_frame`. K-fold would
   shuffle time-ordered rows and leak future load into a fold's training data, inflating
   every metric above. Accepted cost: H2O skips Stacked Ensembles, which need
   cross-validated base predictions.
2. **The test set is scored strictly after promotion.** `run_h2o_automl` never opens
   `test.parquet`. `evaluate_on_test` reloads the *already-registered* version by URI and
   scores it; `tag_test_metrics` writes the result back as model-version tags. There is
   **no pass/fail gate** on the test metrics — deliberately, since a gate that rejected a
   model would turn the holdout into a selection signal and destroy its independence.

### Where the numbers land

- **MLflow run metrics:** `mape`, `rmse`, `mae`, `duration_seconds` (validation);
  `test_mape`, `test_rmse`, `test_mae` (added later by `tag_test_metrics`), so validation
  and test sit side by side on one run.
- **Registered model version tags:** `mape`, `rmse` at registration; `test_mape`,
  `test_rmse`, `test_mae` after holdout scoring — formatted `%.6f`. This makes the
  registry itself queryable for "how good is the thing currently serving traffic."

---

## 7. Results on record

Champion `ampops-demand-forecaster`, algorithm `drf` (H2O Distributed Random Forest).

**v1, full end-to-end run** (`docs/automl_implementation.md`):

| | MAPE | RMSE (MW) | MAE (MW) |
|---|---|---|---|
| Validation tail | 0.0425 | 745.2 | — |
| Sealed test | 0.0302 | 519.3 | 351.3 |

The currently-served **v5** carries validation MAPE 0.042276
(`docs/fastapi_serving_layer.md`). Serving sanity check on a single hour: 2018-07-15
14:00 predicted 15,792.4 MW against an actual 16,558.0 MW — APE 4.62%, consistent with
the validation MAPE.

**Test MAPE (3.02%) beating validation MAPE (4.25%) is not a general result.** The
validation tail is the last 3 months before the holdout, which lands in a different
seasonal position than the 12-month test span; the test window simply contains an easier
mix of hours. `docs/automl_implementation.md` already flags this as incidental to the
particular split. It should not be read as the model generalizing better than it fits.

### Skill against a naive baseline

**No baseline is computed anywhere in the pipeline** — this is a real gap (§8). The
figures below were computed directly against `data/processed/test.parquet` for this
document and are reproducible from it, but **nothing in `src/` or `dags/` produces them**.

The horizon-legal naive forecast for a 24h-ahead task is day-ahead persistence — predict
`load_lag_24h`, i.e. this hour yesterday. On the same 8,591 sealed-test rows:

| Forecast | MAPE | RMSE (MW) | MAE (MW) |
|---|---|---|---|
| **Champion (drf)** | **0.0302** | **519.3** | **351.3** |
| Persistence, `t-24h` | 0.0698 | 1,145.8 | 805.1 |
| Persistence, `t-168h` (same hour last week) | 0.1065 | 1,897.1 | 1,258.2 |
| 24h rolling mean | 0.1322 | 1,916.2 | 1,476.3 |
| 168h rolling mean | 0.1383 | 2,019.8 | 1,545.1 |

Against day-ahead persistence, the champion cuts **MAPE by 57%**, **RMSE by 55%**, and
**MAE by 56%**. This is the context that makes "3% MAPE" mean something: on its own it is
an unanchored number, and a naive lag on load data is already respectable at 7%.

---

## 8. What these metrics do not establish

Stated plainly, because each is a real limit on how far the reported numbers can be
pushed:

1. **The baseline comparison is not part of the pipeline.** §7's table is a one-off
   computation for this document. It should be a task in
   `dags/ampops_training_pipeline.py` logging `baseline_mape` etc. alongside the model's,
   so every run reports skill rather than a bare score.
2. **One split, not a backtest.** Every number comes from a single chronological
   train/validate/test partition. There is no rolling-origin or expanding-window
   evaluation, so the metrics carry no confidence interval and the seasonal-luck effect
   visible in §7 cannot be averaged out.
3. **No segment breakdown.** All metrics are computed over all hours pooled. Peak-hour
   MAPE, summer-vs-winter MAPE, and weekday-vs-weekend MAPE are unmeasured — and for a
   peak-sensitive use case, pooled MAPE is the least informative slice. RMSE and the
   RMSE/MAE ratio are the only current signal that tail errors exist at all.
4. **No production accuracy metric exists yet.** Nothing scores predictions against
   realized load. `ampops.forecasts` in Postgres (composite PK `(grid_id, target_ts,
   model_version)`) is the substrate a scoring job would join actuals onto, and ad-hoc
   `/predict` traffic is deliberately excluded from it so exploratory calls cannot
   pollute that future metric — but the job itself, the drift reports, and the retrain
   trigger are all still stubs.
5. **Replay mode caveat.** The daily forecast DAG runs against already-observed weather
   via `AMPOPS_SIMULATED_TODAY`. Any accuracy measured through that path benefits from
   perfect weather knowledge and would overstate live performance, where weather itself
   is a forecast.
6. **No uncertainty.** Point forecasts only, so nothing here says how confident the model
   is on any given hour.

---

## 9. Code map

| Concern | Location |
|---|---|
| Metric implementations | [src/ampops/training/bakeoff.py:30](src/ampops/training/bakeoff.py#L30) |
| `PRIMARY_METRIC`, `TEST_MONTHS`, `HORIZON_HOURS` | [src/ampops/config.py:107](src/ampops/config.py#L107), [src/ampops/config.py:144](src/ampops/config.py#L144) |
| Validation split constant (`VALIDATION_MONTHS = 3`) | [src/ampops/training/bakeoff.py:27](src/ampops/training/bakeoff.py#L27) |
| AutoML search + `sort_metric="RMSE"` | [src/ampops/training/automl.py:279](src/ampops/training/automl.py#L279) |
| Sealed-test scoring | [src/ampops/training/automl.py:336](src/ampops/training/automl.py#L336) |
| Metric tagging onto the registered version | [src/ampops/training/registry.py:102](src/ampops/training/registry.py#L102) |
| Chronological split | [src/ampops/features/split.py](src/ampops/features/split.py) |
| Leakage test (why the metrics are trustworthy) | [tests/test_features.py:127](tests/test_features.py#L127) |
| Metric-contract test | [tests/test_training.py:14](tests/test_training.py#L14) |

Related: [docs/automl_implementation.md](docs/automl_implementation.md) (search design and
run records) · [docs/AmpOps_Project_Context.md](docs/AmpOps_Project_Context.md) §2.3
(original decision record) · [README.md](README.md) (evaluation summary table).

# AmpOps — Feature Engineering Plan

**Owner:** Sachin · **Week 1 deliverable #2** · Companion doc: [`DATA_CLEANING.md`](DATA_CLEANING.md)

**Scope.** Everything between `data/processed/ampops_hourly.parquet` and the modelling matrix handed
to the bake-off. The clean table already carries the calendar block and the degree days (they are
part of the shared data contract); this doc covers the lag/rolling/interaction layer, the leakage
rules that govern all of it, and the split design the features have to survive.

Section references point at `notebooks/EDA.ipynb`, which is where each choice below was tested rather
than assumed.

---

## 1. The decision everything else hangs on: the forecast horizon

`COMED_MW` at t−1 explains **~99.5 %** of the variance at t. That is not a modelling triumph — it
means "the load an hour ago is a good guess for the load now". Whether we are *allowed* to use it
depends on the horizon we commit to:

| Horizon | Lags legally available | Realistic role |
|---|---|---|
| 1 hour ahead | t−1, t−2, … | trivially accurate, operationally useless |
| **24 hours ahead (day-ahead)** | **t−24 and older** | the industry-standard product; what utilities bid on |
| 1 week ahead | t−168 and older | planning, not dispatch |

**Decision: day-ahead, `HORIZON = 24`.** It makes the weather features meaningful (a day-ahead
forecast genuinely needs a weather forecast), keeps lag features legal, and is the horizon a grader
recognises.

> **The single rule that governs this document: no feature may use load information newer than
> t−24.** Every lag, every rolling window over the target, every aggregate. Weather is exempt —
> at prediction time we assume a weather forecast for the target hour, which is what a real day-ahead
> pipeline has. Assert this in code (`assert min_lag >= HORIZON`) rather than trusting review.

---

## 2. Feature families

Sizes below are the EDA's tested set: **12 calendar + 19 weather + 11 lag = 42 features.**

### 2.1 Calendar — 12 features

Produced in the cleaning stage, from **Chicago local time at the interval start** (see the cleaning
doc §1.6 for why the interval start matters).

```
hour, dow, month, doy, is_weekend, is_holiday
hour_sin, hour_cos, doy_sin, doy_cos, dow_sin, dow_cos
```

**Why cyclical encodings alongside the raw integers.** Hour 23 and hour 0 are adjacent; the integers
do not say so. Trees can carve the boundary with enough splits, but the sin/cos pair gives it for
free and the ridge baseline cannot express the daily cycle at all without them. Keep both — the raw
integer is what trees actually split on, the sin/cos is what makes the linear reference honest.

**Why holidays are non-negotiable.** Holidays behave like a Sunday dropped into a working week, and
the shortfall is biggest during business hours (§5.2). Without the flag the model over-forecasts
every one of them, and those errors are large, systematic, and exactly the kind that show up in a
demo. The flag is **not** derivable from day-of-week.

**The three cycles all have to be encoded** — daily (24 h), weekly (168 h), annual (~8,766 h). The
periodogram shows all three plus harmonics at 12 h and 8 h (§7). A model that ignores one of them
shows its shape in the residuals.

### 2.2 Weather — 19 features

Base variables kept from the 29 available:

```
temperature_2m, apparent_temperature, dew_point_2m, relative_humidity_2m,
wind_speed_10m, cloud_cover, pressure_msl, precipitation, snow_depth
```

Derived:

```
hdd, cdd                          fitted bases: 7.5 °C / 17.5 °C
temp_roll3, temp_roll24, temp_roll72
temp_lag24, temp_delta24
hdd_roll24, cdd_roll24
cdd_x_hour                        = cdd * sin(pi * hour / 24)
```

**Degree days encode the hockey stick.** The load–temperature response is V-shaped, not linear
(§9). Utilities split it into two half-linear terms; the bases were **fitted by 2-D grid search**
over daily-mean R² rather than borrowed from the US 65 °F convention, giving 7.5 °C heating /
17.5 °C cooling. The cooling arm is several times stronger than the heating arm — this is a
summer-peaking system.

**Why only three temperature-family variables survive from a large correlated block.** Twenty-nine
weather variables mostly measure overlapping physics. Correlation-distance clustering (§8) collapses
them into a handful of groups, and VIF on the candidate set puts the temperature proxies well above
10 — near-perfect linear dependence. Five collinear temperature proxies make importance scores
meaningless and add nothing to accuracy. Soil temperatures in particular are a lagged copy of air
temperature and are dropped.

**Thermal-inertia folklore was tested and it failed here (§9.2).** The standard prescription is
long trailing-mean temperature windows for building heat storage. Three independent views say no:
the lag-correlation scan peaks at 0, smoothing beyond ~3 hours *degrades* the correlation, and at the
daily level yesterday's CDD adds almost nothing once today's is in the model. So `temp_roll3` earns
its place; `temp_roll72` stays only as a cheap seasonal-context term, not as an inertia term. **Do
not spend the week building 120 h/168 h thermal windows.**

**`cdd_x_hour` is the one hand-built interaction.** The peak hour shifts by several hours between
winter and summer, and the load–temperature response changes shape by hour of day (§5.1, §9). Trees
get the interaction for free; the term exists so the ridge baseline is not embarrassed by it and so
the interaction is legible in the importance plot.

### 2.3 Load history — 11 features, all ≥ 24 h old

```
load_lag_24, load_lag_25, load_lag_26      # yesterday, same hour and its neighbours
load_lag_48, load_lag_72                   # 2 and 3 days back
load_lag_168, load_lag_336                 # same hour, 1 and 2 weeks back
load_roll24_lag24                          # shift(24).rolling(24).mean()  — yesterday's level
load_roll168_lag24                         # shift(24).rolling(168).mean() — the week's level
load_same_hour_4wk                         # mean(lag_168, lag_336)        — de-noised weekly profile
load_dailyrange_lag24                      # shift(24).rolling(24).max() - min()  — yesterday's swing
```

**Why these lags and not others.** The ACF over 14 days of lags shows the decay is not smooth — it
spikes at multiples of 24 and, more sharply, at multiples of 168 (§7). The PACF says which lags add
information *beyond* the shorter ones. Lag 168 is the strongest legal single predictor because it
matches hour-of-day and day-of-week simultaneously. Lags 25 and 26 are included because the
day-ahead constraint means we are extrapolating across a ramp, and the neighbouring hours pin the
slope.

`load_dailyrange_lag24` is a volatility proxy: the variance of load is itself seasonal — summer is
both higher and more volatile (heteroscedastic, §6) — and yesterday's swing is the cheapest available
signal for today's.

**Rolling windows must be `shift(HORIZON)` *then* `rolling(w)`, never the reverse.** `rolling(w)`
followed by `shift` leaks the current hour into the window. Write it once, in one helper, and test
it.

---

## 3. Leakage rules

1. **No load feature newer than t−24.** Enforced by assertion, verified by a test that checks each
   lag column against a manually shifted reference.
2. **Rolling before shifting is a bug.** See above.
3. **Fit any scaler/encoder on the training fold only**, inside the sklearn `Pipeline`, so
   cross-validation refits it per fold. Trees do not need scaling; the ridge baseline does, and that
   is where the leak would happen.
4. **The degree-day bases were fitted on the full history.** This is a mild in-sample leak. It is
   acceptable — they are physical constants of the building stock, not model parameters, and they are
   frozen in the metadata sidecar. Say so out loud in the presentation rather than hiding it.
5. **Drop, don't impute, the burn-in rows.** `load_lag_336` costs the first two weeks of history;
   `dropna` on the full feature set is the right handling. Imputing them fabricates target history.
6. **Weather is treated as known at prediction time.** This is the standard day-ahead assumption
   (a weather forecast exists), but it means our reported error is optimistic relative to a system
   fed real forecasts. Worth one sentence in the deck.

---

## 4. Split design

The context doc calls for a time-based split with the test set untouched. §4.1 of the EDA adds a
complication: **the series ends 3 August 2018**, so a trailing 3-month test window is *all summer* —
the hardest and most valuable season, but a score measured only there does not generalise to
February.

So report **both**:

- **Primary trailing holdout** — last 3 months, held out until production validation. The headline
  number, stated honestly as summer-only.
- **Rolling-origin CV** — `TimeSeriesSplit(n_splits=5, test_size=24*60)`, five expanding-window folds
  of 60 days spanning different seasons. **The spread across folds is the honest uncertainty on the
  headline number**, and it is the figure to put on the slide.

Never a random split: it leaks future information, and a real dispatch system only ever has the past.

Metrics: **MAPE primary** (scale-free, explains itself to a non-technical grader), RMSE and MAE
secondary, plus **bias** — a model that is right on average and wrong in a pattern is worse than the
metric suggests.

---

## 5. Baselines the model has to beat

A feature set is only justified if it beats the things that need no features at all:

| Baseline | What it tests |
|---|---|
| load 24 h ago | persistence at the horizon |
| load 168 h ago (same hour last week) | the weekly profile alone |
| hour × day-of-week climatology from train | the calendar alone |

Then the ablation that *is* the project's thesis:

| Feature set | Purpose |
|---|---|
| GBM, calendar only | how far the human calendar gets you |
| GBM, weather only | how far the physics gets you |
| GBM, calendar + weather | the coupling claim |
| GBM, calendar + weather + lags | the full day-ahead model |
| Ridge, full set | the linear reference point |

Neither weather-only nor calendar-only comes close to the pair — that contrast is the quantitative
version of this project's premise and belongs on a slide. Ridge stays in the bake-off as the
**reference point, not a candidate**: GBMs win by a wide margin because temperature and hour-of-day
interact, and a linear model cannot express that without explicit interaction terms.

Model: `HistGradientBoostingRegressor` (fast, no scaling needed, handles the interactions natively).
Each configuration is a separate MLflow run — that is how we satisfy the "AutoML" rubric item as an
explicit bake-off.

---

## 6. Validation of the feature set

Not "did it train" — did it *earn its place*:

- **Permutation importance** on the champion, grouped by family (load history / temperature family /
  calendar & other weather). Anything with importance indistinguishable from zero is a candidate for
  removal; a smaller feature set is a smaller serving contract and a smaller drift surface.
- **Residual ACF.** Spikes at 24 h mean the daily cycle is not fully captured. Ljung-Box at lags 24,
  48, 168 will almost certainly reject — on hourly load that is normal and mostly harmless for a
  point forecast, but it is the honest argument for adding lags rather than a clean bill of health.
- **Bias by hour, by month, by temperature bin.** A residual that is unbiased on average but tilted
  across the feature space means a missing feature or a missing interaction. This is where the next
  feature comes from, if there is time for one.
- **Worst-day inspection.** Plot the largest-error day against its temperature. Failure modes in load
  forecasting are hot afternoons; if the worst day is not one, something structural is wrong.

Stopping rule for the week: the feature set is done when the ablation table shows the coupling, the
residual bias plots are flat within noise, and CV spread is reported. Not when the MAPE stops
improving.

---

## 7. Implementation

Target module layout under `src/ampops/features/`:

```
build.py      make_features(df, horizon=24) -> (X, y, feature_names)
              calls the three family builders below, then drops burn-in rows
calendar.py   already produced upstream; this module only selects/encodes
weather.py    degree days, rolling temps, cdd_x_hour
lags.py       shift-then-roll helpers; the leakage assertions live here
sets.py       CAL_F, WX_F, LAG_F and the named FEATURE_SETS for the bake-off
```

`FEATURE_SETS` is the single source of truth for the ablation, the MLflow run names, and the API
request schema — define the lists once and import them everywhere.

**Tests** (`tests/test_features.py`):

1. Every lag column equals a manually computed `shift(k)` of the target, and `min(k) >= HORIZON`.
2. `load_roll24_lag24` at time t uses only values ≤ t−24 (construct a series where the last 24 hours
   are sentinel values and assert they never appear in the output).
3. `make_features` drops exactly the burn-in rows and no others.
4. Degree days are zero on the correct side of each base and linear on the other.
5. Cyclical encodings satisfy `sin² + cos² = 1` and place hour 23 adjacent to hour 0.
6. The feature-name list matches what the serving schema expects (guards the training/serving skew
   that the parquet sidecar exists to prevent).

---

## 8. Open items

- [ ] Decide whether a log-transformed target earns its place — worth one MLflow run, not a default.
      The right skew is real signal, and MAPE already down-weights large values.
- [ ] Prune the feature set once permutation importance is in; a 42-feature serving contract is
      larger than it needs to be.
- [ ] Confirm the final `FEATURE_SETS` names with Collin before MLflow runs start, so the run names
      in the deck match the code.

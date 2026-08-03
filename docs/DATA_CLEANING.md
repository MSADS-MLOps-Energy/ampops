# AmpOps — Data Cleaning Plan

**Owner:** Miguel · **Week 1 deliverable #1** · Companion doc: [`FEATURE_ENGINEERING.md`](FEATURE_ENGINEERING.md)

**Scope.** Everything between the two raw CSVs on disk and `data/processed/ampops_hourly.parquet` —
the single clean, hourly, gap-free table that every other workstream builds on. Feature construction
that happens *downstream* of this table (lags, rolling windows, interactions) belongs to the feature
doc; the calendar and degree-day columns are produced here because they are part of the shared data
contract.

Every step below is justified by evidence in `notebooks/EDA.ipynb` — the section references point at
the cell that proves it. Nothing here is defensive boilerplate; each check exists because the audit
found something, or because the failure it guards against is silent.

---

## 0. Inputs

| File | Rows | Grain | Native clock | Notes |
|---|---|---|---|---|
| `data/COMED_hourly.csv` | 66,497 | hourly, MW | **EPT — Eastern Prevailing Time, DST-aware, hour-ending** | Target. Ships **unsorted**. |
| `data/open-meteo-41.86N87.65W179m.csv` | 87,648 hourly + a second daily block | hourly | **fixed UTC−5, no DST** (header claims `America/Chicago` — it lies) | 29 weather variables, 2010-01-01 → 2019-12-31. |

Output: `data/processed/ampops_hourly.parquet` (+ `ampops_hourly.meta.json` sidecar).

---

## 1. The cleaning steps, in order

### 1.1 Parse the Open-Meteo file as *two* concatenated exports — do not hard-code the split

The file contains an hourly export, then a **second header row**, then a daily re-export of the same
period. Locate the split by scanning for lines beginning `time,` rather than by row index, because a
re-download with a different date range moves that index and a hard-coded slice would silently
truncate or contaminate the frame.

- Read block 1 (hourly) → the feature table.
- Read block 2 (daily) → **discarded.** This closes the open decision in §8 of the context doc: the
  hourly block already carries everything the daily aggregates would give us, and daily degree-days
  are cheaper to derive from hourly temperature than to reconcile from a second export with its own
  clock ambiguity.
- Strip the trailing header line that `read_csv` pulls in as a data row (`time` not matching
  `^\d{4}-\d{2}-\d{2}`).
- Split units out of the column names (`temperature_2m (°C)` → name `temperature_2m`, unit `°C`) and
  keep the unit map — the API schema and the Evidently report both need it.
- Coerce every non-`time` column with `pd.to_numeric(errors="coerce")` so a stray text token becomes
  a NaN we can count, not an `object` column that silently poisons every downstream numeric op.

### 1.2 Sort ComEd chronologically before touching it

`COMED_hourly.csv` is **not** stored in time order (§2.1). Any `.diff()`, `.shift()`, `.rolling()`
or `.interpolate()` on the file as delivered produces garbage without raising. Sort by `Datetime`
with a **stable** sort — the fall-back duplicates must retain their on-disk order, because that order
is what identifies which copy is EDT and which is EST.

### 1.3 Reconcile the two clocks — the highest-risk step in the project

The two files are aligned in winter and **one hour apart all summer**. Merging on the raw timestamp
columns raises no error; it just corrupts half of every year. The EDA establishes both conventions
from the data itself rather than from documentation:

| Evidence | Conclusion |
|---|---|
| `03:00` missing every spring, `02:00` duplicated every fall (§2.1) | ComEd is a DST-observing local clock, **hour-ending** |
| 87,648 perfectly continuous hours, no DST fingerprint (§2.2) | Weather is a **frozen offset** |
| Mean June–July ET₀ peaks in the bucket containing solar noon 12:51 (§2.2) | That offset is **UTC−5**, not UTC−6 |

**Rule: join on UTC. Derive calendar features from `America/Chicago`. Never join on a local clock.**

**ComEd → UTC.** The hour-ending label is *not* a wall-clock instant you can localize directly. On a
fall-back day the wall clock that repeats is `01:00`, not `02:00`; hand `tz_localize` the label
`02:00` twice with an `ambiguous` flag and pandas ignores the flag, producing a duplicate join key
and a hole an hour earlier. The correct procedure follows from what the label means — `02:00`
hour-ending denotes the interval `01:00 → 02:00`:

1. subtract one hour to get the interval **start**;
2. `tz_localize("America/New_York", ambiguous=<first occurrence = EDT>, nonexistent="shift_forward")`;
3. `tz_convert("UTC")`, then add the hour back.

Outside the transitions this is identical to localizing the label; on the four fall-back days it is
the difference between a clean key and a corrupt one. It changes ~0.006 % of rows — which is exactly
why nobody catches it by eye.

**Weather → UTC.** Apply the constant offset from the file's own metadata row
(`utc_offset_seconds // 3600` = −5), then localize to UTC. No DST logic, because the file has none.

**Acceptance test (must be automated):** after conversion, `ts_utc` has **zero duplicates**, and the
only missing hours on a complete UTC grid are the six real ones in §1.5. This single assertion
validates the entire conversion.

### 1.4 Merge

Inner join on `ts_utc`. ComEd's stamp is hour-*ending*, Open-Meteo's instantaneous fields are values
*at* the stamp, so the join pairs each hour of energy with the weather at the close of that hour —
confirmed empirically in §3.3 by scanning candidate shifts and scoring each on the correlation of
first differences (a phase detector). The join drops the weather rows outside the ComEd window.

**Usable window: 2011-01-01 → 2018-08-03, ≈66,503 hours.** Note the end date — the brief assumed
"2011–2018"; the series actually stops on **3 August 2018**, which is why the trailing holdout is
summer-only and why §10.2 of the EDA insists on rolling-origin CV alongside it.

### 1.5 Reindex onto a complete UTC grid and impute

Reindex to `date_range(min, max, freq="h", tz="UTC")` so that downstream `shift(24)` genuinely means
"24 hours ago" rather than "24 rows ago". This is a correctness requirement for every lag feature,
not a cosmetic one.

Of the fifteen timestamp anomalies in the raw file, **nine were artefacts of the clock and six are
real**: the repeated fall-back hour PJM dropped outright in 2011, 2012 and 2013. Impute those six
with `interpolate(method="time")` and flag them with a boolean **`is_imputed`** column that ships in
the parquet — imputed rows must be excludable from evaluation and from the Evidently reference set.

Six hours out of 66,503 (0.009 %) is small enough that the imputation method does not matter; the
flag matters.

### 1.6 Calendar columns — from Chicago local, at the interval **start**

`COMED_MW` stamped `2015-01-01 00:00` is the energy consumed between 23:00 and midnight on
**December 31st**. Attributing it to Jan 1 puts New Year's Eve evening load on New Year's Day and
blurs every holiday and day-of-week effect by one hour. So: convert `ts_utc` to `America/Chicago`,
subtract one hour, and derive `year, month, day, hour, dow, doy, week, is_weekend, season` plus
`is_holiday` (`USFederalHolidayCalendar`) and `is_offday` from that.

Keep a naive copy (`local`) for grouping and plotting, but **never use the local clock as an index**
for `rolling`/`STL`/`asfreq` — it skips and repeats at DST. Analysis series index on UTC.

### 1.7 Degree days

`hdd = max(base_h − T, 0)`, `cdd = max(T − base_c, 0)`, with the bases **fitted from the data**
(§9.1) rather than borrowed from the US 65 °F convention: a 2-D grid search over both bases
maximising daily-mean R² gives **base_h = 7.5 °C, base_c = 17.5 °C**. These are properties of
Chicago's building stock; they belong in the cleaning config, and they are recorded in the metadata
sidecar so the serving container cannot drift from the training definition.

---

## 2. Quality gates — what the pipeline must assert on every run

These are pass/fail checks, not exploratory statistics. Each one currently passes; the point is to
notice the day one stops. They double as the **numeric basis for the Pydantic validators on the
FastAPI service and the reference bounds in the Evidently report**, so define them once, in
`src/ampops/data/quality.py`, and import them in all three places.

### 2.1 Structural

| Check | Expected |
|---|---|
| Duplicate `ts_utc` | 0 |
| Missing hours on the UTC grid | 0 after reindex |
| `is_imputed` count | 6 (alert if it grows) |
| Total nulls in the merged frame | 0 |
| Row count | ≈66,503 |
| Coverage by year × month | full months except 2018 (7 months, no autumn/winter) |

### 2.2 Target plausibility

Load in a large balancing area is bounded — strictly positive, with a floor from always-on demand
and a ceiling from generation capacity:

- non-positive values → 0
- below 1,000 MW → 0 (implausible floor)
- above 30,000 MW → 0 (implausible ceiling)
- non-finite → 0
- **stuck-meter check:** longest run of identical consecutive MW readings. Runs of 2 are rounding
  coincidences; a long flat run is a frozen sensor.
- **Δ-envelope:** the hour-over-hour change distribution is tight. Its ±99.999th percentile is the
  threshold a drift injector has to beat to be realistic, and the natural alarm bound for the
  monitoring stage.

The right skew in the target is **real signal** (summer peaks), not contamination — do not
winsorise it. A log target is worth one experiment, but MAPE already down-weights large values.

### 2.3 Weather range checks (Chicago-specific)

Anything outside these is an ingestion bug, not weather:

| Variable | Plausible range |
|---|---|
| `temperature_2m` | −40 … 45 °C |
| `dew_point_2m` | −45 … 32 °C |
| `apparent_temperature` | −55 … 55 °C |
| `relative_humidity_2m`, all `cloud_cover*` | 0 … 100 % |
| `precipitation`, `rain` | 0 … 120 mm |
| `snow_depth` | 0 … 3 m · `snowfall` 0 … 80 cm |
| `pressure_msl` | 940 … 1070 hPa · `surface_pressure` 920 … 1050 hPa |
| `wind_speed_10m` | 0 … 160 km/h · `wind_speed_100m` 0 … 200 · `wind_gusts_10m` 0 … 200 |
| `wind_direction_*` | 0 … 360 ° |

Also track, without failing the run: **low-cardinality columns** (near-constant → dead weight for the
model) and **exact-zero fraction** (sparse variables like `precipitation`, `snowfall`, `snow_depth`
need care in any scaling or imputation step — do not mean-impute a variable that is zero 90 % of the
time).

---

## 3. Output contract

`data/processed/ampops_hourly.parquet` — one row per hour, sorted by `ts_utc`:

| Group | Columns |
|---|---|
| Keys | `ts_utc` (tz-aware UTC — **the join key**), `local` (naive Chicago, interval start) |
| Target | `COMED_MW` (MW), `is_imputed` (bool) |
| Weather | the 29 Open-Meteo hourly variables, original units, unit map in the sidecar |
| Calendar | `year, month, day, hour, dow, doy, week, is_weekend, is_holiday, is_offday, season` |
| Cyclical | `hour_sin/cos, doy_sin/cos, dow_sin/cos` |
| Derived weather | `hdd, cdd` |

The `ampops_hourly.meta.json` sidecar records the row count, the UTC period, the target name, the
join key, both source timezones, the calendar-derivation rule, the fitted HDD/CDD bases, the
imputed-hour count, and the recommended horizon. **The serving container reads its feature
definitions from this file** — that is what stops training and inference from silently disagreeing.

---

## 4. Implementation & testing

Target module layout under `src/ampops/data/`:

```
loaders.py    read_comed()            -> sorted raw frame
              read_openmeteo_blocks() -> (hourly, daily, units); finds header rows by scan
timezones.py  comed_to_utc()          -> the interval-start localize; the one to unit-test hardest
              weather_to_utc()
merge.py      build_hourly()          -> join, reindex, interpolate, is_imputed
calendar.py   add_calendar()          -> Chicago-local, interval-start; holidays; cyclical
quality.py    BOUNDS, run_checks()    -> shared with serving + monitoring
```

**Unit tests that must exist** (`tests/test_data.py`) — these are the ones that catch silent failure:

1. `comed_to_utc` on a fall-back day produces **two distinct UTC instants** for the duplicated
   `02:00` label, and localizing the label directly does not.
2. `comed_to_utc` on a spring-forward day produces a **continuous** UTC sequence with no gap.
3. Round-trip: the UTC series reindexed onto a complete grid has zero duplicates and exactly six
   NaN hours before interpolation.
4. Calendar attribution: the row stamped `2015-01-01 00:00` UTC-derived-local resolves to
   **Dec 31**, not Jan 1.
5. The Open-Meteo splitter returns two blocks with the right row counts on a fixture that has the
   header rows at different positions than the real file.
6. `run_checks` fails when handed a frame with a negative MW value, an out-of-range temperature, or
   a duplicate key.

Airflow task ordering mirrors §1: `load → split_blocks → to_utc → merge → reindex_impute →
calendar → degree_days → quality_gate → write_parquet`, with `quality_gate` as a hard failure that
stops the DAG rather than a warning.

---

## 5. Handoff notes for the other workstreams

- **Features (Sachin):** consume the parquet, never the CSVs. The grid is complete and regular, so
  `shift(k)` is safe and means hours.
- **Monitoring (Minhae):** the train/test covariate shift is a *free, genuine* drift example —
  because the series ends in August, a winter reference vs. a summer current window differs in
  temperature distribution before anyone injects anything. The §2.2 Δ-envelope and §2.3 bounds are
  the basis for the out-of-bounds and sensor-dropout scenarios.
- **Serving (Sachin):** the §2.3 table is the Pydantic validator spec. The unit map in the sidecar is
  the request schema documentation.
- **Modelling (Collin):** `is_imputed` rows should be excluded from evaluation windows.

# AmpOps — Data Cleaning & Versioning Plan

**Purpose of this document:** trace the path from the two raw source files in `data/raw/` to the `v2` training-ready dataset described in README's "Data Ingestion & Versioning" pipeline stage. Part 1 documents what `notebooks/01_join_and_eda.ipynb` already implements. Part 2 is the forward-looking plan for the work that notebook does *not* cover. Part 3 formalizes the `v1`/`v2` DVC versioning story. For the underlying data-quality reasoning (why the DST realignment is needed, why the weather file needs splitting, etc.), see `docs/AmpOps_Project_Context.md` §2.2–2.3 rather than this document — it isn't re-derived here.

---

## 1. Raw → notebook-cleaned (already implemented)

This stage is implemented in `notebooks/01_join_and_eda.ipynb`. It takes the two files in `data/raw/` — `COMED_hourly.csv` and `open-meteo-41.86N87.65W179m.csv` — and produces a single deduplicated, DST-corrected, joined hourly dataset trimmed to the confirmed overlap window.

| Step | What happens | Why (reference) |
|---|---|---|
| 1. Weather dual-export split | `open-meteo-41.86N87.65W179m.csv` is split at the second header row (row 87,654); only the hourly-grain block (rows 1–87,653) is kept going forward. The trailing daily-aggregate re-export is dropped from the working dataset. | Project Context §2.2.1 |
| 2. COMED chronological sort | `COMED_hourly.csv` is **not** stored in chronological order on disk — rows are grouped into per-day blocks that run in **descending date order within each year** (e.g. `2011-12-31`, then `2011-12-30`, … before rolling to the next year). The notebook sorts by `Datetime` ascending before any downstream step. Any other consumer of the raw file must do the same sort first — do not assume row order implies time order. | Verified directly against the raw file |
| 3. Fall-back duplicate averaging | The 4 duplicated fall-back hours in COMED's local-time column (each with two distinct `COMED_MW` readings) are collapsed to one row per timestamp by **averaging** the two readings. | Project Context §2.3 ("Duplicate fall-back timestamps") |
| 4. DST realignment | COMED's DST-aware local Chicago time is converted onto the weather file's fixed UTC-5 grid (no DST shift), so both series share one consistent offset before joining. The 11 spring-forward hours that don't exist in COMED's local clock remain absent at this stage (see Part 2 for how they're handled downstream). | Project Context §2.2.2 |
| 5. Overlap-window trim | Both series are trimmed to **2011-01-01 through 2018-08-03**, COMED's confirmed range. | Project Context §2.3 |
| 6. Join + sanity check | The realigned COMED series and the hourly weather block are joined on the shared timestamp grid. An hour-of-day sanity check is run post-join to confirm the DST conversion didn't silently misalign the two series. | Project Context §2.2.2, §8 (open gap: "Confirm the DST-realignment approach ... holds up under the hour-of-day sanity check once implemented") |

**Output:** a joined hourly dataset (e.g. `data/processed/joined_hourly.parquet`) covering `COMED_MW` plus the hourly weather fields, deduplicated and DST-corrected, but **without** lag or rolling-window features. This output is the input to Part 2.

---

## 2. Notebook-cleaned → final processed/training-ready dataset (planned, not yet built)

README's pipeline description defines `v2` as "deduplicated, DST-corrected, **with engineered lag features `t-1`, `t-24`, `t-168` and 24-hour rolling means**." The join notebook's output satisfies the first half (dedup + DST correction) but not the feature-engineering half. This section is the plan to close that gap.

### 2.1 Scope

| Item | Plan |
|---|---|
| **Input** | The join notebook's output — the deduplicated, DST-corrected, joined hourly dataset (e.g. `data/processed/joined_hourly.parquet`). |
| **Lag features** | `COMED_MW` lagged at `t-1`, `t-24`, `t-168` hours (previous hour, previous day same hour, previous week same hour) — these match the lags named in README and align with the load-forecasting intuition already recorded in Project Context §2.3 ("Feature engineering" row: lag features t-24h, t-168h). Worth also considering a `t-1` and `t-24` lag on the primary weather driver (e.g. `temperature_2m`), since same-hour-yesterday and prior-hour weather can carry predictive signal for demand — flagged here as a candidate, not a locked decision. |
| **Rolling features** | 24-hour rolling mean of `COMED_MW` (per README). A 24-hour rolling mean of temperature is a natural analog if the team wants richer weather-coupling signal; same caveat as above — candidate, not locked. |
| **Calendar features** | Not part of README's `v2` definition, but Project Context §2.3 ("Feature engineering" row) separately calls for hour-of-day, day-of-week, month, and holiday-flag features. These can be added in the same step since they're cheap and don't depend on any additional decision — worth confirming with whoever owns feature engineering (per the Task Management table in Project Context §9, this is Sachin's deliverable) whether they land here or in a separate step. |
| **Output** | A training-ready file in `data/processed/`, e.g. `data/processed/train_features.parquet` (or similarly named to distinguish it from the join notebook's intermediate `joined_hourly.parquet`). This is the file that should feed the Week 2 Airflow/MLflow pipeline. |
| **Where this logic lives** | Per `CLAUDE.md`, the repo is pre-implementation — no `src/` layout or feature-engineering module exists yet. Two reasonable homes, either is consistent with current repo conventions: (a) a follow-up notebook, e.g. `notebooks/02_feature_engineering.ipynb`, mirroring the pattern already established by `01_join_and_eda.ipynb`; or (b) a script under a future `src/ampops/features/` module (matching the `src/ampops/{data,features,training,serving,monitoring}` layout Project Context §6 recommends), once the team moves past notebook-stage EDA and into the Week 2 Airflow DAG. Given the Week 1 deliverable is explicitly "Feature Engineering Final" (Project Context §9), a notebook is the lower-friction choice for now, with the DAG in Week 2 reimplementing/calling the same logic as a script. |

### 2.2 Open questions — do not assume an answer

- **Residual missing timestamps from spring-forward gaps.** After DST realignment, the 11 spring-forward hours per affected year that don't exist in COMED's local clock will surface as missing timestamps (or NaN rows) in the joined series. Once lag/rolling features are computed, these gaps will also propagate forward into the lag columns for up to 168 hours after each gap. The team has **not** decided how to handle this — options on the table:
  - Leave as `NaN` and let the model/training pipeline handle missing values natively (e.g. tree-based models tolerate NaN; this is arguably the most defensible option given how small the gap count is — 11 hours/year is a rounding error against ~65K rows).
  - Forward-fill the missing hour from the prior hour's reading.
  - Interpolate (e.g. linear) between the surrounding hours.
  This should be decided and recorded in Project Context's Decision Log before the feature-engineering step is implemented, since it affects both the lag/rolling calculations and the eventual train/test split.
- **Which additional lag/rolling columns beyond `COMED_MW`, if any.** Flagged as a candidate above but not locked — needs a decision from whoever owns feature engineering.
- **Calendar/holiday features' home.** Whether these ship in the same feature-engineering step as the lag/rolling features, or as a separate step, is unresolved.

---

## 3. Data versioning plan (plan only — no DVC commands run or config created)

README commits to DVC tracking two dataset versions, `v1` (uncurated) and `v2` (deduplicated, DST-corrected, feature-engineered). This section is a written recommendation for what each version should cover and when to cut a new one. **No `dvc init`, `dvc add`, or other DVC command has been run as part of writing this plan** — this is scope definition only, for whoever picks up DVC setup to execute.

| Version | Covers | Source of truth | When to tag/cut a new version |
|---|---|---|---|
| **`v1`** | The raw, uncurated files as delivered: `data/raw/COMED_hourly.csv` and `data/raw/open-meteo-41.86N87.65W179m.csv`, unmodified — including the weather file's dual-export structure and COMED's out-of-order rows and DST duplicates/gaps. | `data/raw/` | Tracked once, at the point the raw files are considered final inputs (they already are, per Project Context §2.1 — both are "Uploaded"). Re-tag only if the upstream Kaggle/Open-Meteo source files themselves change (not expected during this project). |
| **`v2`** | The fully deduplicated, DST-corrected, **feature-engineered** dataset — i.e. the Part 2 output (`data/processed/train_features.parquet` or equivalent), not the Part 1 intermediate. Matches README's `v2` definition verbatim (dedup + DST-correct + lag features `t-1`/`t-24`/`t-168` + 24h rolling means). | `data/processed/` | Cut a new `v2` version any time the cleaning or feature-engineering logic changes — e.g. the spring-forward-gap decision (§2.2 above) is finalized, additional lag/rolling columns are added, or a bug is found in the DST realignment. Each new version should be tagged so MLflow runs can record which data version they trained against. |

**Where the join notebook's output fits:** `notebooks/01_join_and_eda.ipynb`'s output (`data/processed/joined_hourly.parquet`) is an **intermediate artifact between `v1` and `v2`, not `v2` itself** — it has the dedup/DST-correction half of the `v2` definition but is missing the lag/rolling feature-engineering half (Part 2). It does not need its own DVC-tracked version under the `v1`/`v2` scheme; treat it as a working file that Part 2's feature-engineering step consumes and supersedes. If the team wants intermediate reproducibility (e.g. to avoid re-running the join notebook every time feature logic changes), it can be DVC-tracked separately as a pipeline stage output rather than as a third top-level dataset version — a decision to make once DVC is actually being set up, not now.

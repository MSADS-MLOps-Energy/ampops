# AmpOps — Project Context & Build Plan

**Purpose of this document:** context for building any phase of this project. It captures the decisions that are locked, the recommendations being made now, and the ones that still need a human call. Update the "Decision Log" as choices firm up.

---

## 1. Project Summary

AmpOps is a weather-coupled electricity demand forecasting pipeline, built as the final project for an MLOps course. The pipeline must go end-to-end: raw data → orchestrated training → tracked experiments → containerized inference API → live monitoring/drift detection, matching the course's four required lifecycle stages.

- **Team:** 4 people, roles TBD as of team meeting (today)
- **Build deadline:** August 17 (repo/pipeline complete)
- **Presentation:** August 19, 10–15 min, each member states their individual engineering contribution
- **Timeline available:** ~3 weeks from today (July 28)

---

## 2. Data Layer

### 2.1 Sources
| Dataset | Contents | Coverage | Status |
|---|---|---|---|
| Open-Meteo weather export | Hourly weather (temp, humidity, dew point, wind, pressure, cloud cover, soil temp/moisture, etc.) for a Chicago-area point (41.86°N, −87.65°W, elev. 179m) | 2010-01-01 through 2019-12-31 | Uploaded, needs cleanup (see 2.2) |
| Kaggle "Hourly Energy Consumption" — `COMED_hourly.csv` | Hourly load in MW for the ComEd (Commonwealth Edison, Chicago-area utility) balancing region | Confirmed: 2011-01-01 01:00 through 2018-08-03 00:00, 66,497 rows | Uploaded, needs cleanup (see 2.2) — this is the real training overlap window, not weather's full 2010–2019 range |

The pairing is a good fit: ComEd's service territory is the Chicago metro area, so the weather station coordinates line up with the demand region.

### 2.2 Known data quality issues — resolve before any pipeline work

#### 2.2.1 Weather file is two concatenated exports
The uploaded weather CSV is actually **two Open-Meteo exports concatenated in one file**:
- Rows 1–87,653: hourly grain, 2010–2019
- Row 87,654 onward: a **second header row**, then a daily-aggregated re-export of the *same* 2010–2019 period (mean/max/min temp, sunrise/sunset, daylight duration, etc.)

**Decision made (recommended):** use the hourly grain only — it matches the hourly granularity of `COMED_hourly.csv` and the course's premise of hourly-coupled demand. Split the file into two clean CSVs on ingestion (or just slice at the header row) and treat the daily block as scrap, or optionally mine it later for daily-aggregate features (e.g., daily degree-days) if the team wants richer calendar features.

#### 2.2.2 Timezone/DST mismatch between the two sources — the critical one
The weather export's metadata row (`utc_offset_seconds=-18000`, `timezone_abbreviation=GMT-5`) confirms it is stamped on a **fixed UTC-5 offset that never shifts for daylight saving**. `COMED_hourly.csv`'s `Datetime` column, by contrast, is **local Chicago clock time that observes real US DST transitions**: it has 11 missing spring-forward hours (e.g. `2011-03-13 03:00`) and 4 duplicated fall-back hours, each with two different `COMED_MW` readings (e.g. `2014-11-02 02:00`) — and the exact pattern is inconsistent year to year.

**Implication:** a naive string-match join on the timestamp columns will silently misalign the two datasets by an hour for roughly 7 months of every year (whenever Chicago is on daylight time). This must be resolved — by converting COMED's DST-aware local time onto the same fixed-offset grid the weather data uses — before any join, and the realignment must be validated with an hour-of-day sanity check afterward, not assumed correct.

### 2.3 Open decisions — data layer
| Decision | Recommendation | Why |
|---|---|---|
| Target variable | `COMED_MW`, regression | Given directly by the dataset |
| Overlap window | Confirmed: trim both datasets to 2011-01-01–2018-08-03 (COMED's range) | Weather data (2010–2019) and demand data don't fully overlap; training on non-overlapping rows wastes data and risks silent misalignment |
| Duplicate fall-back timestamps | Average the two COMED_MW readings at each of the 4 duplicated hours (recommended over keep-first) | Both readings are real data for an ambiguous clock hour; averaging is defensible and keeps the series continuous. |
| Evaluation metric | **MAPE as the primary/headline metric**, RMSE as a secondary check | MAPE is scale-free and easy to justify to non-technical graders ("X% average error"); RMSE penalizes large misses (useful since demand spikes are the failure mode that matters operationally) |
| Train/test split | **Time-based split, not random** — e.g., train on all but the final 2–3 months, test on the tail, consistent with the course's requirement that the test set stay isolated until production validation | Random splits leak future information into training for time series; a real demand-forecasting deployment only ever has the past to work with |
| Feature engineering | Calendar features (hour-of-day, day-of-week, month, holiday flag), lag features (t-24h, t-168h), rolling means, and the weather features already present | Non-linear demand-weather coupling is explicitly the framing of this project — calendar effects (weekday/weekend, season) usually explain as much variance as weather alone in load forecasting |
| Model family | Gradient-boosted trees (XGBoost or LightGBM) as the primary candidate, linear regression as a baseline for comparison | Handles non-linear feature interactions well, trains fast (important for a 3-week timeline and repeated retraining during pipeline debugging), and is easy to log/version with MLflow |
| "AutoML" requirement | Satisfy it with a **small, explicit model bake-off** (Linear → RandomForest → XGBoost/LightGBM) logged as separate MLflow runs, rather than standing up a full AutoML framework | The course lists AutoML as an example ("e.g."), not a hard requirement — a clean bake-off with tracked metrics meets the spirit (comparing algorithms) with far less setup risk in the available time |

---

## 3. Recommended Tool Stack

Given a 3-week timeline and a beginner-to-intermediate-comfort team, the priority is **tools that are fast to stand up and hard to misconfigure**, not the most "impressive"-sounding options.

| Stage | Recommendation | Why this over the alternative |
|---|---|---|
| Orchestration | **AirFlow** (open-source, local agent) | Airflow requires standing up a webserver, scheduler, and metadata DB — real operational overhead for a 3-week class project. |
| Experiment tracking + registry | **MLflow** (local or self-hosted, not the managed service) | Free, no account/API key friction (unlike W&B), and the Model Registry is built in — one tool covers both tracking and registry requirements. `mlflow ui` gives you the dashboard screenshot the rubric wants with zero extra setup. |
| Deployment | **Docker + FastAPI** | FastAPI over Flask: automatic OpenAPI/Swagger docs (`/docs`) give you a working interactive demo for free, native Pydantic request validation (which also makes injecting "drifted"/malformed requests for the stress test trivial), and async support if inference needs to scale. BentoML adds a real learning curve for marginal benefit at this scope. |
| Monitoring & drift | **EvidentlyAI** | Purpose-built for exactly this rubric item — one Python call generates an HTML data-drift/model-performance report. Standing up Prometheus + Grafana is a multi-day infra project by itself; Evidently gets you a presentable dashboard in hours. |

---

## 4. Suggested Team Workstreams (4 people)

The four required lifecycle stages map cleanly onto four workstreams — confirm/adjust in today's team meeting:

1. **Data & Features** — clean and join the two source datasets, resolve the overlap window, build the feature pipeline, own the train/test split logic.
2. **Pipeline & Experimentation** — Prefect flow for orchestration, MLflow tracking, the model bake-off, and registering the winning model to the Model Registry.
3. **Deployment** — Dockerfile, FastAPI inference service, request/response schema (Pydantic), containerized smoke tests.
4. **Monitoring & Drift** — Evidently baseline report against the clean test set, design and inject the corrupted/drifted dataset, capture before/after evidence for the presentation.

Each workstream should agree on the **data contract** (schema of the joined dataset, and the API request/response schema) before splitting up, so integration in week 3 isn't a surprise.

---

## 5. Drift Simulation — concrete scenarios to implement

The rubric asks for artificially corrupted data that mimics real field failure modes. Concrete options, worth splitting across a few for a richer demo:

- **Out-of-bounds smart-meter readings:** inject impossible values into `COMED_MW` (negative load, or values 10–100x the historical max) to mimic a meter fault.
- **Column swap:** swap `temperature_2m` and `dew_point_2m` (or similar correlated-but-distinct columns) to mimic a data-pipeline mapping bug.
- **Schema drift:** drop a column the model expects, or rename it, to mimic an upstream schema change.
- **Sensor dropout:** null-flood a subset of rows in one weather field to mimic a failed smart meter/sensor feed.

Feed each scenario through the deployed API and capture what Evidently flags (or doesn't) — the contrast between "silently wrong prediction" and "flagged by monitoring" is the strongest part of the demo.

---

## 6. Repository & Engineering Practices

- **Structure:** `src/`-layout Python package (e.g. `src/ampops/{data,features,training,serving,monitoring}`), not a flat script dump — makes the "modular, professional" repo requirement easy to satisfy and gives each workstream a clear home.
- **Environment/deps:** `requirements.txt` or `pyproject.toml`, pinned versions, plus the `Dockerfile` for the deployment stage.
- **Testing:** unit tests (pytest) for the feature-engineering functions and the API request/response validation at minimum — doesn't need to be exhaustive, but graders and Claude Code both benefit from a `tests/` folder that actually runs.
- **Git workflow:** one branch per workstream/feature, PRs into `main` even if it's just self-review given the team size, so the commit history reflects each person's "distinct engineering contribution" (the rubric explicitly asks each member to state theirs).
- **README:** must let a stranger reproduce the whole pipeline locally — this is graded explicitly, so treat it as a deliverable in its own right, not an afterthought in the last two days.

---

## 7. Timeline (today: July 28 → Aug 19)

| Window | Focus |
|---|---|
| Jul 28 – Aug 3 (Wk 1) | Repo scaffold, roles locked, data cleaned/joined, EDA, target/metric/split finalized, each workstream stubs its interface against the shared data contract |
| Aug 4 – Aug 10 (Wk 2) | Prefect flow + MLflow bake-off running end-to-end; FastAPI+Docker skeleton serving the registered model; Evidently baseline report against clean test data |
| Aug 11 – Aug 16 (Wk 3) | Full integration, drift injection + verification, README finalized, repo cleanup |
| Aug 17 | Repo/pipeline submission deadline |
| Aug 18 | Slide deck build, rehearse individual-contribution talking points |
| Aug 19 | Presentation |

---

## 8. Open Gaps — needs a decision before or during Week 1

- [X] Exact overlap window between the two datasets — confirmed: 2011-01-01 through 2018-08-03 (`COMED_hourly.csv`'s range)
- [X] Role assignment across the 4 workstreams above (today's meeting)
- [X] Confirm MAPE/RMSE as the metric pair, or substitute if the team prefers something else
- [ ] Decide whether to keep any daily-aggregate weather features or discard that block entirely
- [X] Confirm Prefect/MLflow/FastAPI/Evidently stack, or flag if anyone has a strong reason to deviate
- [ ] Finalize the shared data contract (joined dataset schema + API request/response schema) before workstreams diverge
- [ ] Confirm the DST-realignment approach (fixed-offset conversion) holds up under the hour-of-day sanity check once implemented.

## 9. Task Managmenent

### Week 1: 
* Perform data cleaning on COMED_hourly.csv and open-meteo data and join 
* Exploratory Data Analysis
* Feature Engineering - lag features, time series related features etc. 

Deliverable: 
1. Data Cleaning Plan Final **Miguel**
2. Feature Engineering Final **Sachin**

### Week 2: 
* Set up Airflow pipeline - will perform each of these steps (Databricks) **Miguel**
* Set up MLFlow for experimentation (Databricks) **Collin**
* Set up AutoML for model building (Databricks) --> Model Build **Sachin**

Deliverable:
1. Data Preprocessing Pipeline Implemented in AirFlow
2. Final Champ Model Implemented 
3. Model Registered to MLFlow

### Week 3:
* Docker Containerization & FastAPI Build **Sachin**
* Model Monitoring - EvidentlyAI **Minhae**

Deliverable: 
1. Docker Container & FastAPI deployed 
2. EvidentlyAI built and deployed

### Week 4: 
* Final Presentation Deck **Team**





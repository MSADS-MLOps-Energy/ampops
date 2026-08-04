# Finding: COMED timestamps may be Eastern, not Chicago local

**Status:** **RESOLVED — correction adopted.** `config.TIMEZONE_STRATEGY` now
defaults to `"eastern"`, so summer rows shift −1h and winter rows are untouched.
**Raised by:** Miguel, while porting the notebook join into the Airflow pipeline.
**Affects:** the joined dataset, and therefore every workstream downstream of it —
anyone holding a pre-correction `data/processed/*.parquet` must regenerate.

---

## Summary

`notebooks/01_join_and_eda.ipynb` assumes `COMED_hourly.csv` timestamps are
**Chicago wall-clock time** and realigns them onto the weather file's fixed
UTC-5 grid by shifting winter (CST) rows forward one hour.

Three independent checks suggest the stamps are actually **PJM "EPT" (Eastern
Prevailing Time)** — one hour ahead of Chicago. If so, the correction is
backwards: it should shift *summer* rows back one hour and leave winter alone.

This does not affect row counts, null counts, or duplicate counts. The current
join looks perfectly clean either way. That is exactly what makes it worth
checking carefully.

## Evidence

**1. Summer load-temperature coupling.** In summer, ComEd load is driven by air
conditioning and tracks temperature closely, so the correct alignment is the one
that maximizes that correlation. Correlating intraday anomalies (daily mean
removed, so seasonal trend cannot confound it) across candidate shifts:

| COMED shift vs weather grid | −3h | −2h | **−1h** | 0h | +1h | +2h |
|---|---|---|---|---|---|---|
| Summer (JJA) correlation | .7275 | .7843 | **.7925** | .7497 | .6591 | .5275 |

The peak sits at −1h, not 0h. Under the Chicago-local reading, summer stamps
(CDT = UTC-5) are already on the weather grid and should need no shift.

**2. Head-to-head on the same metric.**

| Interpretation | Summer coupling |
|---|---|
| A — Chicago local wall clock (current notebook) | +0.7497 |
| **B — Eastern, hour-beginning** | **+0.7925** |
| C — Eastern, hour-ending | +0.7843 |

**3. The file's own DST structure.** On spring-forward days the source contains
labels `00, 01, 02, 04` — it *keeps* `02:00` and omits `03:00`. Under
hour-*beginning* wall-clock labeling the nonexistent hour is `02:00`, so `02:00`
should be the one missing. A missing `03:00` is an **hour-ending** signature.

Note what this does and does not show. It is evidence against the current
implementation's assumption (plain wall-clock, hour-beginning). It does *not*
identify the zone — hour-ending Central produces the same pattern as hour-ending
Eastern. Treat it as corroboration, not as proof of "Eastern".

## The two explanations are the same correction

There are two plausible stories for why the stamps run an hour ahead in summer:

- **PJM Eastern (EPT).** PJM publishes every zone on Eastern time, including
  ComEd, which physically sits in Central.
- **Chicago local, hour-ending.** The label marks the *end* of the interval, so
  `08:00` means `[07:00, 08:00)`.

Verified empirically: these apply **identical shifts** — summer −1h, winter 0h —
diverging only on a handful of DST-transition rows out of 66,493:

| Interpretation | Summer shift | Winter shift |
|---|---|---|
| A — current implementation | 0h | +1h |
| B — Eastern, hour-beginning | −1h | 0h |
| D — Chicago local, hour-ending | −1h | 0h |

So the team does **not** need to settle which story is true. Both predict the
same correction, and they disagree with the current implementation in the same
direction. The decision is binary: apply the summer −1h correction, or don't.

**Caveat:** the winter equivalent of test 1 is inconclusive in both directions
(correlation ≈ 0.15–0.20 regardless of shift). Winter load is driven by lighting
and occupancy schedules rather than temperature, so there is no sharp physical
anchor in that season. The case rests on the summer signal, which is strong and
clean, plus the structural evidence in (3).

## Why the notebook's sanity check did not catch this

The notebook's DST check compares the realigned join against an uncorrected
join and confirms a +1h winter difference. That verifies the transformation did
what it was written to do — it does not test the assumption behind it against
anything external. Both interpretations pass it. `check_dst_alignment` in
`src/ampops/data/validate.py` inherits that limitation; the tests above are the
external anchor it lacks.

## Confirmation: every model improved

The decisive corroboration. After adopting the correction and regenerating the
datasets, all three bake-off candidates improved on the same validation split —
which is what you expect when features stop being misaligned with the target,
and is not what you would see if the change were arbitrary:

| Model | Before (chicago_local) | After (eastern) |
|---|---|---|
| Linear | 5.45% MAPE | **5.23%** |
| RandomForest | 4.87% MAPE | **4.74%** |
| XGBoost | 4.45% MAPE | **4.18%** |

Summer intraday load-vs-temperature coupling moved 0.7497 → 0.7925 as predicted.
Winter coupling moved 0.1797 → 0.1564, within the noise band that made the
winter test inconclusive in the first place.

## What changed in code

- `config.TIMEZONE_STRATEGY` defaults to `"eastern"`. Override with
  `AMPOPS_TZ_STRATEGY=chicago_local` to reproduce the original notebook.
- `ampops.data.clean.realign_to_fixed_utc5` dispatches to
  `realign_eastern` or `realign_chicago_local`. Both are tested.
- `ampops.data.validate.check_dst_alignment` is strategy-aware: it now asserts
  the *summer* −1h correction and that winter is untouched. Under the old
  strategy it asserted the mirror image.

Both branches keep the same output contract: 66,493 rows × 32 columns, no
nulls, no duplicate timestamps.

## Caveat worth keeping in mind

`check_dst_alignment` proves the transformation ran as intended. It cannot prove
the intent was right — both strategies pass their own version of it. The
external evidence is the summer coupling test and the model improvements above,
not the gate.

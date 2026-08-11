# DAG-trigger bug: `OSError: [Errno 35] Resource deadlock avoided`

**Status: fixed** (2026-08-10) by applying option 2 below — `data/interim`, `data/processed` and `logs/` now live in named Docker volumes instead of host bind mounts, so the DAG's hot rewrite paths never cross VirtioFS. See "Applied fix" near the bottom for exactly what changed and what deliberately did not. One bind mount that has failed before (`data/raw`) is still bind-mounted by design; see "Residual risk".

The rest of this document is the original investigation, kept as-is because it's the reasoning the fix rests on.

## Symptom

A task fails with a traceback like:

```
File "/opt/airflow/src/ampops/utils/io.py", line 38, in write_parquet
    df.to_parquet(path, index=False)
...
OSError: [Errno 35] Resource deadlock avoided: '/opt/airflow/data/processed/joined_hourly.parquet'
```

Observed on two different tasks in two different sessions:

- `ingest_raw`, reading `data/raw/open-meteo-41.86N87.65W179m.csv` (`pd.read_csv`)
- `clean_and_join`, writing `data/processed/joined_hourly.parquet` (`df.to_parquet`), reproduced **twice in a row**, including once against a completely fresh Docker Desktop VM

It has also been reproduced with a plain `cp` and `wc -c`/`wc -l` — this is **not** specific to pandas/pyarrow or to any particular library.

## Environment

- Docker Desktop 4.63.0, arm64 (Apple Silicon)
- macOS 26.5.1 (BuildVersion 25F80)
- Hypervisor: Apple `Virtualization.framework` (the VM shows up on the host as `com.apple.Virtualization.VirtualMachine.xpc`)
- File sharing backend: whatever Docker Desktop 4.63.0 defaults to (not explicitly changed) — almost certainly VirtioFS, not gRPC-FUSE or legacy osxfs
- Affected mounts: the `dags/`, `src/`, `data/`, `logs/`, `plugins/` host bind mounts declared under `x-airflow-common.volumes` in `docker-compose.yml`

## Investigation (what was ruled out, and what the evidence actually shows)

1. **Not Spotlight.** Initial theory (from an earlier session) was a macOS Spotlight indexer (`mdworker`) holding the CSV open concurrently. Re-investigated this session: `lsof` on the host at the moment of failure showed `mdworker`/`mds_stores` holding **zero** handles on any file under `data/`. Ruled out.

2. **Not iCloud Desktop & Documents sync**, despite the repo living under `~/Desktop` (a classic iCloud sync root). `brctl status` does show iCloud actively syncing *some* files under this repo (notably Airflow log files), but `lsof` confirms iCloud's `bird`/cloud-docs daemon holds no handles on the specific files that failed.

3. **The actual (sole) holder is Docker Desktop's own VM process.** `lsof` on the host showed `com.apple.Virtualization.VirtualMachine.xpc` — Docker Desktop's hypervisor process — holding read-mode file descriptors on the exact files that then failed to open from inside the container. It was also holding **multiple stale generations** of frequently-edited source files simultaneously (e.g. 5 different inode numbers for `dags/ampops_training_pipeline.py`, one per edit made during the session), which looked at first like a plain file-descriptor leak.

4. **Not a session-length FD leak, though.** To test that theory, Docker Desktop was fully restarted (`quit` + relaunch) specifically to get a brand-new VM process with zero accumulated state. The very next DAG trigger against that fresh VM (only ~3 minutes old) failed **identically**, on the same file. A pure "leak that builds up over a long session" theory cannot explain an immediate failure against a fresh VM. Ruled out as the sole/primary cause.

5. **Root cause: a per-inode VirtioFS lock-state bug**, confirmed by direct experiment:
   - The *host* could always read the affected file fine (`wc -c` from the host terminal never failed).
   - The *container* could read **other, untouched files in the same directory** fine (e.g. `test.parquet`, `features.parquet` copied without error) at the exact same time it failed on `train.parquet`.
   - So the failure is scoped to one specific inode, not the whole mount and not the whole session. Something about a prior `open()` on that exact inode (very plausibly the interrupted/failed `pd.read_parquet` call from an earlier failed attempt) leaves that inode's lock/mmap state on the VirtioFS host-guest boundary in a bad state, and every subsequent container-side `open()` of that same inode — from any process, including a plain `cp` — gets rejected with `EDEADLK`, seemingly indefinitely (it did not self-resolve after ~10+ minutes or a full VM restart while the same inode number persisted, since restarting the VM doesn't change the file's inode on the host filesystem).

6. **Not the code that generates `joined_hourly.parquet`, either.** Given the bug first surfaced writing this specific file, it was worth asking directly: is `clean_and_join`'s own code (`join.py`, `clean.py`, `utils/io.py::write_parquet`) doing something unusual that provokes this? Investigated and ruled out:
   - `write_parquet()` is a plain `df.to_parquet(path, index=False)` — no manual file-handle management, no memory-mapping, no unusual open modes. `join_hourly()`/`naive_join()` are plain `pd.merge` calls. Nothing here differs from any other write in the pipeline.
   - Decisive counter-evidence: the *same* `EDEADLK` error class hit a completely different file, in a completely different task, with completely unrelated generating code, in an earlier session — a plain `pd.read_csv` on the raw weather CSV inside `ingest_raw`. If the bug tracked with `join.py`'s code, that failure wouldn't have happened at all. It clearly doesn't track with *what code touches the file* — it tracks with the *access pattern* described in point 5 above.
   - What does correlate: `joined_hourly.parquet` — like every other intermediate this DAG produces (`comed_hourly.parquet`, `weather_hourly.parquet`, `features.parquet`, `train.parquet`, `test.parquet`) — gets rewritten to a **fixed, well-known path on every single DAG run and every retry**, because Airflow runs `clean_and_join` → `validate_joined` → `build_features` as three separate subprocesses and this pipeline's own design deliberately hands data between them as file paths, never in-memory DataFrames (see the DAG module's own docstring). That's confirmed by contrast with `scripts/run_pipeline_local.py`, which runs the identical clean → join → build_features logic in a single process and only writes `joined_hourly.parquet` as an optional on-disk deliverable — it never reads it back, because in a single process there's no need to. So the repeated-rewrite exposure is a direct, structural consequence of splitting the pipeline into this many separate Airflow tasks that each persist their output to a fixed path — not a property of any one file's generating code. `joined_hourly.parquet` isn't special; it's just whichever fixed path happened to have the worst prior access history when the bug struck this time. This matters for the fix below: it means the *known-required-to-exist* files worth protecting are specifically the ones on this repeated-rewrite treadmill, not the ones with unusual code behind them.

## Verified workaround (not a fix)

Copying the affected file to a **new path from the host side** (`cp data/processed/train.parquet data/processed/train_run.parquet`, run in the host terminal, not inside the container) produces a fresh inode the container has never touched. The container can then read that fresh copy immediately without error. This was used to unblock a real, live AutoML run (see `docs/automl_implementation.md`, "Validation performed") by copying `train.parquet`/`test.parquet` to fresh host-side copies, then into the container's own `/tmp` (a path that never crosses VirtioFS again).

This is a manual, one-off workaround — it is **not** something the DAG can do for itself today, since the poisoned inode is only discovered by hitting the error in the first place, and the DAG's own tasks (`clean_and_join`, etc.) write to fixed, well-known paths (`config.JOINED_PARQUET`, etc.) by design, not fresh paths per run.

## Options considered

Two structural options, either of which should make the DAG reliably operational without manual intervention. **Option 2 is the one that was applied** — see "Applied fix" below.

1. **Switch Docker Desktop's file-sharing backend from VirtioFS to gRPC-FUSE.** Docker Desktop → Settings → Resources → File sharing. This is the standard community workaround for this exact bug class (VirtioFS mmap/lock-negotiation issues on macOS are a known, still-open category of Docker Desktop bugs). Pros: no code or `docker-compose.yml` change needed. Cons: slower I/O than VirtioFS; it's a per-developer-machine setting, not something enforceable or documented in the repo itself — every teammate would need to change it locally, and there's no way to verify a teammate has it set correctly from CI or from the repo.

2. **Move the hot read/write paths off the host bind mount onto named Docker volumes.** In `docker-compose.yml`, change `data/` (or at least `data/interim/`, `data/processed/`) and `logs/` from host bind mounts (`./data:/opt/airflow/data`) to named volumes (`data-volume:/opt/airflow/data`). Named volumes live entirely inside the Linux VM's own filesystem and never cross the VirtioFS host-guest bridge, so this bug becomes structurally impossible for those paths. Pros: robust, works for every developer regardless of local Docker Desktop settings, enforceable via the committed `docker-compose.yml`. Cons: `data/processed/*.parquet` and `logs/` would no longer be directly browsable from Finder/the host terminal — inspecting them would require `docker compose cp` or `docker compose exec`.

   Per the investigation above (point 6), the risk isn't tied to *which* file — it's tied to *fixed paths the DAG rewrites on every run/retry*. That's every intermediate this pipeline produces: `data/interim/comed_hourly.parquet`, `data/interim/weather_hourly.parquet`, `data/processed/joined_hourly.parquet`, `data/processed/features.parquet`, `data/processed/train.parquet`, `data/processed/test.parquet`. A reasonable split, if full host visibility into `data/` matters (e.g. `joined_hourly.parquet` is also a project deliverable referenced by `notebooks/01_join_and_eda.ipynb` and validated against `config.EXPECTED_JOINED_ROWS`/`EXPECTED_JOINED_COLS`, so there's a real reason to want it browsable): move only the purely-internal intermediates that nothing outside the DAG reads (`data/interim/*`, `features.parquet`, `train.parquet`, `test.parquet`) to a named volume, and leave `data/raw/` and `data/processed/joined_hourly.parquet` bind-mounted. That still doesn't make the bind-mounted files bulletproof — the very first incident was a plain read of the raw weather CSV, so read-heavy files aren't automatically exempt — but it removes the highest-frequency rewrite targets from VirtioFS entirely. So far the 3 recorded incidents were 1 read of a `data/raw` file and 2 writes of the same `data/processed/joined_hourly.parquet` — both categories this split would still leave bind-mounted, so treat "leave raw and the joined-hourly deliverable bind-mounted" as a risk-reduction move for the deepest, most-rewritten intermediates, not a guarantee that the remaining bind-mounted files are safe.

## Applied fix

Option 2, with a deliberately partial scope. Three files changed; **nothing under `src/ampops/` was touched** — the container-side paths are identical, so `ampops.config` resolves exactly as before.

- **`docker-compose.yml`** — in `x-airflow-common.volumes`, `./data:/opt/airflow/data` and `./logs:/opt/airflow/logs` are replaced by:
  - `./data/raw:/opt/airflow/data/raw:ro` — still a host bind mount, now read-only. Keeps the two raw CSVs drop-in from the host per README "Getting the data", with no seeding step. Nothing in the pipeline writes here.
  - `ampops-data-interim:/opt/airflow/data/interim`, `ampops-data-processed:/opt/airflow/data/processed`, `ampops-logs:/opt/airflow/logs` — named volumes, declared in the top-level `volumes:` block. These are the fixed paths the DAG rewrites on every run and every retry (point 6 above), plus the highest-write-frequency mount in the stack.
  - `./dags` and `./src` stay bind-mounted so code edits are still live without a rebuild. They're small, read-mostly, and no incident has ever been recorded on them.

- **`docker/airflow/Dockerfile`** — a `mkdir -p` + `chown -R 50000:0` + `chmod -R 775` of `/opt/airflow/data` in the `USER root` block. **This is required, not cosmetic.** Docker initializes an empty named volume from whatever exists at its mount path in the image, *including ownership*; `apache/airflow:2.9.3-python3.11` has no `/opt/airflow/data` at all, so without this the two volumes come up `root:root` while the containers run as `user: "50000:0"`, and every `write_parquet()` fails with `EACCES` instead. (`/opt/airflow/logs` already ships as `50000:0` `drwxrwxr-x`, so the logs volume needs nothing.)

- **`Makefile`** — `make data-export` (volume → `./data/processed`) and `make data-import` (the reverse), both via `docker compose cp`, which streams over the Docker API and never touches VirtioFS. There is deliberately **no** auto-export task in the DAG: writing to a bind-mounted fixed path at the end of every run would re-introduce exactly the access pattern that causes this bug. `make airflow-reset` (`docker compose down -v`) now also wipes generated pipeline data and task logs — its comment was updated to say so.

`.env.example` and `README.md` were updated to match (the `AIRFLOW_UID=$(id -u)` advice no longer applies to `data`/`logs`, and generated parquets no longer appear on the host automatically).

### Validation

`make airflow-down && make airflow-up` (rebuilds the image), then verified the volumes come up `50000:0` `775` and writable as uid 50000, and that the raw CSVs are still visible through the read-only bind mount.

Then triggered `ampops_training_pipeline` **three times consecutively** — the bar set above, given the bug had reproduced 100% of the time (3/3) before this. All three runs reached `success` on **all 9 tasks**, including `ingest_raw`, `clean_and_join` and `split_train_test`, the three tasks that had never previously survived. A `grep` for `Errno 35` / `EDEADLK` / `Resource deadlock` across every task log in the `ampops-logs` volume after all three runs returned nothing.

Each run registered a new `ampops-demand-forecaster` version end-to-end with full tags (`semantic_version`, `algorithm=drf`, validation `mape`/`rmse`, and `test_mape`/`test_rmse`/`test_mae`), with `@champion` resolving to the newest — i.e. the AutoML → register → test-evaluate path now completes through the scheduler with no manual intervention. Representative run: validation MAPE 0.0423 / RMSE 741.9, test MAPE 0.0301 / RMSE 518.9.

Also confirmed after the fix:

- `make data-export` pulls the generated parquets to the host, and the exported `joined_hourly.parquet` is 66,493 rows × 32 cols — exactly `config.EXPECTED_JOINED_ROWS`/`EXPECTED_JOINED_COLS`.
- The full test suite still passes in the rebuilt container: **72 passed** (`python -m pytest -q` in `airflow-scheduler`; `pytest` is not in the image, so `python -m pip install pytest` first).
- Host-side flows are untouched — `make pipeline-local` still runs against `./data` and produces identical shapes, because `AMPOPS_DATA_DIR` is only set inside the containers.

## Residual risk

`data/raw` is still a host bind mount, and **incident #1 was a plain `pd.read_csv` on the raw weather CSV** — so that read path is not structurally immune. It is much lower-exposure than the paths that moved (read-only, read once per run, never rewritten), but it is not zero.

If `ingest_raw` ever fails with `EDEADLK` again, escalate in this order:

1. **Mint a fresh inode from the host** — `cp data/raw/<file> data/raw/<file>.new && mv data/raw/<file>.new data/raw/<file>`, run in the host terminal. This leaves the poisoned inode behind and unblocks the container immediately. Same mechanism as the "Verified workaround" above, applied to raw.
2. **Move `data/raw` into a volume too** — swap the bind mount for `ampops-data-raw:/opt/airflow/data/raw` and add a `make data-seed` target (`docker compose cp data/raw/. airflow-scheduler:/opt/airflow/data/raw/`) to load it. This takes `data/` off VirtioFS entirely; the cost is that dropping a new CSV on the host no longer suffices.
3. **Option 1, gRPC-FUSE** — the only remaining move, since `dags/` and `src/` must stay bind-mounted for live editing. Per-machine setting, not enforceable from the repo.

## Related

- `docs/automl_implementation.md` — "Validation performed" section documents where this was hit during AutoML validation, and the live run that used the workaround above.
- `CLAUDE.md`'s "Build decision log" (gitignored, local-only) has a terser version of this same investigation.

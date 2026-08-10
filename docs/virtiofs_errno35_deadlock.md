# DAG-trigger bug: `OSError: [Errno 35] Resource deadlock avoided`

**Status: open, unresolved. Currently worked around, not fixed.** `airflow dags trigger ampops_training_pipeline` cannot be relied on to complete end-to-end on this stack yet — it fails non-deterministically on ordinary file reads/writes inside tasks that touch `data/`. This document is a handoff for whoever picks up fixing it for real; see "Recommended fix" at the bottom.

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

## Verified workaround (not a fix)

Copying the affected file to a **new path from the host side** (`cp data/processed/train.parquet data/processed/train_run.parquet`, run in the host terminal, not inside the container) produces a fresh inode the container has never touched. The container can then read that fresh copy immediately without error. This was used to unblock a real, live AutoML run (see `docs/automl_implementation.md`, "Validation performed") by copying `train.parquet`/`test.parquet` to fresh host-side copies, then into the container's own `/tmp` (a path that never crosses VirtioFS again).

This is a manual, one-off workaround — it is **not** something the DAG can do for itself today, since the poisoned inode is only discovered by hitting the error in the first place, and the DAG's own tasks (`clean_and_join`, etc.) write to fixed, well-known paths (`config.JOINED_PARQUET`, etc.) by design, not fresh paths per run.

## Recommended fix (for whoever picks this up)

Two structural options, either of which should make the DAG reliably operational without manual intervention. Neither has been applied yet — this needs a decision, then implementation:

1. **Switch Docker Desktop's file-sharing backend from VirtioFS to gRPC-FUSE.** Docker Desktop → Settings → Resources → File sharing. This is the standard community workaround for this exact bug class (VirtioFS mmap/lock-negotiation issues on macOS are a known, still-open category of Docker Desktop bugs). Pros: no code or `docker-compose.yml` change needed. Cons: slower I/O than VirtioFS; it's a per-developer-machine setting, not something enforceable or documented in the repo itself — every teammate would need to change it locally, and there's no way to verify a teammate has it set correctly from CI or from the repo.

2. **Move the hot read/write paths off the host bind mount onto named Docker volumes.** In `docker-compose.yml`, change `data/` (or at least `data/interim/`, `data/processed/`) and `logs/` from host bind mounts (`./data:/opt/airflow/data`) to named volumes (`data-volume:/opt/airflow/data`). Named volumes live entirely inside the Linux VM's own filesystem and never cross the VirtioFS host-guest bridge, so this bug becomes structurally impossible for those paths. Pros: robust, works for every developer regardless of local Docker Desktop settings, enforceable via the committed `docker-compose.yml`. Cons: `data/processed/*.parquet` and `logs/` would no longer be directly browsable from Finder/the host terminal — inspecting them would require `docker compose cp` or `docker compose exec`. `data/raw/` (read-only, rarely rewritten) could likely stay a bind mount since the bug seems tied to files that get repeatedly opened/rewritten, not to read-only static files in general — though the very first incident *was* a plain read of the raw weather CSV, so this isn't guaranteed safe either; worth validating empirically before assuming raw data is exempt.

Either fix should be validated by triggering `ampops_training_pipeline` several times in a row (the bug has reproduced 100% of the time so far, 3/3 attempts across two sessions, so a handful of clean runs would be reasonably strong evidence of a fix) before considering this resolved.

## Related

- `docs/automl_implementation.md` — "Validation performed" section documents where this was hit during AutoML validation, and the live run that used the workaround above.
- `CLAUDE.md`'s "Build decision log" (gitignored, local-only) has a terser version of this same investigation.

# Download datasets from S3, upload/download models to/from S3

## What changed

- Extracted the S3 plumbing that was inline in `upload_datasets_to_s3.py`
  into [`scripts/_s3_common.py`](../../scripts/_s3_common.py) — client
  construction, local file discovery (glob + exclude-dir), remote object
  discovery (paginated `list_objects_v2` + fnmatch + exclude-dir), the
  upload/download-one-file functions, and a shared threaded runner with a
  `tqdm` progress bar. All four scripts below are thin argparse wrappers
  around it now; [`upload_datasets_to_s3.py`](../../scripts/upload_datasets_to_s3.py)
  was refactored to match (same CLI, same defaults, no behavior change).
- Added [`scripts/download_datasets_from_s3.py`](../../scripts/download_datasets_from_s3.py) —
  pulls `s3://nyx/bronze/*.csv` down into `data/`.
- Added [`scripts/upload_models_to_s3.py`](../../scripts/upload_models_to_s3.py) —
  pushes `models/**/*` to `s3://nyx/models/`.
- Added [`scripts/download_models_from_s3.py`](../../scripts/download_models_from_s3.py) —
  pulls `s3://nyx/models/*` down into `models/`.

## Why

Same rationale as the original dataset-upload script: local-only 17GB of
model checkpoints and 1.2GB of datasets aren't shareable or recoverable if
this machine is lost. Round-tripping (upload + download) is needed for
onboarding a second machine or restoring after a wipe, not just backup.

## Design notes

- **`models/` prefix, not `bronze/`**: bronze/silver/gold is a data-lake
  layering convention for datasets, not applicable to model artifacts, so
  models get their own top-level prefix in the same `nyx` bucket. Override
  with `--prefix` / `--bucket` if that convention doesn't fit — I picked it
  as a reasonable default rather than asking, since it's a one-flag fix if
  wrong.
- **`models/` scope excludes `.git`**: `models/Nyx/` is a full nested git
  checkout (its own `.git/` directory, ~a few hundred MB of pack files).
  The default include pattern is `**/*` (unlike the CSV-only dataset
  default) because model artifacts span `.pth`, `.h5`, `.safetensors`,
  `.dict`, tokenizer `.json`/`.txt` files, but `.git` and `__pycache__` are
  always excluded by directory name — same mechanism as `_dl_smoke`/`archive`
  for datasets, just a different default set. `--exclude-dir` still works
  to add more.
- **Bigger transfer chunks for models**: added a shared `TransferConfig`
  (64MB multipart threshold/chunksize, 8 concurrent parts) used by both
  upload and download — boto3's default 8MB parts and default concurrency
  are tuned for many-small-files, not multi-GB `.safetensors`/`.pth`
  checkpoints.
- **Download-side matching uses `fnmatch`, not `pathlib.glob`**: remote
  keys are flat strings, not a walkable filesystem, so `discover_remote_objects`
  lists everything under the prefix via a paginator and filters with
  `fnmatch.fnmatch(rel, pattern)` — `*` in fnmatch matches across `/`
  (regex `.*`), so `"*.csv"` still matches `nyx-2-instruct/finetuning_llm.csv`-style
  nested keys without needing `**` glob semantics.
- **Idempotency is still size-only** (see prior changelog) — same
  known limitation on both the upload and now the download side: a
  same-size local file is assumed to match and is skipped unless `--force`.

## Verified

Ran all four scripts with `--dry-run` against the live endpoint (`.env`
already had working `S3_EREUNA_*` credentials pointing at the `nyx`
bucket, confirming the endpoint is reachable and the earlier env var
naming happened to line up with what was already provisioned):

- `upload_datasets_to_s3.py --dry-run`: 17 local CSVs found, 16 already
  present remotely at matching size, 1 would-upload.
- `download_datasets_from_s3.py --dry-run`: 18 remote objects under
  `bronze/` (one more than the 17 found locally — there's an extra CSV in
  the bucket not present in this checkout; not resolved here, just noting
  it for awareness), 16 skipped, 2 would-download.
- `upload_models_to_s3.py --dry-run`: 20 local files, 13.1 GB, correctly
  excludes `models/Nyx/.git/`; nothing exists remotely yet so all 20
  would-upload.
- `download_models_from_s3.py --dry-run`: correctly reports nothing under
  `s3://nyx/models/` yet (matches the upload script's finding).

No actual transfer was performed for the model scripts in this session —
that's a 13GB+ push, worth kicking off deliberately rather than as a
byproduct of writing the script.

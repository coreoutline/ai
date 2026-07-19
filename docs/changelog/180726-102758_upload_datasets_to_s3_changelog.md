# Upload datasets to s3.ereuna.org

## What changed

- Added [`scripts/upload_datasets_to_s3.py`](../../scripts/upload_datasets_to_s3.py) —
  a CLI that uploads local dataset files to the `nyx` bucket on the
  s3.ereuna.org object store, under the `bronze/` prefix.
- Added `S3_EREUNA_ENDPOINT_URL`, `S3_EREUNA_ACCESS_KEY`, `S3_EREUNA_SECRET_KEY`
  to [`.env.example`](../../.env.example) — kept separate from the existing
  generic `AWS_ACCESS_KEY`/`AWS_SECRET_ACCESS_KEY` pair so switching between
  a real AWS account and this self-hosted endpoint doesn't require
  overwriting credentials.
- Added `boto3` and `python-dotenv` to [`requirements.txt`](../../requirements.txt).
- **Fixed an unresolved git merge conflict in `requirements.txt`** (lines 5-14,
  `<<<<<<< HEAD` / `=======` / `>>>>>>>` markers were committed as-is). Merged
  both sides, deduped `psutil`, and dropped `tqdmpandas` — there is no such
  PyPI package; `tqdm` already ships the `tqdm.pandas()` integration used
  elsewhere in the repo. Flagging this since it means `pip install -r
  requirements.txt` was broken on `main` before this change.

## Why

Datasets under `data/` (1.2 GB) live only on this machine. Pushing the raw,
unprocessed CSVs to a `bronze/` prefix on object storage gives a durable,
shareable source-of-truth for the medallion-style pipeline (bronze = raw,
presumably silver/gold = cleaned/aggregated downstream) without bloating the
git repo.

## Scope decision

`data/` mixes real datasets with scratch/debug output (`output.txt`,
`verdict.txt`, `content.txt`, `code_contents.txt`, `archive.zip`, `archive/`,
`_dl_smoke/`). The script defaults to uploading `**/*.csv` only, and
explicitly skips `_dl_smoke/`, `archive/`, `wandb/`, and `ad-hoc-notebooks/`
directories even if they contained CSVs — `_dl_smoke/` in particular holds
stale duplicates of top-level `combined_*.csv` files. Both the include glob
(`--include`) and exclude list (`--exclude-dir`) are CLI flags, and non-CSV
formats (parquet, jsonl, etc.) can be added later without touching the
upload/skip logic.

## Design notes

- **boto3 over `aws s3 sync` / `rclone`**: the repo is already a Python
  project with no existing CLI tooling installed for S3 sync, and boto3
  talks to any S3-compatible endpoint (MinIO, Ceph RGW, etc.) via
  `endpoint_url` — no assumption is made about what s3.ereuna.org runs.
  Trade-off: `aws s3 sync` or `rclone` would need a separate binary
  installed and a `~/.aws/config` profile, which is more setup for a
  single-purpose script but would give resumable multipart sync for free
  on very large files. Revisit if datasets grow well past what a 4-worker
  `ThreadPoolExecutor` handles comfortably.
- **Idempotent by size check**: before uploading, the script does a
  `head_object` and skips the file if a remote object of the same byte size
  already exists (`--force` to override). This is not a checksum/ETag
  comparison — good enough for "did this file already land," not for
  detecting silent corruption. A future hardening pass could compare
  `ETag` (MD5, for non-multipart uploads) if that guarantee matters.
- **`--dry-run`**: prints what would be uploaded/skipped without
  transferring, useful for verifying the include/exclude globs before
  moving 1.2 GB.

## Usage

```bash
# fill in .env from .env.example first:
#   S3_EREUNA_ENDPOINT_URL=https://s3.ereuna.org
#   S3_EREUNA_ACCESS_KEY=...
#   S3_EREUNA_SECRET_KEY=...

python3.11 scripts/upload_datasets_to_s3.py --dry-run   # verify file list first
python3.11 scripts/upload_datasets_to_s3.py              # real upload
```

Verified locally: `discover_files()` against the real `data/` directory
matches exactly the 17 CSVs the user confirmed should be in scope (excludes
`_dl_smoke/` and `archive/` correctly); running the CLI with no credentials
set fails fast with a clear error and exit code 1 rather than raising a raw
boto3 traceback.

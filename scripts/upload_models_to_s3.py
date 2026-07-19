#!/usr/bin/env python3.11
"""Upload local model checkpoints to the s3.ereuna.org object store.

Usage:
    python3.11 scripts/upload_models_to_s3.py
    python3.11 scripts/upload_models_to_s3.py --dry-run
    python3.11 scripts/upload_models_to_s3.py --source models --bucket nyx --prefix models/
    python3.11 scripts/upload_models_to_s3.py --include "**/*.pth" --include "**/*.safetensors"

Credentials are read from the environment (or a local .env file):
    S3_EREUNA_ENDPOINT_URL, S3_EREUNA_ACCESS_KEY, S3_EREUNA_SECRET_KEY
"""

import argparse
import sys
from pathlib import Path

import _s3_common as s3

DEFAULT_INCLUDE_GLOBS = ["**/*"]
DEFAULT_EXCLUDE_DIRS = {".git", "__pycache__"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="models", help="Local directory to upload from (default: models)")
    parser.add_argument("--bucket", default="nyx", help="Target bucket (default: nyx)")
    parser.add_argument("--prefix", default="models/", help="Key prefix inside the bucket (default: models/)")
    parser.add_argument(
        "--include", action="append", dest="include_globs",
        help="Glob pattern (relative to --source) to include; repeatable. Default: **/* (everything)",
    )
    parser.add_argument(
        "--exclude-dir", action="append", dest="exclude_dirs", default=[],
        help="Directory name to skip; repeatable. Defaults: .git, __pycache__",
    )
    parser.add_argument("--force", action="store_true", help="Re-upload even if a same-size object already exists")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded without transferring")
    parser.add_argument("--workers", type=int, default=4, help="Parallel upload workers (default: 4)")
    args = parser.parse_args()

    s3.configure_logging()
    client = s3.build_client()

    source = Path(args.source).resolve()
    if not source.is_dir():
        s3.logger.error("Source directory does not exist: %s", source)
        return 1

    include_globs = args.include_globs or DEFAULT_INCLUDE_GLOBS
    exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(args.exclude_dirs)

    files = s3.discover_local_files(source, include_globs, exclude_dirs)
    if not files:
        s3.logger.warning("No files matched %s under %s (excluding %s)", include_globs, source, sorted(exclude_dirs))
        return 0

    total_gb = sum(f.stat().st_size for f in files) / (1024 ** 3)
    s3.logger.info(
        "Found %d file(s) (%.1f GB) under %s matching %s (excluding dirs: %s)",
        len(files), total_gb, source, include_globs, sorted(exclude_dirs),
    )

    jobs = [
        (client, args.bucket, path, s3.key_for_local(path, source, args.prefix), args.force, args.dry_run)
        for path in files
    ]
    counts, _ = s3.run_parallel(jobs, s3.upload_one, args.workers, desc="Uploading")
    return s3.log_summary(counts)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3.11
"""Download datasets from the s3.ereuna.org object store (bronze layer) to data/.

Usage:
    python3.11 scripts/download_datasets_from_s3.py
    python3.11 scripts/download_datasets_from_s3.py --dry-run
    python3.11 scripts/download_datasets_from_s3.py --dest data --bucket nyx --prefix bronze/
    python3.11 scripts/download_datasets_from_s3.py --include "*.csv" --include "*.parquet"

Credentials are read from the environment (or a local .env file):
    S3_EREUNA_ENDPOINT_URL, S3_EREUNA_ACCESS_KEY, S3_EREUNA_SECRET_KEY
"""

import argparse
import sys
from pathlib import Path

import _s3_common as s3

DEFAULT_INCLUDE_GLOBS = ["*.csv"]
DEFAULT_EXCLUDE_DIRS = {"_dl_smoke", "archive", "wandb", "ad-hoc-notebooks"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default="data", help="Local directory to download into (default: data)")
    parser.add_argument("--bucket", default="nyx", help="Source bucket (default: nyx)")
    parser.add_argument("--prefix", default="bronze/", help="Key prefix inside the bucket (default: bronze/)")
    parser.add_argument(
        "--include", action="append", dest="include_globs",
        help="fnmatch pattern (relative to --prefix) to include; repeatable. Default: *.csv",
    )
    parser.add_argument(
        "--exclude-dir", action="append", dest="exclude_dirs", default=[],
        help="Directory name to skip; repeatable. Defaults: _dl_smoke, archive, wandb, ad-hoc-notebooks",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if a same-size local file already exists")
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded without transferring")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers (default: 4)")
    args = parser.parse_args()

    s3.configure_logging()
    client = s3.build_client()

    dest = Path(args.dest).resolve()
    include_globs = args.include_globs or DEFAULT_INCLUDE_GLOBS
    exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(args.exclude_dirs)

    objects = s3.discover_remote_objects(client, args.bucket, args.prefix, include_globs, exclude_dirs)
    if not objects:
        s3.logger.warning(
            "No objects matched %s under s3://%s/%s (excluding %s)",
            include_globs, args.bucket, args.prefix, sorted(exclude_dirs),
        )
        return 0

    s3.logger.info(
        "Found %d object(s) under s3://%s/%s matching %s (excluding dirs: %s)",
        len(objects), args.bucket, args.prefix, include_globs, sorted(exclude_dirs),
    )

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    jobs = [
        (client, args.bucket, obj["key"], obj["size"], s3.local_path_for_key(obj["rel"], dest), args.force, args.dry_run)
        for obj in objects
    ]
    counts, _ = s3.run_parallel(jobs, s3.download_one, args.workers, desc="Downloading")
    return s3.log_summary(counts)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3.11
"""Upload local datasets to the s3.ereuna.org object store (bronze layer).

Usage:
    python3.11 scripts/upload_datasets_to_s3.py
    python3.11 scripts/upload_datasets_to_s3.py --dry-run
    python3.11 scripts/upload_datasets_to_s3.py --source data --bucket nyx --prefix bronze/
    python3.11 scripts/upload_datasets_to_s3.py --include "**/*.csv" --include "**/*.parquet"

Credentials are read from the environment (or a local .env file):
    S3_EREUNA_ENDPOINT_URL, S3_EREUNA_ACCESS_KEY, S3_EREUNA_SECRET_KEY
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

DEFAULT_INCLUDE_GLOBS = ["**/*.csv"]
DEFAULT_EXCLUDE_DIRS = {"_dl_smoke", "archive", "wandb", "ad-hoc-notebooks"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("upload_datasets_to_s3")


def discover_files(source: Path, include_globs: list[str], exclude_dirs: set[str]) -> list[Path]:
    matched: dict[Path, None] = {}
    for pattern in include_globs:
        for path in source.glob(pattern):
            if not path.is_file():
                continue
            if exclude_dirs & set(path.relative_to(source).parts[:-1]):
                continue
            matched[path] = None
    return sorted(matched)


def build_client(endpoint_url: str, access_key: str, secret_key: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def s3_key_for(path: Path, source: Path, prefix: str) -> str:
    rel = PurePosixPath(*path.relative_to(source).parts)
    return f"{prefix.rstrip('/')}/{rel}"


def remote_size(client, bucket: str, key: str) -> int | None:
    try:
        return client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def upload_one(client, bucket: str, path: Path, key: str, force: bool, dry_run: bool) -> str:
    if not force:
        existing = remote_size(client, bucket, key)
        if existing == path.stat().st_size:
            return "skipped"
    if dry_run:
        return "would-upload"
    client.upload_file(str(path), bucket, key)
    return "uploaded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data", help="Local directory to upload from (default: data)")
    parser.add_argument("--bucket", default="nyx", help="Target bucket (default: nyx)")
    parser.add_argument("--prefix", default="bronze/", help="Key prefix inside the bucket (default: bronze/)")
    parser.add_argument(
        "--include", action="append", dest="include_globs",
        help="Glob pattern (relative to --source) to include; repeatable. Default: **/*.csv",
    )
    parser.add_argument(
        "--exclude-dir", action="append", dest="exclude_dirs", default=[],
        help="Directory name to skip; repeatable. Defaults: _dl_smoke, archive, wandb, ad-hoc-notebooks",
    )
    parser.add_argument("--force", action="store_true", help="Re-upload even if a same-size object already exists")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded without transferring")
    parser.add_argument("--workers", type=int, default=4, help="Parallel upload workers (default: 4)")
    args = parser.parse_args()

    load_dotenv()
    endpoint_url = os.environ.get("S3_EREUNA_ENDPOINT_URL")
    access_key = os.environ.get("S3_EREUNA_ACCESS_KEY")
    secret_key = os.environ.get("S3_EREUNA_SECRET_KEY")
    if not all([endpoint_url, access_key, secret_key]):
        logger.error(
            "Missing S3_EREUNA_ENDPOINT_URL / S3_EREUNA_ACCESS_KEY / S3_EREUNA_SECRET_KEY. "
            "Set them in .env (see .env.example) or the environment."
        )
        return 1

    source = Path(args.source).resolve()
    if not source.is_dir():
        logger.error("Source directory does not exist: %s", source)
        return 1

    include_globs = args.include_globs or DEFAULT_INCLUDE_GLOBS
    exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(args.exclude_dirs)

    files = discover_files(source, include_globs, exclude_dirs)
    if not files:
        logger.warning("No files matched %s under %s (excluding %s)", include_globs, source, sorted(exclude_dirs))
        return 0

    logger.info(
        "Found %d file(s) under %s matching %s (excluding dirs: %s)",
        len(files), source, include_globs, sorted(exclude_dirs),
    )

    client = build_client(endpoint_url, access_key, secret_key)

    counts = {"uploaded": 0, "skipped": 0, "would-upload": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for path in files:
            key = s3_key_for(path, source, args.prefix)
            futures[pool.submit(upload_one, client, args.bucket, path, key, args.force, args.dry_run)] = (path, key)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
            path, key = futures[future]
            try:
                result = future.result()
                counts[result] += 1
                logger.debug("%s -> s3://%s/%s [%s]", path, args.bucket, key, result)
            except Exception:
                counts["failed"] += 1
                logger.exception("Failed to upload %s -> s3://%s/%s", path, args.bucket, key)

    logger.info(
        "Done. uploaded=%d skipped=%d would-upload=%d failed=%d",
        counts["uploaded"], counts["skipped"], counts["would-upload"], counts["failed"],
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

"""Shared helpers for the s3.ereuna.org sync scripts (upload/download datasets/models).

Not a standalone CLI. Credentials are read from the environment (or a local
.env file): S3_EREUNA_ENDPOINT_URL, S3_EREUNA_ACCESS_KEY, S3_EREUNA_SECRET_KEY.
"""

import fnmatch
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

logger = logging.getLogger("s3_sync")

# Large model checkpoints (.pth, .safetensors) benefit from bigger parts and
# more concurrent parts than boto3's 8MB/10-thread defaults.
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=8,
)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def build_client():
    load_dotenv()
    endpoint_url = os.environ.get("S3_EREUNA_ENDPOINT_URL")
    access_key = os.environ.get("S3_EREUNA_ACCESS_KEY")
    secret_key = os.environ.get("S3_EREUNA_SECRET_KEY")
    if not all([endpoint_url, access_key, secret_key]):
        raise SystemExit(
            "Missing S3_EREUNA_ENDPOINT_URL / S3_EREUNA_ACCESS_KEY / S3_EREUNA_SECRET_KEY. "
            "Set them in .env (see .env.example) or the environment."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def _is_excluded(rel_dir_parts, exclude_dirs: set) -> bool:
    return bool(exclude_dirs & set(rel_dir_parts))


def discover_local_files(source: Path, include_globs: list, exclude_dirs: set) -> list:
    """Files under `source` matching any of `include_globs`, skipping any path
    with a component in `exclude_dirs`. Returns sorted, deduped Paths."""
    matched = {}
    for pattern in include_globs:
        for path in source.glob(pattern):
            if not path.is_file():
                continue
            if _is_excluded(path.relative_to(source).parts[:-1], exclude_dirs):
                continue
            matched[path] = None
    return sorted(matched)


def key_for_local(path: Path, source: Path, prefix: str) -> str:
    rel = PurePosixPath(*path.relative_to(source).parts)
    return f"{prefix.rstrip('/')}/{rel}"


def discover_remote_objects(client, bucket: str, prefix: str, include_globs: list, exclude_dirs: set) -> list:
    """Objects under `bucket`/`prefix` matching any of `include_globs` (fnmatch
    against the key relative to prefix), skipping any path with a component in
    `exclude_dirs`. Returns list of {key, rel, size} sorted by key."""
    norm_prefix = prefix.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=norm_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(norm_prefix):]
            rel_parts = PurePosixPath(rel).parts
            if _is_excluded(rel_parts[:-1], exclude_dirs):
                continue
            if include_globs and not any(fnmatch.fnmatch(rel, pat) for pat in include_globs):
                continue
            objects.append({"key": key, "rel": rel, "size": obj["Size"]})
    return sorted(objects, key=lambda o: o["key"])


def local_path_for_key(rel: str, dest: Path) -> Path:
    return dest.joinpath(*PurePosixPath(rel).parts)


def remote_size(client, bucket: str, key: str):
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
    client.upload_file(str(path), bucket, key, Config=TRANSFER_CONFIG)
    return "uploaded"


def download_one(client, bucket: str, key: str, size: int, dest_path: Path, force: bool, dry_run: bool) -> str:
    if not force and dest_path.exists() and dest_path.stat().st_size == size:
        return "skipped"
    if dry_run:
        return "would-download"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest_path), Config=TRANSFER_CONFIG)
    return "downloaded"


def run_parallel(jobs: list, worker_fn, workers: int, desc: str):
    """Runs worker_fn(*job) for each job in `jobs` across a thread pool.
    Returns (counts: dict[result_label, int], failures: list[job])."""
    counts = {}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker_fn, *job): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            job = futures[future]
            try:
                result = future.result()
                counts[result] = counts.get(result, 0) + 1
            except Exception:
                counts["failed"] = counts.get("failed", 0) + 1
                failures.append(job)
                logger.exception("Failed job: %s", job)
    return counts, failures


def log_summary(counts: dict) -> int:
    parts = " ".join(f"{label}={count}" for label, count in sorted(counts.items()))
    logger.info("Done. %s", parts or "nothing to do")
    return 1 if counts.get("failed") else 0

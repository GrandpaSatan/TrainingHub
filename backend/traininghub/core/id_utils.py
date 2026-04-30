from __future__ import annotations

import re
from datetime import datetime, timezone


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str, fallback: str = "item") -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    return slug or fallback


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_dataset_id(slug: str) -> str:
    return f"ds_{timestamp()}_{slugify(slug, 'dataset')}"


def make_job_id(prefix: str, slug: str) -> str:
    return f"{prefix}_{timestamp()}_{slugify(slug, 'job')}"


def make_artifact_id(prefix: str, slug: str) -> str:
    return f"{prefix}_{timestamp()}_{slugify(slug, 'artifact')}"

"""Source-manifest schema and license approval gate.

Collection is deliberately list-driven: this module validates rows supplied by
the user and never discovers sources or infers permission from a URL.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


SOURCE_COLUMNS = (
    "source_id",
    "kind",
    "url",
    "local_path",
    "license_status",
    "license_name",
    "license_url",
    "farm_id",
    "cow_id",
    "video_id",
    "passage_id",
    "camera_id",
    "notes",
)
VALID_LICENSE_STATUSES = frozenset({"approved", "restricted", "unresolved"})
LEGACY_UNAPPROVED_LICENSES = frozenset(
    {"", "check-before-use", "unknown", "unresolved"}
)


class LicenseApprovalError(ValueError):
    """Raised before collection when any source lacks explicit approval."""


def _joined(values: Iterable[str]) -> str:
    return ", ".join(sorted({str(value) for value in values}))


def normalize_source_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize new and legacy source manifests without approving their use."""
    required = {"source_id", "kind", "url", "local_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"sources CSV is missing columns: {missing}")
    if frame.empty:
        raise ValueError("sources CSV contains no active rows")

    result = frame.copy()
    original_columns = set(result.columns)
    for column in SOURCE_COLUMNS:
        if column not in result:
            result[column] = ""

    has_legacy = "license" in original_columns
    statuses: list[str] = []
    names: list[str] = []
    invalid_statuses: list[str] = []

    for _, row in result.iterrows():
        source_id = str(row.get("source_id", "")).strip() or "<blank source_id>"
        raw_status = str(row.get("license_status", "")).strip().lower()
        legacy = str(row.get("license", "")).strip() if has_legacy else ""
        legacy_key = legacy.lower()

        if raw_status:
            status = raw_status
            if status not in VALID_LICENSE_STATUSES:
                invalid_statuses.append(f"{source_id}={status!r}")
        else:
            # Compatibility is intentionally narrow: placeholder values never
            # become approval just because the legacy field is non-null.
            status = "unresolved" if legacy_key in LEGACY_UNAPPROVED_LICENSES else "approved"

        name = str(row.get("license_name", "")).strip() or legacy
        statuses.append(status)
        names.append(name)
    if invalid_statuses:
        allowed = _joined(VALID_LICENSE_STATUSES)
        raise ValueError(
            f"invalid license_status ({_joined(invalid_statuses)}); expected one of: {allowed}"
        )
    result["license_status"] = statuses
    result["license_name"] = names
    ordered = list(SOURCE_COLUMNS) + [
        column for column in result.columns if column not in SOURCE_COLUMNS
    ]
    return result.loc[:, ordered]


def normalize_and_require_approved_sources(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized source table after enforcing the license gate.

    New manifests must use ``license_status``. For backward compatibility, an
    older non-empty ``license`` value is treated as approved unless it is one of
    the explicit unresolved placeholders. Restricted or unresolved rows always
    fail; callers should validate the full table before touching the network or
    creating collection outputs.
    """
    result = normalize_source_schema(frame)
    blocked = result.loc[
        result["license_status"].ne("approved"), ["source_id", "license_status"]
    ]
    if len(blocked):
        details = ", ".join(
            f"{row.source_id} ({row.license_status})" for row in blocked.itertuples()
        )
        raise LicenseApprovalError(
            "Only sources with license_status='approved' may be resolved or downloaded; "
            f"blocked: {details}"
        )
    return result

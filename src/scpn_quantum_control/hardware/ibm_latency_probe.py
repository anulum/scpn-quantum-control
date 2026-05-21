# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IBM latency probe helpers
"""Helpers for IBM Runtime latency telemetry extraction and normalisation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast


def iso_utc_now() -> str:
    """Return current UTC timestamp in RFC3339-like format."""
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: Any) -> datetime | None:
    """Parse provider timestamp values into timezone-aware UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_call(job: Any, name: str) -> Any:
    target = getattr(job, name, None)
    if target is None:
        return None
    if callable(target):
        try:
            return target()
        except Exception as exc:  # defensive metadata surface probe
            return f"<error:{type(exc).__name__}>"
    return target


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        parsed = parse_timestamp(value)
        return parsed.isoformat() if parsed is not None else value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def extract_job_telemetry(job: Any) -> dict[str, Any]:
    """Extract provider-side telemetry fields from a runtime job."""
    raw: dict[str, Any] = {}
    for name in (
        "job_id",
        "session_id",
        "program_id",
        "primitive_id",
        "creation_date",
        "running",
        "result",
        "metrics",
        "usage",
        "properties",
        "status",
    ):
        if name == "result":
            continue
        raw[name] = _safe_call(job, name)
    metrics = raw.get("metrics")
    if isinstance(metrics, dict):
        for key in (
            "created",
            "queued",
            "running",
            "finished",
            "completed",
            "execution",
            "usage",
        ):
            if key in metrics:
                raw[f"metrics_{key}"] = metrics.get(key)
    return cast(dict[str, Any], _json_safe(raw))


def _duration_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def derive_timing_windows(telemetry: dict[str, Any]) -> dict[str, float | None]:
    """Derive queue/run windows from provider telemetry where available."""
    created = parse_timestamp(telemetry.get("creation_date"))
    queued = parse_timestamp(telemetry.get("queued"))
    running = parse_timestamp(telemetry.get("running"))
    finished = parse_timestamp(telemetry.get("finished"))
    completed = parse_timestamp(telemetry.get("completed"))
    metric_queued = parse_timestamp(telemetry.get("metrics_queued"))
    metric_running = parse_timestamp(telemetry.get("metrics_running"))
    metric_finished = parse_timestamp(telemetry.get("metrics_finished"))
    queue_start = queued or metric_queued or created
    run_start = running or metric_running
    run_end = finished or completed or metric_finished
    return {
        "provider_queue_seconds": _duration_seconds(queue_start, run_start),
        "provider_run_seconds": _duration_seconds(run_start, run_end),
        "provider_total_seconds": _duration_seconds(created or queue_start, run_end),
    }

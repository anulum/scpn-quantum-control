# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IBM latency probe helper tests

from __future__ import annotations

from datetime import datetime, timezone

from scpn_quantum_control.hardware.ibm_latency_probe import (
    derive_timing_windows,
    parse_timestamp,
)


def test_parse_timestamp_supports_z_suffix() -> None:
    ts = parse_timestamp("2026-05-22T00:00:01Z")
    assert ts is not None
    assert ts.tzinfo is not None
    assert ts.isoformat().startswith("2026-05-22T00:00:01")


def test_parse_timestamp_normalises_naive_datetime_to_utc() -> None:
    raw = datetime(2026, 5, 22, 0, 0, 1)
    parsed = parse_timestamp(raw)
    assert parsed is not None
    assert parsed.tzinfo == timezone.utc


def test_derive_timing_windows_from_provider_fields() -> None:
    telemetry = {
        "creation_date": "2026-05-22T00:00:00Z",
        "queued": "2026-05-22T00:00:10Z",
        "running": "2026-05-22T00:00:20Z",
        "finished": "2026-05-22T00:00:50Z",
    }
    windows = derive_timing_windows(telemetry)
    assert windows["provider_queue_seconds"] == 10.0
    assert windows["provider_run_seconds"] == 30.0
    assert windows["provider_total_seconds"] == 50.0


def test_derive_timing_windows_accepts_metrics_fallback_keys() -> None:
    telemetry = {
        "creation_date": "2026-05-22T00:00:00Z",
        "metrics_queued": "2026-05-22T00:00:05Z",
        "metrics_running": "2026-05-22T00:00:15Z",
        "metrics_finished": "2026-05-22T00:00:35Z",
    }
    windows = derive_timing_windows(telemetry)
    assert windows["provider_queue_seconds"] == 10.0
    assert windows["provider_run_seconds"] == 20.0
    assert windows["provider_total_seconds"] == 35.0

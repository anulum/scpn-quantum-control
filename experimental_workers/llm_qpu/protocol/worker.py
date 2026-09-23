# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — isolated experimental worker boundary
"""Report the worker boundary and refuse all unimplemented operations."""

from __future__ import annotations

import json
import sys

_MAX_REQUEST_BYTES = 65_536
_SCHEMA = "scpn.experimental.llm_qpu.worker_boundary.v1"


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    """Read one bounded request and refuse any operation except discovery.

    Returns
    -------
    int
        Zero for a boundary description or two for a refused request.

    """
    raw = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    if len(raw) > _MAX_REQUEST_BYTES:
        _emit({"schema": _SCHEMA, "status": "refused", "reason": "request too large"})
        return 2
    try:
        request = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _emit({"schema": _SCHEMA, "status": "refused", "reason": "invalid JSON"})
        return 2
    if type(request) is not dict or set(request) != {"op"} or request["op"] != "describe":
        _emit({"schema": _SCHEMA, "status": "refused", "reason": "unsupported operation"})
        return 2
    _emit(
        {
            "schema": _SCHEMA,
            "status": "experimental_no_compute",
            "supported_operations": ["describe"],
            "hardware_submission_enabled": False,
            "provider_credentials_required": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

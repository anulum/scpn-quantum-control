# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — shared codec and HAL dependency contracts
"""Exercise typed canonical bytes and the real HAL export without Studio."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_quantum_control import canonical_encoding
from scpn_quantum_control.studio_workspace import canonical


@pytest.mark.parametrize(
    ("value", "tagged"),
    [
        (None, "null"),
        (True, "true"),
        (2**64 - 1, '["integer","18446744073709551615"]'),
        (-0.0, '["float64","8000000000000000"]'),
        (["integer", "1"], '["array",["integer","1"]]'),
        ({"β": 1, "a": 2}, '["object",[["a",["integer","2"]],["β",["integer","1"]]]]'),
    ],
)
def test_shared_bytes_and_workspace_api(value: object, tagged: str) -> None:
    """Retain independently declared typed bytes through both public entry points.

    Parameters
    ----------
    value
        Native scalar or structured value admitted by the public codec.
    tagged
        Independent canonical JSON byte oracle.

    """
    expected = ("shared.v1\n" + tagged).encode("utf-8")
    assert canonical_encoding.canonical_bytes("shared.v1", value) == expected
    assert canonical.canonical_bytes("shared.v1", value) == expected
    assert (
        canonical_encoding.canonical_digest("shared.v1", value)
        == hashlib.sha256(expected).hexdigest()
    )
    assert canonical.canonical_digest("shared.v1", value) == hashlib.sha256(expected).hexdigest()
    assert canonical.MAX_DEPTH == canonical_encoding.MAX_DEPTH
    assert canonical.MAX_INTEGER_DIGITS == canonical_encoding.MAX_INTEGER_DIGITS


@pytest.mark.parametrize("value", [float("nan"), b"bytes", {1: "invalid"}, "\udfff"])
def test_shared_refusal_and_workspace_api(value: object) -> None:
    """Keep fail-closed scalar admission identical for HAL and workspace callers.

    Parameters
    ----------
    value
        Actual unsupported canonical value.

    """
    for encode in (canonical_encoding.canonical_bytes, canonical.canonical_bytes):
        with pytest.raises(ValueError):
            encode("shared.v1", value)


def test_hal_profile_projection_keeps_studio_unloaded() -> None:
    """Build the complete native public catalogue in a fresh process without Studio."""
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ, PYTHONPATH=str(root / "src") + os.pathsep + str(root / "oscillatools/src")
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; "
            "from scpn_quantum_control.hardware.provider_capability_discovery "
            "import build_backend_profiles; "
            "profiles=build_backend_profiles(observed_at='2026-10-03'); "
            "assert not any(name.startswith('scpn_quantum_control.studio') "
            "for name in sys.modules); "
            "print(json.dumps(profiles,ensure_ascii=False))",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    actual = json.loads(process.stdout)
    expected = json.loads((root / "data/studio/backend_profiles.json").read_text())
    assert actual == expected
    assert actual["body"]["no_submit"] is True
    assert len(actual["body"]["profiles"]) == 45

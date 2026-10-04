# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — actual standalone operator review verification
"""Run generated native verification scripts through their physical public CLI."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")

from test_operator_review_dossier import review_inputs  # noqa: E402

from scpn_quantum_control.canonical_encoding import canonical_digest  # noqa: E402
from scpn_quantum_control.studio.executive_execute import ExecuteActionHandler  # noqa: E402
from scpn_quantum_control.studio.operator_review_script import build_review_script  # noqa: E402
from scpn_quantum_control.studio.workspace import write_json  # noqa: E402


@pytest.mark.parametrize(
    "case", ["show", "match", "mismatch", "length", "missing", "empty", "oversized", "confirm"]
)
def test_physical_review_script_never_submits(case: str, tmp_path: Path) -> None:
    """The actual standalone CLI verifies bytes or refuses without a provider path.

    Parameters
    ----------
    case
        Exact-byte success, real file refusal or unsupported submission argument.
    tmp_path
        Owned script/payload files inside the declared task allocation.

    """
    request, inputs = review_inputs()
    dossier = ExecuteActionHandler().prepare_review(request, **inputs)
    script = dossier.script
    path = tmp_path / script.filename
    path.write_text(script.source, encoding="utf-8")
    argv = [sys.executable, str(path)]
    if case == "confirm":
        argv.append("--confirm")
    elif case != "show":
        payload = tmp_path / "compiled.bin"
        if case != "missing":
            data = inputs["compiled_payload"]
            if case == "mismatch":
                data = b"x" * len(data)
            elif case == "length":
                data += b"changed-length"
            elif case == "empty":
                data = b""
            elif case == "oversized":
                data = b"x" * (1024 * 1024 + 1)
            payload.write_bytes(data)
        argv.extend(["--payload", str(payload)])
    result = subprocess.run(
        argv, cwd=tmp_path, text=True, capture_output=True, timeout=15, check=False
    )
    assert result.returncode == (0 if case in ("show", "match") else 2 if case == "confirm" else 1)
    assert result.stdout == (dossier.text if case in ("show", "match") else "")
    assert "NotImplementedError" not in script.source
    assert script.source.splitlines()[:7] == [
        # Expected header of the emitted script, not a licence tag of this file.
        # REUSE-IgnoreStart
        "# SPDX-License-Identifier: AGPL-3.0-or-later",
        # REUSE-IgnoreEnd
        "# Commercial license available",
        "# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "# © Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "# ORCID: 0009-0009-3560-0851",
        "# Contact: www.anulum.li | protoscience@anulum.li",
        "# SCPN Quantum Control — standalone operator review verifier",
    ]
    assert "submit_circuit" not in script.source
    tree = ast.parse(script.source)
    modules = {
        name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert modules == {"argparse", "hashlib", "sys", "pathlib"}
    assert not (tmp_path / "counts.json").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "root",
        "keys",
        "schema",
        "hash",
        "body",
        "no_submit",
        "payload",
        "payload_hash",
        "size",
        "oversized_text",
        "body_extra",
        "subhash",
        "execution_hash",
    ],
)
def test_script_builder_refuses_changed_source(fault: str) -> None:
    """Unsupported metadata cannot be exported as a valid native verifier.

    Parameters
    ----------
    fault
        One malformed source shape or identity, with other binding kept exact.

    """
    request, inputs = review_inputs()
    dossier = ExecuteActionHandler().prepare_review(request, **inputs)
    wire = dossier.to_dict()
    if fault == "root":
        raw = "[]"
    elif fault == "oversized_text":
        raw = "α" * (1024 * 1024)
    else:
        if fault == "keys":
            wire["extra"] = True
        elif fault == "schema":
            wire["schema"] = "studio.operator-review-dossier.v2"
        elif fault == "hash":
            wire["sha256"] = "0" * 64
        elif fault == "body":
            wire["body"] = []
        elif fault == "no_submit":
            wire["body"]["no_submit"] = False
        elif fault == "payload":
            wire["body"]["payload"] = []
        elif fault == "payload_hash":
            wire["body"]["payload"]["sha256"] = "bad"
        elif fault == "body_extra":
            wire["body"]["credential"] = "unsupported"
        elif fault == "subhash":
            wire["body"]["plan_sha256"] = "a" * 64
        elif fault == "execution_hash":
            wire["body"]["execution_sha256"] = "a" * 64
        else:
            wire["body"]["payload"]["size_bytes"] = True
        if fault not in ("keys", "schema", "hash"):
            wire["sha256"] = canonical_digest(
                wire["schema"], {k: v for k, v in wire.items() if k != "sha256"}
            )
        raw = write_json(wire)
    with pytest.raises(ValueError):
        build_review_script(raw)


def test_native_export_bundle_retains_original_dossier_and_script() -> None:
    """Browser consumers receive native source bytes instead of generating a script."""
    request, inputs = review_inputs()
    dossier = ExecuteActionHandler().prepare_review(request, **inputs)
    bundle = dossier.export_bundle()
    assert bundle["schema"] == "studio.operator-review-export.v1"
    body = bundle["body"]
    assert isinstance(body, dict)
    assert body["dossier_text"] == dossier.text
    assert body["dossier_sha256"] == dossier.sha256
    assert body["script"] == dossier.script.to_dict()
    assert body["no_submit"] is True
    assert bundle["sha256"] == canonical_digest(
        "studio.operator-review-export.v1", {k: v for k, v in bundle.items() if k != "sha256"}
    )

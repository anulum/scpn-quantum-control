# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Serialization surface audit tests
"""Tests for unsafe deserialisation surface auditing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TOOL = ROOT / "tools" / "audit_serialization_surface.py"
_SPEC = importlib.util.spec_from_file_location("audit_serialization_surface", _TOOL)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

candidate_files = _MODULE.candidate_files
findings_to_json = _MODULE.findings_to_json
main = _MODULE.main
scan_text = _MODULE.scan_text


def test_serialization_audit_flags_qpy_load() -> None:
    findings = scan_text(
        Path("scripts/load_qpy.py"),
        "from qiskit import qpy\nqpy.load(handle)\n",
    )

    assert findings[0].symbol == "qiskit.qpy.load"


def test_serialization_audit_allows_reviewed_qpy_wrapper_only() -> None:
    reviewed_findings = scan_text(
        Path("src/scpn_quantum_control/hardware/hal_qiskit.py"),
        "from qiskit import qpy\n"
        "def _reviewed_qpy_load_circuits(data):\n"
        "    return qpy.load(data)\n",
    )
    unreviewed_findings = scan_text(
        Path("src/scpn_quantum_control/hardware/hal_qiskit.py"),
        "from qiskit import qpy\ndef _other_loader(data):\n    return qpy.load(data)\n",
    )

    assert reviewed_findings == ()
    assert unreviewed_findings[0].symbol == "qiskit.qpy.load"


def test_serialization_audit_flags_pickle_loads() -> None:
    findings = scan_text(
        Path("src/example.py"),
        "import pickle as p\np.loads(payload)\n",
    )

    assert findings[0].symbol == "pickle.loads"


def test_serialization_audit_flags_np_load_allow_pickle_true() -> None:
    findings = scan_text(
        Path("scripts/example.py"),
        "import numpy as np\nnp.load(path, allow_pickle=True)\n",
    )

    assert findings[0].symbol == "numpy.load"
    assert "allow_pickle=True" in findings[0].reason


def test_serialization_audit_allows_default_np_load_and_json() -> None:
    findings = scan_text(
        Path("scripts/example.py"),
        "import json\nimport numpy as np\njson.load(handle)\nnp.load(path)\n",
    )

    assert findings == ()


def test_serialization_audit_candidate_files_scans_default_roots(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "module.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "docs" / "conf.py").write_text("print('not default')\n", encoding="utf-8")

    files = candidate_files(tmp_path)

    assert Path("src/module.py") in files
    assert Path("docs/conf.py") not in files


def test_serialization_audit_json_output_is_deterministic() -> None:
    findings = scan_text(Path("scripts/example.py"), "import joblib\njoblib.load(path)\n")

    decoded = json.loads(findings_to_json(findings))

    assert decoded[0]["symbol"] == "joblib.load"
    assert sorted(decoded[0]) == ["column", "line", "path", "reason", "symbol"]


def test_serialization_audit_cli_returns_nonzero_on_findings(
    tmp_path: Path, capsys: object
) -> None:
    fixture = tmp_path / "unsafe.py"
    fixture.write_text("import marshal\nmarshal.loads(payload)\n", encoding="utf-8")

    assert main(["--project-root", str(tmp_path), "--input", str(fixture)]) == 1
    assert "marshal.loads" in capsys.readouterr().out


def test_serialization_audit_cli_returns_zero_on_safe_file(tmp_path: Path, capsys: object) -> None:
    fixture = tmp_path / "safe.py"
    fixture.write_text("import json\njson.load(handle)\n", encoding="utf-8")

    assert main(["--project-root", str(tmp_path), "--input", str(fixture)]) == 0
    assert "no unsafe" in capsys.readouterr().out


def test_serialization_audit_flags_weights_only_false() -> None:
    findings = scan_text(
        Path("scripts/replay.py"),
        "import torch\ntorch.load(path, weights_only=False)\n",
    )

    assert findings[0].symbol == "load(weights_only=False)"
    assert "digest-gated" in findings[0].reason


def test_serialization_audit_allows_digest_gated_torch_load_wrapper_only() -> None:
    gated_findings = scan_text(
        Path("src/scpn_quantum_control/phase/torch_aot_autograd_export.py"),
        "def _torch_load_graph(torch_module, path, *, expected_sha256):\n"
        "    return torch_module.load(path, weights_only=False)\n",
    )
    ungated_findings = scan_text(
        Path("src/scpn_quantum_control/phase/torch_aot_autograd_export.py"),
        "def _other_loader(torch_module, path):\n"
        "    return torch_module.load(path, weights_only=False)\n",
    )

    assert gated_findings == ()
    assert ungated_findings[0].symbol == "load(weights_only=False)"


def test_serialization_audit_allows_weights_only_true_and_absent() -> None:
    findings = scan_text(
        Path("scripts/replay.py"),
        "import torch\ntorch.load(path, weights_only=True)\njson.load(handle)\n",
    )

    assert findings == ()

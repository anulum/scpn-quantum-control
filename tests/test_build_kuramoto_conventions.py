# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto convention source producer tests
"""Exercise original-source qualification and generated-file CLI transactions."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from scpn_quantum_control import kuramoto_convention_matrix
from scpn_quantum_control.studio_workspace.canonical import canonical_digest
from tools import build_kuramoto_conventions as producer

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def convention_source_projection(tmp_path: Path) -> Path:
    """Copy actual owning source bytes into an independent producer input tree."""
    document = producer.build_kuramoto_conventions(_REPO)
    owners = cast(dict[str, dict[str, object]], document["source_owners"])
    paths = {str(record["path"]) for record in owners.values()} | {
        str(document["definition_path"])
    }
    paths |= {
        str(path)
        for record in owners.values()
        for path in cast(tuple[str, ...], record["direct_test_paths"])
    }
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REPO / relative, target)
    return tmp_path


def test_convention_matrix_qualifies_real_declarations_and_backend_chains() -> None:
    """Actual public owners bind model rows to source bytes and documented contracts."""
    document = producer.build_kuramoto_conventions(_REPO)
    assert document["schema"] == producer.SCHEMA
    unsigned = {key: value for key, value in document.items() if key != "identity"}
    assert document["identity"] == canonical_digest(producer.SCHEMA, unsigned)
    assert len(cast(list[object], document["conventions"])) == len(kuramoto_convention_matrix())
    owners = cast(dict[str, dict[str, object]], document["source_owners"])
    for name, record in owners.items():
        assert record["qualified_name"] == name
        assert (
            record["source_sha256"]
            == hashlib.sha256((_REPO / str(record["path"])).read_bytes()).hexdigest()
        )
        assert record["summary"] and record["signature"]
        assert "not measured" in str(record["dispatch_claim"])
    force = owners["oscillatools.accel.networked_kuramoto.networked_kuramoto_force"]
    chains = cast(dict[str, list[dict[str, str]]], force["module_declared_dispatch_chains"])
    assert {route["tier"] for routes in chains.values() for route in routes} == {
        "rust",
        "julia",
        "python",
    }
    assert "oscillatools/tests/test_networked_kuramoto.py" in cast(
        tuple[str, ...], force["direct_test_paths"]
    )
    method = owners["oscillatools.accel.kuramoto_system.KuramotoSystem.networked"]
    assert "initial_phases" in str(method["signature"]) and "scheme" in str(method["signature"])


def test_projection_is_reproducible_and_source_drift_changes_identity(
    convention_source_projection: Path,
) -> None:
    """An exact actual-source copy agrees, while genuine owner-byte drift is explicit."""
    projection = convention_source_projection
    before = producer.build_kuramoto_conventions(_REPO)
    assert producer.build_kuramoto_conventions(projection) == before
    source = projection / "oscillatools/src/oscillatools/accel/networked_kuramoto.py"
    source.write_bytes(source.read_bytes() + b"\n")
    changed = producer.build_kuramoto_conventions(projection)
    assert changed["identity"] != before["identity"]
    assert changed["conventions"] == before["conventions"]
    assert (
        _REPO / "oscillatools/src/oscillatools/accel/networked_kuramoto.py"
    ).read_bytes() != source.read_bytes()


def test_cli_generates_checks_and_refuses_drift_without_overwriting_saved_files(
    convention_source_projection: Path,
) -> None:
    """Real file generation and check mode retain prior outputs when drift is found."""
    projection = convention_source_projection
    args = ["--repo", str(projection)]
    assert producer.main([*args, "--check"]) == 1
    assert not (projection / producer.MANIFEST).exists()
    assert producer.main(args) == 0
    saved = {
        path: (projection / path).read_bytes() for path in [producer.MANIFEST, producer.GUIDE]
    }
    assert producer.main([*args, "--check"]) == 0
    assert saved == producer.build_convention_artifacts(projection)
    decoded = json.loads(saved[producer.MANIFEST])
    assert decoded["schema"] == producer.SCHEMA
    assert b"Population-mean inputs explicitly become" in saved[producer.GUIDE]
    guide = projection / producer.GUIDE
    guide.write_bytes(saved[producer.GUIDE] + b"altered\n")
    assert producer.main([*args, "--check"]) == 1
    assert guide.read_bytes() == saved[producer.GUIDE] + b"altered\n"
    assert (projection / producer.MANIFEST).read_bytes() == saved[producer.MANIFEST]


@pytest.mark.parametrize(
    "damage", ["missing_owner", "missing_declaration", "invalid_python", "definition_drift"]
)
def test_invalid_source_cannot_replace_previously_generated_artifacts(
    convention_source_projection: Path, damage: str
) -> None:
    """Malformed actual source qualification refuses before either output is written."""
    projection = convention_source_projection
    args = ["--repo", str(projection)]
    assert producer.main(args) == 0
    saved = {
        path: (projection / path).read_bytes() for path in [producer.MANIFEST, producer.GUIDE]
    }
    path = projection / "oscillatools/src/oscillatools/accel/networked_kuramoto.py"
    if damage == "missing_owner":
        path.rename(path.with_suffix(".unavailable"))
    elif damage == "missing_declaration":
        path.write_text(
            path.read_text().replace(
                "def networked_kuramoto_force(", "def unavailable_networked_kuramoto_force("
            )
        )
    elif damage == "invalid_python":
        path.write_text(path.read_text() + "\ndef invalid(:\n")
    else:
        definition = projection / "src/scpn_quantum_control/kuramoto_model_conventions.py"
        definition.write_bytes(definition.read_bytes() + b"\n")
    assert producer.main(args) == 1
    assert producer.main([*args, "--check"]) == 1
    assert all(
        (projection / relative).read_bytes() == payload for relative, payload in saved.items()
    )


@pytest.mark.parametrize(
    "reference",
    [
        "../networked",
        "qiskit.quantum_info.Statevector",
        "oscillatools.accel.absent.absent",
        "oscillatools.accel.kuramoto_system.KuramotoSystem.absent",
    ],
)
def test_public_source_qualifier_refuses_unknown_or_malformed_references(
    reference: str,
) -> None:
    """The source inspection API cannot substitute a declaration from another package."""
    with pytest.raises(ValueError):
        producer.qualify_kuramoto_source_owner(_REPO, reference)


def test_public_source_qualifier_requires_original_native_documentation(
    convention_source_projection: Path,
) -> None:
    """A removed native contract cannot be hidden by a remaining function name."""
    path = (
        convention_source_projection
        / "oscillatools/src/oscillatools/accel/kuramoto_ott_antonsen.py"
    )
    original = path.read_text()
    start = original.index('    r"""Ott–Antonsen vector field')
    stop = original.index('    """', start + 8) + len('    """')
    path.write_text(original[:start] + original[stop:])
    with pytest.raises(ValueError, match="native documentation"):
        producer.qualify_kuramoto_source_owner(
            convention_source_projection,
            "oscillatools.accel.kuramoto_ott_antonsen.ott_antonsen_field",
        )


@pytest.mark.parametrize(
    "replacement",
    [
        "[]",
        "make_dispatch_chain()",
        "['rust']",
        "[('rust',)]",
        "[(7, _rust_networked_kuramoto_force)]",
        "[('rust', 'missing_wrapper')]",
    ],
)
def test_cli_refuses_incomplete_original_dispatch_inventory_without_output_loss(
    convention_source_projection: Path, replacement: str
) -> None:
    """Corrupt an actual force-chain declaration without hiding lost backend rows."""
    projection = convention_source_projection
    args = ["--repo", str(projection)]
    assert producer.main(args) == 0
    saved = {
        path: (projection / path).read_bytes() for path in [producer.MANIFEST, producer.GUIDE]
    }
    path = projection / "oscillatools/src/oscillatools/accel/networked_kuramoto.py"
    source = path.read_text()
    declaration = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_NETWORKED_KURAMOTO_FORCE_CHAIN"
    )
    assert declaration.value is not None and declaration.value.end_lineno is not None
    assert declaration.value.end_col_offset is not None
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[: declaration.value.lineno - 1])) + declaration.value.col_offset
    end = (
        sum(map(len, lines[: declaration.value.end_lineno - 1])) + declaration.value.end_col_offset
    )
    path.write_text(source[:start] + replacement + source[end:])
    assert producer.main(args) == 1
    assert producer.main([*args, "--check"]) == 1
    assert all((projection / relative).read_bytes() == data for relative, data in saved.items())


def test_cli_native_entrypoint_uses_the_same_production_generator(
    convention_source_projection: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Executing the original script as a CLI generates the real qualified files."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_REPO / "tools/build_kuramoto_conventions.py"),
            "--repo",
            str(convention_source_projection),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(_REPO / "tools/build_kuramoto_conventions.py"), run_name="__main__")
    assert exit_info.value.code == 0
    assert producer.main(["--repo", str(convention_source_projection), "--check"]) == 0


def test_source_generation_does_not_import_or_execute_optional_numerical_backends() -> None:
    """A fresh native process inventories source facts without accelerator/provider loads."""
    program = """
import json, sys
from pathlib import Path
from tools.build_kuramoto_conventions import build_kuramoto_conventions
document = build_kuramoto_conventions(Path.cwd())
forbidden = [name for name in sys.modules if name.split('.')[0] in {'jax', 'juliacall', 'qiskit', 'scpn_quantum_engine'}]
print(json.dumps({'rows': len(document['conventions']), 'forbidden': forbidden}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    observed = json.loads(completed.stdout)
    assert observed == {"rows": len(kuramoto_convention_matrix()), "forbidden": []}

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Source-qualified Kuramoto convention producer
"""Generate reproducible convention documentation from original source owners.

AST inspection verifies declarations and inventories declared dispatch chains
without importing a solver, executing an experiment or loading an accelerator.
Source hashes qualify this static inventory; runtime and benchmark evidence
remain separate and cannot be inferred from a row or a backend label.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from scpn_quantum_control import kuramoto_model_conventions as conventions
from scpn_quantum_control.studio_workspace.canonical import canonical_digest

SCHEMA = "kuramoto_conventions.v1"
MANIFEST = Path("docs/_generated/kuramoto_conventions.json")
GUIDE = Path("docs/kuramoto_conventions.md")
_DEFINITION = Path("src/scpn_quantum_control/kuramoto_model_conventions.py")
_OWNER_FIELDS = (
    "evolution_owner",
    "force_owner",
    "observable_owner",
    "state_jacobian_owner",
    "sensitivity_owner",
)


def qualify_kuramoto_source_owner(repo: Path, qualified: str) -> dict[str, object]:
    """Read an original declaration and its source dispatch metadata.

    Parameters
    ----------
    repo
        Scientific repository source root; no source is imported or executed.
    qualified
        Original Python declaration under scpn_quantum_control or oscillatools.

    Returns
    -------
    dict[str, object]
        Exact source/declaration identity, native documentation and declared
        module dispatch metadata, with explicit runtime qualification limits.

    Raises
    ------
    ValueError
        Reference, source declaration, native documentation or a declared
        dispatch chain is unsupported. Incomplete chains are never omitted.
    OSError, SyntaxError
        Original source cannot be read or parsed.

    """
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", qualified):
        raise ValueError("owner reference requires a qualified Python declaration")
    components = qualified.split(".")
    if components[0] == "oscillatools":
        base = repo / "oscillatools/src"
    elif components[0] == "scpn_quantum_control":
        base = repo / "src"
    else:
        raise ValueError("owner reference is outside the original scientific packages")
    for length in range(len(components) - 1, 0, -1):
        path = base.joinpath(*components[:length]).with_suffix(".py")
        if path.is_file():
            break
    else:
        raise ValueError("convention source owner is missing")
    payload = path.read_bytes()
    tree = ast.parse(payload, filename=str(path))
    scope: list[ast.stmt] = tree.body
    declaration: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None = None
    for part in components[length:]:
        declaration = next(
            (
                node
                for node in scope
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == part
            ),
            None,
        )
        if declaration is None:
            raise ValueError("convention source declaration is missing")
        scope = declaration.body
    assert declaration is not None
    source_doc = ast.get_docstring(declaration)
    if not source_doc:
        raise ValueError("convention source declaration requires native documentation")
    relative = path.relative_to(repo)
    direct_tests = tuple(
        candidate.as_posix()
        for candidate in (
            Path("tests") / f"test_{path.stem}.py",
            Path("oscillatools/tests") / f"test_{path.stem}.py",
        )
        if (repo / candidate).is_file()
    )
    chains: dict[str, list[dict[str, str]]] = {}
    value: ast.expr | None
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets, value = [statement.target], statement.value
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or not target.id.endswith("_CHAIN"):
                continue
            if not isinstance(value, (ast.Tuple, ast.List)) or not value.elts:
                raise ValueError("declared dispatch chain requires explicit nonempty routes")
            routes = []
            for element in value.elts:
                if not isinstance(element, (ast.Tuple, ast.List)) or len(element.elts) != 2:
                    raise ValueError("declared dispatch route requires a tier and wrapper")
                tier, wrapper = element.elts
                if (
                    not isinstance(tier, ast.Constant)
                    or not isinstance(tier.value, str)
                    or not isinstance(wrapper, ast.Name)
                ):
                    raise ValueError(
                        "declared dispatch route requires a literal tier and named wrapper"
                    )
                routes.append(
                    {
                        "tier": tier.value,
                        "wrapper": ".".join(components[:length]) + "." + wrapper.id,
                    }
                )
            chains[target.id] = routes
    return {
        "qualified_name": qualified,
        "path": relative.as_posix(),
        "line": declaration.lineno,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "declaration_sha256": hashlib.sha256(
            ast.dump(declaration, include_attributes=False).encode()
        ).hexdigest(),
        "signature": ast.unparse(declaration.args)
        if isinstance(declaration, (ast.FunctionDef, ast.AsyncFunctionDef))
        else "class",
        "summary": source_doc.split("\n\n", 1)[0],
        "direct_test_paths": direct_tests,
        "module_declared_dispatch_chains": chains,
        "dispatch_claim": "source declarations only; availability, selected tier and parity are not measured here",
    }


def build_kuramoto_conventions(repo: Path) -> dict[str, object]:
    """Qualify the complete convention matrix against actual source declarations.

    Parameters
    ----------
    repo
        Canonical repository or a source-qualified projection with the same
        convention definition bytes as this producer's imported owner.

    Returns
    -------
    dict[str, object]
        Versioned rows, exact source/declaration hashes, native documentation,
        direct test owners and source-declared backend chains. Identity binds
        these static facts and does not assert runtime/backend qualification.

    Raises
    ------
    ValueError
        A definition differs from the imported source, or a referenced original
        declaration/documentation is missing. No output is written on refusal.
    OSError, SyntaxError
        A source file cannot be read or parsed.

    """
    definition = (repo / _DEFINITION).read_bytes()
    if definition != Path(conventions.__file__).read_bytes():
        raise ValueError("convention definition differs from the imported source owner")
    rows = conventions.kuramoto_convention_matrix()
    owners = {
        reference
        for row in rows
        for field in _OWNER_FIELDS
        if (reference := getattr(row, field)) is not None
    } | {reference for row in rows for reference in row.backend_owners}
    document: dict[str, object] = {
        "schema": SCHEMA,
        "definition_path": _DEFINITION.as_posix(),
        "definition_sha256": hashlib.sha256(definition).hexdigest(),
        "claim_boundary": "source-qualified implemented conventions; no runtime availability, accuracy or benchmark promotion",
        "conventions": [asdict(row) for row in rows],
        "source_owners": {
            name: qualify_kuramoto_source_owner(repo, name) for name in sorted(owners)
        },
    }
    return {**document, "identity": canonical_digest(SCHEMA, document)}


def build_convention_artifacts(repo: Path) -> dict[Path, bytes]:
    """Build JSON and Markdown together after all source qualification passes.

    Parameters
    ----------
    repo
        Actual source root supplied to the source-owner qualifier.

    Returns
    -------
    dict[pathlib.Path, bytes]
        Reproducible UTF-8 generated files keyed by project-relative output path.

    Raises
    ------
    ValueError, OSError, SyntaxError
        An owning source fails qualification before any output is produced.

    """
    document = build_kuramoto_conventions(repo)
    lines = [
        "# Kuramoto model conventions",
        "",
        "Generated from the original scientific owners by `tools/build_kuramoto_conventions.py`. "
        "The JSON companion binds exact source/declaration hashes, original native documentation and declared dispatch chains. "
        "These are source capabilities; installed availability, selected runtime tiers, convergence, gradients and benchmarks need their own evidence.",
        "",
        "The public `kuramoto_convention_matrix()` and `kuramoto_model_convention(model, solver)` preserve distinct model identities and refuse unsupported pairs. "
        "`build_scientific_phase_system(problem, design, dt=..., model=..., scheme=...)` validates an original `KuramotoProblem` and explicit `ScientificDesign` through the original Euler/RK4 `KuramotoSystem`, without importing Studio. "
        "It admits only instantaneous plain finite networked/mean-field inputs. Other matrix rows refer to their original separate owners; inventory support does not make them valid inputs of this factory.",
        "",
        "```python",
        "import numpy as np",
        "import scpn_quantum_control as qc",
        "",
        "problem = qc.build_kuramoto_problem(np.zeros((2, 2)), np.array([0.2, 0.4]))",
        'design = qc.ScientificDesign(model="phase_kuramoto", normalisation="pairwise_sum",',
        '    coordinate_space="logical", units=qc.ScientificUnits("s", "rad/s", "rad/s", "rad", "1"),',
        '    topology=(), initial_state=np.array([0.0, 0.1]), observable="phase_order_parameter",',
        '    observable_weights=np.ones(2), objective=qc.DesignObjective("simulate", None, "1"))',
        "system = qc.build_scientific_phase_system(problem, design, dt=1/32)",
        "trajectory = system.trajectory(32)  # initial row plus 32 evolved samples",
        "```",
        "",
        "Positive coupling uses `sin(theta[k]-theta[j])`, so it attracts two identical phases. "
        "Networked coefficients are a pairwise sum. Population-mean inputs explicitly become `K_nm/N`; "
        "the finite mean-field factory requires uniform off-diagonal effective coefficients and recovers the original scalar K. "
        "It refuses heterogeneous matrices, scalar overflow, non-finite steps, quantum amplitudes and supplied delay history. "
        "No unit conversion, phase wrapping, continuum approximation or model substitution occurs. "
        "The factory evolves phases only; a declared weighted observable still requires its own matching observable consumer.",
        "",
        "## Model and solver matrix",
        "",
        "| Model | Solver | Interpretation | Evolution owner |",
        "|---|---|---|---|",
    ]
    rows = conventions.kuramoto_convention_matrix()
    for row in rows:
        lines.append(
            f"| `{row.model}` | `{row.solver}` | {row.interpretation} | `{row.evolution_owner or 'none: force operator only'}` |"
        )
    for row in rows:
        lines.extend(
            [
                "",
                f"## {row.model} / {row.solver}",
                "",
                f"State: {row.state}. Equation: `{row.equation}`.",
                "",
                f"Normalisation: {row.normalisation}. Topology: {row.topology}.",
                "",
                f"History: {row.history}. Noise: {row.noise}.",
                "",
                f"Force owner: `{row.force_owner or 'none: reduced coordinates'}`. "
                f"Observable owner: `{row.observable_owner or 'none: use the original trajectory record'}`.",
                "",
                f"State Jacobian: `{row.state_jacobian_owner or 'none declared'}`. "
                f"Sensitivity: `{row.sensitivity_owner or 'none declared'}`; {row.sensitivity_semantics}.",
                "",
                *[f"- {assumption}" for assumption in row.assumptions],
            ]
        )
    lines.extend(
        [
            "",
            "## Source and evidence boundary",
            "",
            f"Static matrix identity: `{document['identity']}`. "
            "Original declarations, source hashes, native summaries, direct test paths and source-declared dispatch chains are in "
            "[`_generated/kuramoto_conventions.json`](_generated/kuramoto_conventions.json). "
            "A registered test path is navigation, not proof the test executed. "
            "Run `PYTHONPATH=src:oscillatools/src python tools/build_kuramoto_conventions.py --check` to reject generated drift. "
            "The original finite analytic regressions exercise actual public numerical owners; they do not qualify every matrix row or optional backend.",
            "",
        ]
    )
    return {
        MANIFEST: (
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
        GUIDE: "\n".join(lines).encode(),
    }


def main(argv: list[str] | None = None) -> int:
    """Generate or check the source-qualified convention files through the CLI.

    Parameters
    ----------
    argv
        CLI arguments; None uses the process argument vector.

    Returns
    -------
    int
        Zero for exact agreement or successful generation; one for generated
        drift or refused source qualification. Check mode never writes files.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        outputs = build_convention_artifacts(args.repo)
    except (ValueError, OSError, SyntaxError):
        print("Kuramoto convention sources are unavailable or inconsistent")
        return 1
    if args.check:
        drift = [
            str(path)
            for path, data in outputs.items()
            if not (args.repo / path).is_file() or (args.repo / path).read_bytes() != data
        ]
        if drift:
            print("Kuramoto convention output drift: " + ", ".join(drift))
            return 1
    else:
        for path, data in outputs.items():
            target = args.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print("Kuramoto convention sources and outputs agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

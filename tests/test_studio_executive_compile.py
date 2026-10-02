# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive compile handler tests
"""Tests for the read-only XY compile ``compile`` handler."""

from __future__ import annotations

import runpy
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from scpn_quantum_control.studio.executive import (
        ActionRegistry,
        ExecutiveRequest,
        preview_action,
        resolve_verb_contract,
        run_action,
    )
    from scpn_quantum_control.studio.executive_compile import (
        COMPILE_VERB,
        CompileActionHandler,
        _as_float,
        _normalise_compile,
        _safe_slug,
    )
else:
    pytest.importorskip("scpn_studio_platform", reason="studio extra not installed")
    from scpn_quantum_control.studio.executive import (
        ActionRegistry,
        ExecutiveRequest,
        preview_action,
        resolve_verb_contract,
        run_action,
    )
    from scpn_quantum_control.studio.executive_compile import (
        COMPILE_VERB,
        CompileActionHandler,
        _as_float,
        _normalise_compile,
        _safe_slug,
    )

_NETWORK: dict[str, Any] = {
    "K_nm": [[0.0, 0.4, 0.1], [0.4, 0.0, 0.3], [0.1, 0.3, 0.0]],
    "omega": [-0.1, 0.05, 0.05],
    "time": 0.1,
    "trotter_steps": 1,
    "trotter_order": 1,
}


def _registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(CompileActionHandler())
    return registry


def _request(*, backend: str | None = None, **overrides: Any) -> ExecutiveRequest:
    parameters = dict(_NETWORK)
    parameters.update(overrides)
    return ExecutiveRequest(
        verb=COMPILE_VERB, action_id="compile-3node", parameters=parameters, backend=backend
    )


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def test_compile_builds_a_verified_bit_exact_unit() -> None:
    """Build and self-verify the public bit-exact compile unit."""
    record = run_action(_request(), registry=_registry())
    assert record.result.status == "succeeded"
    outputs = record.result.outputs
    assert outputs["verified"] is True
    assert outputs["n_nodes"] == 3
    assert outputs["recompute_schema"] == "studio.xy-compile-recompute.v1"
    assert outputs["verifiability_mode"] == "recompute"
    assert outputs["exactness_class"] == "bit-exact"
    assert outputs["input_sha256"].startswith("sha256:")
    assert record.script is not None


def test_compile_plan_defaults_backend_read_only() -> None:
    """Expose the read-only plan with its default Python backend."""
    plan = preview_action(_request(), registry=_registry())
    assert plan.backend == "python"
    assert plan.requires_approval is False
    assert len(plan.steps) == 4


def test_compile_accepts_declared_rust_backend() -> None:
    """Accept the Rust backend declared by the compile verb contract."""
    plan = preview_action(_request(backend="rust"), registry=_registry())
    assert plan.backend == "rust"


def test_compile_rejects_undeclared_backend() -> None:
    """Reject a backend absent from the public compile contract."""
    handler = CompileActionHandler()
    contract = resolve_verb_contract(COMPILE_VERB)
    with pytest.raises(ValueError, match="is not declared for the compile verb"):
        handler.plan(_request(backend="abacus"), contract)


def test_generated_compile_script_embeds_digest_and_compiles() -> None:
    """Generate a syntactically valid script with the sealed input digest."""
    record = run_action(_request(), registry=_registry())
    assert record.script is not None
    source = record.script.source
    compile(source, record.script.filename, "exec")
    assert record.result.outputs["input_sha256"] in source
    assert "build_xy_compile_recompute_unit" in source
    assert "verify_xy_compile_recompute_unit" in source


def test_generated_compile_script_executes_real_entrypoint(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run the generated script and verify its sealed digest through the public API."""
    record = run_action(_request(), registry=_registry())
    assert record.script is not None
    script_path = tmp_path / record.script.filename
    script_path.write_text(record.script.source, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(script_path)])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(script_path), run_name="__main__")

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == (
        f"input_sha256={record.result.outputs['input_sha256']} verified"
    )


def test_compile_trotter_order_two_is_accepted() -> None:
    """Build and verify the supported second-order Trotter route."""
    record = run_action(_request(trotter_order=2, trotter_steps=2), registry=_registry())
    assert record.result.outputs["trotter_order"] == 2
    assert record.result.outputs["verified"] is True


@pytest.mark.parametrize("bad", [1.0, 2.0, 1.5, True, False, "1", None, float("nan")])
def test_compile_preview_rejects_noninteger_order(bad: object) -> None:
    """Reject noninteger Trotter orders before producing an executable plan."""
    with pytest.raises(ValueError, match="trotter_order"):
        preview_action(_request(trotter_order=bad), registry=_registry())


# --------------------------------------------------------------------------- #
# _as_float
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, "1", None, float("inf"), float("nan")])
def test_as_float_rejects_bad(bad: Any) -> None:
    """Reject booleans, non-numbers, and non-finite scalar inputs."""
    with pytest.raises(ValueError):
        _as_float("v", bad)


def test_as_float_accepts_numbers() -> None:
    """Normalise integer scalar input to a finite float."""
    assert _as_float("v", 2) == 2.0


# --------------------------------------------------------------------------- #
# _safe_slug
# --------------------------------------------------------------------------- #
def test_safe_slug_normal_and_empty() -> None:
    """Produce filesystem-safe action slugs and a non-empty fallback."""
    assert _safe_slug("compile-3node.1") == "compile_3node_1"
    assert _safe_slug("!!!") == "action"


# --------------------------------------------------------------------------- #
# _normalise_compile validation branches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides",
    [
        {"K_nm": "matrix"},
        {"K_nm": [1.0, [0.0, 1.0]]},
        {"K_nm": [[0.0]]},
        {"K_nm": [[0.0] * 17 for _ in range(17)], "omega": [0.0] * 17},
        {"K_nm": [[0.0, 1.0], [1.0, 0.0, 0.0]], "omega": [0.0, 0.0]},
        {"K_nm": [[1.0, 0.0], [0.0, 0.0]], "omega": [0.0, 0.0]},
        {"K_nm": [[0.0, 1.0], [2.0, 0.0]], "omega": [0.0, 0.0]},
        {"omega": "not-a-list"},
        {"omega": [0.1, 0.2]},
        {"time": 0.0},
        {"time": -1.0},
        {"trotter_steps": 0},
        {"trotter_steps": 999},
        {"trotter_steps": True},
        {"trotter_steps": "two"},
        {"trotter_order": 3},
        {"trotter_order": True},
    ],
)
def test_normalise_compile_rejects_invalid(overrides: dict[str, Any]) -> None:
    """Reject malformed or unbounded networks through compile normalisation."""
    parameters = dict(_NETWORK)
    parameters.update(overrides)
    with pytest.raises(ValueError):
        _normalise_compile(parameters)


def test_normalise_compile_accepts_bounded_network() -> None:
    """Normalise a bounded symmetric network without changing its order."""
    compile_spec = _normalise_compile(_NETWORK)
    assert len(compile_spec["K_nm"]) == 3
    assert compile_spec["trotter_order"] == 1


def test_supported_program_source_emits_original_ir_without_execution() -> None:
    """Preserve phase, condition and readout through the actual executive action."""
    source = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; rz(-0.7853981633974492) q[0]; if(c==2) x q[1]; measure q[1] -> c[0];'
    request = ExecutiveRequest(
        verb=COMPILE_VERB, action_id="source-emission", parameters={"program_source": source}
    )
    record = run_action(request, registry=_registry())
    assert record.result.status == "succeeded"
    assert record.result.outputs["execution_status"] == "emitted_not_executed"
    program = record.result.outputs["program"]
    assert program["source"] == source
    assert program["measurements"] == [[1, 0]]
    assert program["operations"][0]["parameters"] == ["bfe921fb54442d20"]
    assert program["operations"][1]["condition"] == {"register": "c", "value": "2"}
    assert "verified" not in record.result.outputs


def test_source_reproduction_script_emits_only_original_source(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Execute the generated safe producer through its actual public import."""
    source = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; // inert quote """\nrz(-0.0) q[0];'
    record = run_action(
        ExecutiveRequest(
            verb=COMPILE_VERB, action_id='source """', parameters={"program_source": source}
        ),
        registry=_registry(),
    )
    assert record.script is not None
    script = tmp_path / record.script.filename
    script.write_text(record.script.source, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(script)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    assert (
        capsys.readouterr().out.strip()
        == f"source_sha256={record.result.outputs['source_sha256']} emitted_not_executed"
    )


@pytest.mark.parametrize(
    "parameters,backend,message",
    [
        ({"program_source": "source", "K_nm": []}, None, "cannot be combined"),
        ({"program_source": "source"}, "rust", "requires the Python backend"),
        ({"program_source": "import os"}, None, "grammar"),
    ],
)
def test_source_preview_refuses_unsupported_requests(
    parameters: dict[str, Any], backend: str | None, message: str
) -> None:
    """No source plan substitutes a backend, mixes modes or admits Python."""
    with pytest.raises(ValueError, match=message):
        preview_action(
            ExecutiveRequest(
                verb=COMPILE_VERB,
                action_id="refused-source",
                parameters=parameters,
                backend=backend,
            ),
            registry=_registry(),
        )


def test_source_plan_is_sealed_against_request_mutation() -> None:
    """Keep the actual immutable plan after the caller changes its draft mapping."""
    source = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0];'
    parameters: dict[str, Any] = {"program_source": source}
    plan = preview_action(
        ExecutiveRequest(verb=COMPILE_VERB, action_id="sealed-source", parameters=parameters),
        registry=_registry(),
    )
    parameters["program_source"] = "import os"
    assert plan.parameters["program_source"] == source
    with pytest.raises(TypeError):
        plan.parameters["program_source"] = "changed"  # type: ignore[index]
    assert CompileActionHandler().execute(plan).outputs["program"]["source"] == source


def test_forged_source_plan_digest_is_refused_before_emission() -> None:
    """Refuse an independently replaced plan whose actual source identity differs."""
    source = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0];'
    plan = preview_action(
        ExecutiveRequest(
            verb=COMPILE_VERB, action_id="digest-refusal", parameters={"program_source": source}
        ),
        registry=_registry(),
    )
    altered = replace(plan, parameters={"program_source": source, "source_sha256": "0" * 64})
    with pytest.raises(ValueError, match="differs from its sealed"):
        CompileActionHandler().execute(altered)

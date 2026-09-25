# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental model descriptor contract
"""Exercise the provider-free model wire through its standalone worker."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    ArtifactHeader,
    ModelDescriptor,
    canonical_bytes,
    decode_contract,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _REPO_ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_SCHEMA = "scpn.experimental.llm_qpu.model_descriptor.v1"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"


def _model(*, quantization: str = "gguf_q6_k") -> ModelDescriptor:
    """Build a design-only schema fixture, never a claim about a loaded model."""
    fields = {
        "model_id": "contract-fixture",
        "checkpoint_digest": "a" * 64,
        "tokenizer_digest": "b" * 64,
        "chat_template_digest": "c" * 64,
        "runtime_build_digest": "d" * 64,
        "loader_id": "local-loader-contract",
        "quantization": quantization,
        "tensor_dtype": "float32",
        "block_count": 4,
        "hidden_width": 16,
        "tap_block_index": 2,
        "tap_stream": "residual_hidden_state",
        "tap_boundary": "after_norm",
        "probe_evidence_digest": "e" * 64,
    }
    scientific = {"schema": _SCHEMA, "object_kind": "model_descriptor", **fields}
    header = ArtifactHeader(
        object_kind="model_descriptor",
        content_digest=hashlib.sha256(canonical_bytes(scientific)).hexdigest(),
        parents=tuple(
            sorted(
                fields[name]
                for name in (
                    "checkpoint_digest",
                    "tokenizer_digest",
                    "chat_template_digest",
                    "runtime_build_digest",
                    "probe_evidence_digest",
                )
            )
        ),
        base_repo_commit=_BASE,
        implementation_revision=_REVISION,
        execution_origin="offline_design",
        data_origin="owner_checkpoint",
        claim_scope="design_only",
    )
    return ModelDescriptor(**fields, header=header)


def _worker(request: dict[str, object], tmp_path: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "IQM_TOKEN": "poison", "IBM_QUANTUM_TOKEN": "poison"},
        capture_output=True,
        timeout=15,
        check=False,
    )


def test_model_descriptor_standalone_roundtrip(tmp_path: Path) -> None:
    model = _model()
    request = {"op": "roundtrip_model_descriptor", "model": model.to_wire()}
    response = _worker(request, tmp_path)
    assert response.returncode == 0, response.stdout.decode()
    payload = json.loads(response.stdout)
    assert payload["status"] == "validated_roundtrip_no_compute"
    assert payload["hardware_submission_enabled"] is False
    assert payload["model_sha256"] == hashlib.sha256(canonical_bytes(model.to_wire())).hexdigest()
    assert decode_contract(canonical_bytes(payload["model"])) == model


def test_q8_checkpoint_descriptor_roundtrips_without_inferred_probe(tmp_path: Path) -> None:
    model = _model(quantization="gguf_q8_0")
    response = _worker({"op": "roundtrip_model_descriptor", "model": model.to_wire()}, tmp_path)
    assert response.returncode == 0, response.stdout.decode()
    payload = json.loads(response.stdout)
    assert payload["model"]["quantization"] == "gguf_q8_0"
    assert payload["model"]["header"]["claim_scope"] == "design_only"
    with pytest.raises(ValueError, match="quantization"):
        _model(quantization="gguf_q8_1")


def test_model_descriptor_refuses_inferred_width_and_fallback(tmp_path: Path) -> None:
    model = _model()
    for field, value, reason in (
        ("hidden_width", True, "observed hidden width"),
        ("block_count", 0, "model block count"),
        ("tap_block_index", 4, "tap block index"),
        ("tap_stream", "pooled_embedding", "fallback"),
        ("tap_boundary", "unknown", "boundary"),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(model, **{field: value})
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(model, model_id="changed-after-freeze")
    with pytest.raises(ValueError, match="confirmation"):
        replace(model.header, claim_scope="confirmation")
    wire = model.to_wire()
    wire["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        decode_contract(canonical_bytes(wire))
    forged = model.to_wire()
    forged["header"]["content_digest"] = "f" * 64
    response = _worker({"op": "roundtrip_model_descriptor", "model": forged}, tmp_path)
    assert response.returncode == 2
    assert "content digest mismatch" in json.loads(response.stdout)["reason"]

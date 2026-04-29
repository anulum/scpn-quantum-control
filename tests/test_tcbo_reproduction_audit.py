# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for TCBO p_h1 Reproduction Audit Runner
"""Tests for the TCBO p_h1 reproduction audit helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_tcbo_reproduction_audit.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "_run_tcbo_reproduction_audit",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit_module = _load_script_module()
classify_observer_source = audit_module.classify_observer_source
deterministic_phase_stream = audit_module.deterministic_phase_stream


def test_classify_observer_source_identifies_delay_embedded_vietoris_rips():
    source = """
    cloud = delay_embed_multi(signal, self.cfg.embed_dim, self.cfg.tau_delay)
    result = _compute_ripser(cloud, maxdim=1)
    """

    classification = classify_observer_source(source)

    assert classification["uses_vietoris_rips_delay_embedding"] is True
    assert classification["uses_coupling_weighted_complex"] is False


def test_classify_observer_source_requires_coupling_and_simplicial_terms():
    source = """
    weight = K_ij * abs(cos_delta)
    filtration = weighted_simplicial_complex(simplex_weights=weight)
    """

    classification = classify_observer_source(source)

    assert classification["uses_coupling_weighted_complex"] is True


def test_deterministic_phase_stream_is_reproducible():
    first = list(
        deterministic_phase_stream(
            kind="incoherent_noise",
            n_layers=4,
            steps=3,
            seed=11,
        )
    )
    second = list(
        deterministic_phase_stream(
            kind="incoherent_noise",
            n_layers=4,
            steps=3,
            seed=11,
        )
    )

    assert len(first) == 3
    assert first[0].shape == (4,)
    assert all(np.allclose(a, b) for a, b in zip(first, second, strict=True))

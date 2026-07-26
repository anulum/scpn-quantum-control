# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio lazy import boundary tests
"""Cold-import tests for the optional Studio platform dependency boundary."""

from __future__ import annotations

import subprocess
import sys
from textwrap import dedent


def test_core_studio_submodule_imports_without_studio_platform() -> None:
    """A core replay module must not import the optional Studio platform."""
    script = dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder

        class _BlockStudioPlatform(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "scpn_studio_platform" or fullname.startswith(
                    "scpn_studio_platform."
                ):
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, _BlockStudioPlatform())

        import scpn_quantum_control.studio as studio

        assert "scpn_quantum_control.studio.benchmark_databank_bundle" not in sys.modules

        from scpn_quantum_control.studio import program_ad_replay_artifact
        from scpn_quantum_control.studio.program_ad_replay_artifact import (
            PROGRAM_AD_REPLAY_SCHEMA,
        )

        assert program_ad_replay_artifact.PROGRAM_AD_REPLAY_SCHEMA == PROGRAM_AD_REPLAY_SCHEMA
        assert studio.REFERENCE_VALIDATION_SCHEMA == (
            "studio.reference-validation-certifications.v1"
        )
        assert "REFERENCE_VALIDATION_SCHEMA" in dir(studio)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

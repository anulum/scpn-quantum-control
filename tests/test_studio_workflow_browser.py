# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — original workflow browser target refusal
"""Refuse invalid original browser hosts before creating transport or browser state."""

import pytest

from tools.studio_workflow_browser import run_workflow_journey


@pytest.mark.parametrize(
    "target",
    [
        "https://127.0.0.1/",
        "http://example.org/",
        "http://localhost/",
        "http://127.0.0.1:9/?token=wrong",
        "http://user@127.0.0.1:9/",
    ],
)
def test_original_workflow_browser_refuses_unowned_or_ambiguous_target(target: str) -> None:
    """Reject an invalid target before launching an actual browser.

    Parameters
    ----------
    target
        Concrete external, encrypted, hostname, query or credential-bearing URL.

    """
    evidence: dict[str, object] = {}
    with pytest.raises(ValueError):
        run_workflow_journey(target, evidence=evidence)
    assert evidence == {}


@pytest.mark.parametrize("source", ["http://127.0.0.1:9/", "http://127.0.0.1:10/nested/"])
def test_original_workflow_source_requires_a_distinct_owned_root(source: str) -> None:
    """Keep source qualification separate from an identical or nested producer.

    Parameters
    ----------
    source
        An identical authority or a non-root source path.

    """
    with pytest.raises(ValueError, match="distinct owned root"):
        run_workflow_journey("http://127.0.0.1:9/", source_url=source)

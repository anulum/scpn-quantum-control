# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- provider job id contract tests
"""Contract tests for provider job-id canonicalisation and validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware import (
    hal_azure,
    hal_cirq,
    hal_dwave,
    hal_ionq,
    hal_iqm,
    hal_oqc,
    hal_pasqal,
    hal_qbraid,
    hal_qiskit,
    hal_quandela,
    hal_quantinuum,
    hal_quera_bloqade,
    hal_rigetti,
    hal_strangeworks,
)


class _WithAttr:
    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_provider_id_extractors_reject_control_characters() -> None:
    bad = _WithAttr(id="job-\n42")
    with pytest.raises(ValueError):
        hal_azure._job_id(bad)
    with pytest.raises(ValueError):
        hal_qbraid._job_id(bad)
    with pytest.raises(ValueError):
        hal_strangeworks._job_id(bad)
    with pytest.raises(ValueError):
        hal_iqm._job_id(_WithAttr(job_id="job-\n42"))

    with pytest.raises(ValueError):
        hal_cirq._provider_job_id(_WithAttr(id="job-\n42"))
    with pytest.raises(ValueError):
        hal_dwave._provider_job_id(_WithAttr(info={"id": "job-\n42"}))
    with pytest.raises(ValueError):
        hal_oqc._provider_job_id(_WithAttr(id="job-\n42"))
    with pytest.raises(ValueError):
        hal_pasqal._provider_job_id(_WithAttr(id="job-\n42"))
    with pytest.raises(ValueError):
        hal_quandela._provider_job_id(_WithAttr(id="job-\n42"))
    with pytest.raises(ValueError):
        hal_quera_bloqade._provider_job_id(_WithAttr(id="job-\n42"))
    with pytest.raises(ValueError):
        hal_quantinuum._provider_job_id(_WithAttr(id="job-\n42"))
    with pytest.raises(ValueError):
        hal_rigetti._provider_job_id(_WithAttr(id="job-\n42"))
    with pytest.raises(ValueError):
        hal_qiskit._provider_job_id(_WithAttr(job_id="job-\n42"), provider_name="qiskit")
    with pytest.raises(ValueError):
        hal_ionq._provider_job_id_from_response({"id": "job-\n42"})


def test_provider_id_extractors_trim_padding_and_preserve_value() -> None:
    padded = _WithAttr(id="  job-42  ")
    assert hal_azure._job_id(padded) == "job-42"
    assert hal_qbraid._job_id(padded) == "job-42"
    assert hal_strangeworks._job_id(padded) == "job-42"
    assert hal_cirq._provider_job_id(padded) == "job-42"
    assert hal_ionq._provider_job_id_from_response({"id": "  job-42  "}) == "job-42"
    assert hal_dwave._provider_job_id(_WithAttr(info={"problem_id": "  job-42  "})) == "job-42"
    assert hal_iqm._job_id(_WithAttr(job_id="  job-42  ")) == "job-42"
    assert hal_oqc._provider_job_id(padded) == "job-42"
    assert hal_pasqal._provider_job_id(padded) == "job-42"
    assert hal_quandela._provider_job_id(padded) == "job-42"
    assert hal_quera_bloqade._provider_job_id(padded) == "job-42"
    assert hal_quantinuum._provider_job_id(padded) == "job-42"
    assert hal_rigetti._provider_job_id(padded) == "job-42"
    assert (
        hal_qiskit._provider_job_id(_WithAttr(job_id="  job-42  "), provider_name="qiskit")
        == "job-42"
    )


def test_provider_id_extractors_reject_object_repr_placeholders() -> None:
    placeholder = _WithAttr(id="<ProviderJob object at 0xDEADBEEF>")

    with pytest.raises(ValueError):
        hal_azure._job_id(placeholder)
    with pytest.raises(ValueError):
        hal_qbraid._job_id(placeholder)
    with pytest.raises(ValueError):
        hal_strangeworks._job_id(placeholder)
    with pytest.raises(ValueError):
        hal_iqm._job_id(_WithAttr(job_id="<ProviderJob object at 0xDEADBEEF>"))
    with pytest.raises(ValueError):
        hal_cirq._provider_job_id(placeholder)
    with pytest.raises(ValueError):
        hal_oqc._provider_job_id(placeholder)
    with pytest.raises(ValueError):
        hal_pasqal._provider_job_id(placeholder)
    with pytest.raises(ValueError):
        hal_quandela._provider_job_id(placeholder)
    with pytest.raises(ValueError):
        hal_quera_bloqade._provider_job_id(placeholder)
    with pytest.raises(ValueError):
        hal_quantinuum._provider_job_id(placeholder)
    with pytest.raises(ValueError):
        hal_rigetti._provider_job_id(placeholder)
    with pytest.raises(ValueError):
        hal_qiskit._provider_job_id(
            _WithAttr(job_id="<ProviderJob object at 0xDEADBEEF>"),
            provider_name="qiskit",
        )
    with pytest.raises(ValueError):
        hal_ionq._provider_job_id_from_response({"id": "<ProviderJob object at 0xDEADBEEF>"})
    with pytest.raises(ValueError):
        hal_dwave._provider_job_id(
            _WithAttr(info={"problem_id": "<ProviderJob object at 0xDEADBEEF>"})
        )

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the ML-DSA-65 implementation
"""Branch and fail-closed tests for the pure-Python ML-DSA-65 implementation.

Covers the NTT/INTT native-engine fallbacks, the hint-decoding rejections, the
half-byte coefficient sampler under the eta=2 parameter regime, the signing
hint-weight rejection, the verification norm-rejection and the external
interface size guards.
"""

from __future__ import annotations

import logging
import sys
import types
import warnings
from typing import Any

import pytest

from scpn_quantum_control.crypto import ml_dsa
from scpn_quantum_control.crypto.ml_dsa import (
    K,
    N,
    _coeff_from_half_byte,
    _encode_message,
    _hint_bit_unpack,
    _intt_python,
    _ntt_python,
    _sample_in_ball,
    intt,
    key_gen,
    ntt,
    sign,
    verify,
)


@pytest.fixture()
def _fresh_dispatch_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the once-per-process dispatch bookkeeping for observability tests."""
    monkeypatch.setattr(ml_dsa, "_ntt_fallback_logged", False)
    monkeypatch.delenv(ml_dsa.REQUIRE_NATIVE_ENV, raising=False)


def _install_raising_engine(monkeypatch: pytest.MonkeyPatch, export: str) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("engine refused the transform")

    stub = types.ModuleType("scpn_quantum_engine")
    setattr(stub, export, _boom)
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)


@pytest.mark.usefixtures("_fresh_dispatch_state")
def test_ntt_falls_back_to_python_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A raising native NTT export falls back with ONE logged warning per process."""
    _install_raising_engine(monkeypatch, "ml_dsa_ntt")

    poly = list(range(N))
    with caplog.at_level(logging.WARNING, logger="scpn_quantum_control.crypto.ml_dsa"):
        assert ntt(poly) == _ntt_python(poly)
        assert ntt(poly) == _ntt_python(poly)
    fallback_records = [r for r in caplog.records if "pure-Python reference path" in r.message]
    assert len(fallback_records) == 1
    assert "ml_dsa_ntt" in fallback_records[0].message


@pytest.mark.usefixtures("_fresh_dispatch_state")
def test_intt_falls_back_to_python_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A raising native INTT export falls back to the Python kernel with a warning."""
    _install_raising_engine(monkeypatch, "ml_dsa_intt")

    poly = list(range(N))
    with caplog.at_level(logging.WARNING, logger="scpn_quantum_control.crypto.ml_dsa"):
        assert intt(poly) == _intt_python(poly)
    assert any("ml_dsa_intt" in r.message for r in caplog.records)


@pytest.mark.usefixtures("_fresh_dispatch_state")
def test_missing_engine_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An absent engine module falls back to Python, naming the unavailable kernel."""
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", None)

    poly = list(range(N))
    with caplog.at_level(logging.WARNING, logger="scpn_quantum_control.crypto.ml_dsa"):
        assert ntt(poly) == _ntt_python(poly)
    assert any("unavailable" in r.message for r in caplog.records)


@pytest.mark.usefixtures("_fresh_dispatch_state")
def test_strict_mode_raises_when_kernel_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCPN_REQUIRE_NATIVE_CRYPTO=1 turns an unavailable kernel into RuntimeError."""
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", None)
    monkeypatch.setenv(ml_dsa.REQUIRE_NATIVE_ENV, "1")

    with pytest.raises(RuntimeError, match="unavailable"):
        ntt(list(range(N)))


@pytest.mark.usefixtures("_fresh_dispatch_state")
def test_strict_mode_raises_when_kernel_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCPN_REQUIRE_NATIVE_CRYPTO=1 turns a failing kernel call into RuntimeError."""
    _install_raising_engine(monkeypatch, "ml_dsa_ntt")
    monkeypatch.setenv(ml_dsa.REQUIRE_NATIVE_ENV, "1")

    with pytest.raises(RuntimeError, match="failed"):
        ntt(list(range(N)))


@pytest.mark.usefixtures("_fresh_dispatch_state")
def test_native_path_is_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A working native kernel produces no fallback log at all."""
    stub = types.ModuleType("scpn_quantum_engine")
    stub.ml_dsa_ntt = _ntt_python  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    poly = list(range(N))
    with caplog.at_level(logging.WARNING, logger="scpn_quantum_control.crypto.ml_dsa"):
        assert ntt(poly) == _ntt_python(poly)
    assert not caplog.records


@pytest.fixture()
def _fresh_research_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the once-per-process research-boundary warning flag."""
    monkeypatch.setattr(ml_dsa, "_research_warning_emitted", False)


@pytest.mark.usefixtures("_fresh_research_warning")
def test_key_gen_emits_research_warning_once() -> None:
    """The first key-material operation warns about the research boundary — once."""
    with pytest.warns(UserWarning, match="research implementation of FIPS 204"):
        pair = key_gen(bytes(32))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sign(pair.secret_key, b"payload")


@pytest.mark.usefixtures("_fresh_research_warning")
def test_research_warning_is_suppressible() -> None:
    """suppress_research_warning=True acknowledges the boundary without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pair = key_gen(bytes(32), suppress_research_warning=True)
        sign(pair.secret_key, b"payload", suppress_research_warning=True)


def test_hint_unpack_rejects_out_of_range_end() -> None:
    """A hint encoding whose end index exceeds OMEGA is rejected."""
    data = bytearray(ml_dsa.OMEGA + K)
    data[ml_dsa.OMEGA] = ml_dsa.OMEGA + 100
    assert _hint_bit_unpack(bytes(data)) is None


def test_hint_unpack_rejects_unsorted_indices() -> None:
    """A hint block whose indices are not strictly increasing is rejected."""
    data = bytearray(ml_dsa.OMEGA + K)
    data[ml_dsa.OMEGA] = 2  # end index for the first row
    data[0] = 5
    data[1] = 3  # non-increasing pair inside the block
    assert _hint_bit_unpack(bytes(data)) is None


def test_coeff_from_half_byte_eta_two_regime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under the eta=2 regime the half-byte sampler maps small nibbles to coefficients."""
    monkeypatch.setattr(ml_dsa, "ETA", 2)
    assert _coeff_from_half_byte(0) == 2
    assert _coeff_from_half_byte(7) == 2 - (7 % 5)


def test_encode_message_rejects_long_context() -> None:
    """A context longer than 255 bytes is rejected."""
    with pytest.raises(ValueError, match="context must be at most 255 bytes"):
        _encode_message(b"m", bytes(256))


def test_sign_rejects_wrong_secret_key_size() -> None:
    """A secret key of the wrong size is rejected."""
    with pytest.raises(ValueError, match="secret_key must be"):
        sign(b"too-short", b"message")


def test_sign_rejects_wrong_randomness_size() -> None:
    """A randomness blob that is not 32 bytes is rejected."""
    pair = key_gen(bytes(range(32)))
    with pytest.raises(ValueError, match="randomness must be 32 bytes"):
        sign(pair.secret_key, b"message", randomness=b"short")


def test_sign_retries_when_hint_weight_exceeds_omega(monkeypatch: pytest.MonkeyPatch) -> None:
    """A signing attempt whose hint weight exceeds OMEGA is rejected and retried."""
    pair = key_gen(bytes(range(32)))
    real_make_hint = ml_dsa._make_hint
    forced = {"count": 0}
    limit = K * N

    def _flooded_make_hint(z_arg: int, r_arg: int) -> int:
        if forced["count"] < limit:
            forced["count"] += 1
            return 1
        return real_make_hint(z_arg, r_arg)

    monkeypatch.setattr(ml_dsa, "_make_hint", _flooded_make_hint)
    signature = sign(pair.secret_key, b"hint-weight retry")
    assert isinstance(signature, bytes)
    assert forced["count"] == limit


def test_sample_in_ball_extends_exhausted_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """The challenge sampler re-squeezes the XOF when the initial stream is exhausted.

    A wrapper truncates the first squeeze to nine bytes so the per-coefficient
    loop runs past the buffer and must extend it from the underlying XOF.
    """
    real_factory = ml_dsa._shake256_xof

    def _short_first_factory(seed: bytes) -> Any:
        real = real_factory(seed)
        calls = {"n": 0}

        class _TruncatingXof:
            def digest(self, length: int) -> bytes:
                calls["n"] += 1
                if calls["n"] == 1:
                    return real.digest(length)[:9]
                return real.digest(length)

        return _TruncatingXof()

    monkeypatch.setattr(ml_dsa, "_shake256_xof", _short_first_factory)
    challenge = _sample_in_ball(bytes(32))
    assert len(challenge) == N
    assert sum(1 for value in challenge if value != 0) == ml_dsa.TAU


def test_sign_retries_when_ct0_norm_exceeds_gamma2(monkeypatch: pytest.MonkeyPatch) -> None:
    """A signing attempt whose ct0 norm reaches gamma2 is rejected and retried.

    For the ML-DSA-65 parameter set the ct0 norm is bounded below gamma2, so this
    FIPS 204 safety rejection never fires for valid keys; a one-shot norm override
    on the first hint-bearing attempt exercises the fail-closed contract.
    """
    pair = key_gen(bytes(range(32)))
    real_norm = ml_dsa._centered_norm
    real_make_hint = ml_dsa._make_hint
    state = {"armed": False, "fired": False}

    def _arming_make_hint(z_arg: int, r_arg: int) -> int:
        if not state["fired"]:
            state["armed"] = True
        return real_make_hint(z_arg, r_arg)

    def _one_shot_norm(poly: Any) -> int:
        if state["armed"] and not state["fired"]:
            state["fired"] = True
            state["armed"] = False
            return ml_dsa.GAMMA2
        return real_norm(poly)

    monkeypatch.setattr(ml_dsa, "_make_hint", _arming_make_hint)
    monkeypatch.setattr(ml_dsa, "_centered_norm", _one_shot_norm)
    signature = sign(pair.secret_key, b"ct0 norm rejection")
    assert isinstance(signature, bytes)
    assert state["fired"]


def test_verify_rejects_oversized_signature_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    """A signature whose response norm exceeds the bound fails verification."""
    pair = key_gen(bytes(range(32)))
    message = b"norm-rejection"
    signature = sign(pair.secret_key, message)
    assert verify(pair.public_key, message, signature)

    monkeypatch.setattr(ml_dsa, "_centered_norm", lambda _poly: ml_dsa.GAMMA1)
    assert verify(pair.public_key, message, signature) is False

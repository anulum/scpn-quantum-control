"""Tests for bridge/sc_to_quantum.py."""
import numpy as np
import pytest

from scpn_quantum_control.bridge.sc_to_quantum import (
    angle_to_probability,
    bitstream_to_statevector,
    measurement_to_bitstream,
    probability_to_angle,
)


def test_prob_angle_roundtrip():
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        theta = probability_to_angle(p)
        assert abs(angle_to_probability(theta) - p) < 1e-12


def test_boundary_clipping():
    assert probability_to_angle(-0.1) == probability_to_angle(0.0)
    assert probability_to_angle(1.5) == probability_to_angle(1.0)


def test_bitstream_to_statevector_norm():
    bits = np.array([1, 1, 0, 0, 1, 0])
    sv = bitstream_to_statevector(bits)
    assert abs(np.linalg.norm(sv) - 1.0) < 1e-12


def test_bitstream_to_statevector_all_ones():
    bits = np.ones(100, dtype=np.uint8)
    sv = bitstream_to_statevector(bits)
    # p=1.0 -> theta=pi -> sv = [0, 1]
    assert abs(sv[1] - 1.0) < 1e-10


def test_measurement_to_bitstream_length():
    counts = {"0": 70, "1": 30}
    bs = measurement_to_bitstream(counts, 256)
    assert len(bs) == 256
    assert set(np.unique(bs)).issubset({0, 1})


def test_measurement_to_bitstream_bias():
    counts = {"0": 0, "1": 1000}
    bs = measurement_to_bitstream(counts, 500)
    assert np.all(bs == 1)

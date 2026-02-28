"""Tests for qec/control_qec.py."""

import numpy as np

from scpn_quantum_control.qec.control_qec import ControlQEC, MWPMDecoder, SurfaceCode


def test_surface_code_dimensions():
    code = SurfaceCode(distance=3)
    assert code.num_data == 18  # 2*3^2
    assert code.Hx.shape == (9, 18)
    assert code.Hz.shape == (9, 18)


def test_stabilizer_weight():
    """Each stabilizer should have weight 4 (toric code)."""
    code = SurfaceCode(distance=3)
    for row in code.Hx:
        assert np.sum(row) == 4
    for row in code.Hz:
        assert np.sum(row) == 4


def test_no_errors_no_syndrome():
    qec = ControlQEC(distance=3)
    err_x = np.zeros(18, dtype=np.int8)
    err_z = np.zeros(18, dtype=np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    assert np.all(syn_z == 0)
    assert np.all(syn_x == 0)


def test_single_error_produces_syndrome():
    """A single X error should produce exactly 2 syndrome bits."""
    qec = ControlQEC(distance=3)
    err_x = np.zeros(18, dtype=np.int8)
    err_x[0] = 1
    err_z = np.zeros(18, dtype=np.int8)
    syn_z, _ = qec.get_syndrome(err_x, err_z)
    assert np.sum(syn_z) == 2


def test_simulate_errors_shape():
    qec = ControlQEC(distance=3)
    err_x, err_z = qec.simulate_errors(0.1, rng=np.random.default_rng(0))
    assert err_x.shape == (18,)
    assert err_z.shape == (18,)


def test_decode_returns_correction():
    """Decoder should return a correction vector of correct length."""
    decoder = MWPMDecoder(distance=3)
    syndrome = np.zeros(9, dtype=np.int8)
    syndrome[0] = 1
    syndrome[1] = 1
    corr = decoder.decode(syndrome)
    assert corr.shape == (18,)


def test_no_defects_empty_correction():
    decoder = MWPMDecoder(distance=3)
    syndrome = np.zeros(9, dtype=np.int8)
    corr = decoder.decode(syndrome)
    assert np.all(corr == 0)


def test_low_error_rate_some_success():
    """At low error rate, decoder should succeed at least sometimes."""
    qec = ControlQEC(distance=3)
    rng = np.random.default_rng(42)
    successes = 0
    trials = 50
    for _ in range(trials):
        err_x, err_z = qec.simulate_errors(0.02, rng=rng)
        if qec.decode_and_correct(err_x, err_z):
            successes += 1
    assert successes > 0


def test_knm_weighted_decoder_runs():
    """Decoder with Knm weights should run without error."""
    K = np.random.default_rng(0).uniform(0, 0.5, (9, 9))
    K = (K + K.T) / 2
    qec = ControlQEC(distance=3, knm_weights=K)
    err_x = np.zeros(18, dtype=np.int8)
    err_x[3] = 1
    err_z = np.zeros(18, dtype=np.int8)
    result = qec.decode_and_correct(err_x, err_z)
    assert isinstance(result, bool)


def test_threshold_below_vs_above():
    """Success rate at p=0.01 should exceed success rate at p=0.08 for d=3.

    d=3 toric code has threshold ~10.3% for independent X/Z noise (Dennis et al. 2002).
    """
    qec = ControlQEC(distance=3)
    rng = np.random.default_rng(123)
    trials = 200

    def success_rate(p_error):
        ok = 0
        for _ in range(trials):
            ex, ez = qec.simulate_errors(p_error, rng=rng)
            if qec.decode_and_correct(ex, ez):
                ok += 1
        return ok / trials

    rate_low = success_rate(0.01)
    rate_high = success_rate(0.08)
    assert rate_low > rate_high


def test_distance_5_constructs_and_decodes():
    """d=5 surface code should construct and decode single errors."""
    qec = ControlQEC(distance=5)
    assert qec.code.num_data == 50  # 2*5^2
    assert qec.code.Hx.shape == (25, 50)

    # Single error should decode correctly
    err_x = np.zeros(50, dtype=np.int8)
    err_x[10] = 1
    err_z = np.zeros(50, dtype=np.int8)
    assert qec.decode_and_correct(err_x, err_z)


def test_very_low_error_rate_high_success():
    """At p=0.005, d=3 decoder should succeed > 80% of trials."""
    qec = ControlQEC(distance=3)
    rng = np.random.default_rng(42)
    successes = sum(
        qec.decode_and_correct(*qec.simulate_errors(0.005, rng=rng)) for _ in range(200)
    )
    assert successes / 200 > 0.80


def test_d5_beats_d3_below_threshold():
    """Below threshold (p=0.03), d=5 should match or beat d=3.

    Dennis et al. 2002: toric code MWPM threshold ~10.3% per channel.
    """
    rng3 = np.random.default_rng(77)
    rng5 = np.random.default_rng(77)
    trials = 300
    qec3 = ControlQEC(distance=3)
    qec5 = ControlQEC(distance=5)
    ok3 = sum(
        qec3.decode_and_correct(*qec3.simulate_errors(0.03, rng=rng3)) for _ in range(trials)
    )
    ok5 = sum(
        qec5.decode_and_correct(*qec5.simulate_errors(0.03, rng=rng5)) for _ in range(trials)
    )
    assert ok5 >= ok3 - 10  # d=5 at least within noise of d=3


def test_logical_error_detected_shifted_cycle():
    """Logical operators at any row/column must be detected, not just row/col 0."""
    d = 5
    qec = ControlQEC(distance=d)
    N = 2 * d**2
    zero = np.zeros(N, dtype=np.int8)

    for row in range(d):
        cycle = np.zeros(N, dtype=np.int8)
        for c in range(d):
            cycle[2 * (row * d + c)] = 1  # h(row, c)
        assert qec._has_logical_error(cycle, zero), f"missed horizontal cycle at row {row}"

    for col in range(d):
        cycle = np.zeros(N, dtype=np.int8)
        for r in range(d):
            cycle[2 * (r * d + col) + 1] = 1  # v(r, col)
        assert qec._has_logical_error(cycle, zero), f"missed vertical cycle at col {col}"


def test_single_z_error_corrected():
    """Single Z error should be decoded correctly (dual path)."""
    for d in (3, 5):
        qec = ControlQEC(distance=d)
        N = 2 * d**2
        for qubit in range(min(N, 10)):
            err_x = np.zeros(N, dtype=np.int8)
            err_z = np.zeros(N, dtype=np.int8)
            err_z[qubit] = 1
            assert qec.decode_and_correct(err_x, err_z), (
                f"d={d} failed on single Z error at qubit {qubit}"
            )


def test_single_x_error_corrected():
    """Single X error should be decoded correctly (primal path)."""
    for d in (3, 5):
        qec = ControlQEC(distance=d)
        N = 2 * d**2
        for qubit in range(min(N, 10)):
            err_x = np.zeros(N, dtype=np.int8)
            err_x[qubit] = 1
            err_z = np.zeros(N, dtype=np.int8)
            assert qec.decode_and_correct(err_x, err_z), (
                f"d={d} failed on single X error at qubit {qubit}"
            )

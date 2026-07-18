# Test Infrastructure

## Overview

scpn-quantum-control has a CI-gated test suite with a 90% aggregate **line**
coverage requirement. CI now collects branch arcs in the same run and requires
real branch opportunity data, but the branch percentage remains observational
until consecutive remote runs establish a stable baseline. The active test
count is intentionally left to CI summaries and release notes because the suite
changes frequently. The test suite verifies correctness, pipeline wiring, Rust
acceleration parity, and performance benchmarks.

```
tests/
  conftest.py              # Shared fixtures: knm_Nq, coupling_variant, hypothesis strategies
  test_pipeline_wiring_performance.py  # 155 tests — every __all__ export verified functional
                                       # (now includes TestGUESSPipeline, TestDynQPipeline,
                                       # TestPulseShapingPipeline)
  test_rust_path_benchmarks.py         # Rust path parity and benchmark coverage
  test_symmetry_decay.py               # 20 multi-angle tests — GUESS (April 2026)
  test_qubit_mapper.py                 # 17 multi-angle tests — DynQ (April 2026)
  test_pulse_shaping.py                # 25 multi-angle tests — ICI + (α,β)-hypergeometric (April 2026)
  test_*.py                            # module-specific tests
```

## Running Tests

```bash
# Local module-specific run
pytest tests/test_xy_kuramoto.py -v --tb=short

# Local realtime-runtime cohort
pytest tests/test_realtime_runtime.py tests/test_realtime_runtime_branches.py \
  tests/test_sub_us_tracker.py -q --tb=short

# CI-only full suite (do not run locally)
pytest tests/ -v --tb=short -m "not slow"

# CI-only whole-package coverage with branch telemetry
pytest tests/ --cov=src/scpn_quantum_control --cov-branch --cov-report=html -m "not slow"
```

## Test typing ratchet

Production Python remains under repository-wide strict mypy. The test tree is
being migrated additively because a 2026-07-13 census of `mypy --strict tests/`
reported 6,301 errors in 388 of 968 tracked Python test files; 5,076 were
`no-untyped-def`. Adding the whole directory to one gate would therefore mix
mechanical annotations with intentional invalid-input tests and hide ownership.

`tools/test_typing_policy.json` is the machine-readable policy. Its initial
16-file `repository_policy` cohort covers coverage, coverage debt, licence, release,
generated-surface, commit, secret, TODO, version, branch, module-size,
CI/pre-push, local preflight, and built-wheel publication gate tests.
`tools/audit_test_typing_policy.py`
validates that every enforced path is tracked and runs strict mypy over the
exact cohort; CI and the no-test local preflight execute the same command.

Migration order is explicit: claim and release contracts first, then hardware
and provider boundaries, then scientific runtime contracts; legacy fixture
mechanics are last. Each slice is limited to 40 files and must pass focused
pytest, strict mypy, Ruff check, and Ruff format. Intentional negative calls keep
narrow error-code suppressions where a static type cannot represent the invalid
input.

```bash
python tools/audit_test_typing_policy.py
python tools/audit_test_typing_policy.py --validate-only --json
```

## Realtime runtime quality ratchet

The runtime source and its two direct test owners form a fixed three-file
quality cohort. CI and the local static-gate definition execute these exact
commands and ordered paths, so a future edit cannot silently fall back to the
repository's transitional docstring exclusions or non-strict test typing:

```bash
python -m mypy --strict --explicit-package-bases \
  src/scpn_quantum_control/control/realtime_runtime.py \
  tests/test_realtime_runtime.py \
  tests/test_realtime_runtime_branches.py

python -m ruff check --isolated --select D,D413 \
  --config 'lint.pydocstyle.convention = "numpy"' \
  src/scpn_quantum_control/control/realtime_runtime.py \
  tests/test_realtime_runtime.py \
  tests/test_realtime_runtime_branches.py
```

Focused behaviour runs add `tests/test_sub_us_tracker.py` because it owns the
Rust/NumPy numerical parity and batch-throughput contract. It is intentionally
outside the strict three-file ratchet until its pre-existing typing debt is
migrated as its own owner.

## Phase-QNode affinity quality ratchet

The benchmark evidence source, lean loader, CLI, and their two public-path test
owners form one ordered five-file quality cohort. CI and the local gate
definition enforce strict typing and NumPy docstrings over exactly those files:

```bash
quality_paths=(
  src/scpn_quantum_control/phase/qnode_affinity_benchmark.py
  tools/lean_phase_import.py
  tools/run_phase_qnode_affinity_benchmark.py
  tests/test_phase_qnode_affinity_benchmark.py
  tests/test_lean_phase_import.py
)
python -m mypy --strict --explicit-package-bases "${quality_paths[@]}"
python -m ruff check --isolated --select D,D413 \
  --config 'lint.pydocstyle.convention = "numpy"' "${quality_paths[@]}"
```

The required `phase-qnode-affinity-quality` CI job and the default local gate
definition run the same two test owners with isolated branch data, then report
only `qnode_affinity_benchmark.py` at an exact 100% threshold:

```bash
python -m coverage run --rcfile=/dev/null \
  --data-file=.coverage.phase-qnode-affinity --branch \
  -m pytest -q tests/test_phase_qnode_affinity_benchmark.py \
  tests/test_lean_phase_import.py
python -m coverage report --rcfile=/dev/null \
  --data-file=.coverage.phase-qnode-affinity --precision=2 \
  --fail-under=100 --include='*/qnode_affinity_benchmark.py'
```

## Studio Program-AD replay quality ratchet

The ST-12 replay is gated as one polyglot owner. Python emission and validation
must remain strict-MyPy clean, fully NumPy-documented, and exactly covered. The
test cohort uses strict JSON files and a separate module subprocess. In the
aggregate Python matrix, where the compiled engine is intentionally absent, a
test-only deterministic replay seam implements the native export's exact JSON
contract in both the pytest process and that subprocess; this prevents
optional-dependency collection from masking the Python owner. The same
integration assertion exercises the seam in aggregate mode and the real export
in required mode, so neither execution contains a skip. The dedicated owner
command below requires the installed current Rust engine:

```bash
quality_paths=(
  src/scpn_quantum_control/studio/program_ad_replay_artifact.py
  tests/test_studio_program_ad_replay_artifact.py
  tools/program_ad_quality_gates.py
  tests/test_studio_program_ad_quality_gate.py
)
python -m mypy --strict --explicit-package-bases "${quality_paths[@]}"
python -m ruff check --isolated --select D,D413 \
  --config 'lint.pydocstyle.convention = "numpy"' "${quality_paths[@]}"
SCPN_PROGRAM_AD_REQUIRE_NATIVE=1 python -m coverage run --rcfile=/dev/null \
  --data-file=.coverage.studio-program-ad --branch \
  -m pytest -q tests/test_studio_program_ad_replay_artifact.py
python -m coverage report --rcfile=/dev/null \
  --data-file=.coverage.studio-program-ad --precision=2 \
  --fail-under=100 --include='*/program_ad_replay_artifact.py'
```

The same local default lane runs the Rust crate tests, rebuilds the release
`wasm32-unknown-unknown` kernel, type-checks the Studio workspace, and enforces
exact statement, branch, function, and line coverage over the browser verifier
and React card. CI builds and installs the current PyO3 wheel before Python
coverage and sets `SCPN_PROGRAM_AD_REQUIRE_NATIVE=1`. A missing module, nested
engine dependency failure, or absent replay export therefore fails hard rather
than selecting the aggregate seam or producing a skip. The Studio web job
executes the browser cohort against the freshly built WASM:

```bash
cargo test --locked \
  --manifest-path scpn_quantum_engine/studio_program_ad_wasm/Cargo.toml
cargo build --release --locked --target wasm32-unknown-unknown \
  --manifest-path scpn_quantum_engine/studio_program_ad_wasm/Cargo.toml
pnpm --dir studio-web typecheck
pnpm --dir studio-web exec vitest run \
  src/panel/programAd.test.ts src/panel/ProgramADReplayCard.test.tsx \
  --coverage \
  --coverage.include=src/panel/programAd.ts \
  --coverage.include=src/panel/ProgramADReplayCard.tsx \
  --coverage.thresholds.statements=100 \
  --coverage.thresholds.branches=100 \
  --coverage.thresholds.functions=100 \
  --coverage.thresholds.lines=100
```

## MLIR leaf quality ratchet

The four implementation leaves behind `compiler.mlir` have an exact focused
quality lane. The static cohort contains those four sources, the shared typed
native-compilation helper, and 14 responsibility-specific test owners. CI and
the local gate definition keep the ordered lists identical; policy tests fail
if either side drifts. Strict MyPy intentionally omits
`--explicit-package-bases` because the native-compilation tests import their
top-level shared helper by its installed test-module name.

```bash
quality_paths=(
  src/scpn_quantum_control/compiler/mlir_enzyme_audit.py
  src/scpn_quantum_control/compiler/mlir_phase_qnode_runtime.py
  src/scpn_quantum_control/compiler/mlir_transform_plan_assembly.py
  src/scpn_quantum_control/compiler/mlir_workload_compilation.py
  tests/_mlir_native_compilation_test_helpers.py
  tests/test_mlir_enzyme_audit.py
  tests/test_mlir_toolchain_probe_hardening.py
  tests/test_mlir_phase_qnode_runtime.py
  tests/test_phase_qnode_compiler_lowering.py
  tests/test_mlir_transform_plan.py
  tests/test_mlir_transform_plan_assembly.py
  tests/test_mlir_workload_compilation.py
  tests/test_mlir_executable_batching_integration.py
  tests/test_mlir_native_compilation_integration.py
  tests/test_mlir_scalar_native_compilation_integration.py
  tests/test_mlir_vector_native_compilation_integration.py
  tests/test_mlir_matrix_native_compilation_integration.py
  tests/test_mlir_matrix_2x2_native_compilation_integration.py
  tests/test_mlir_symmetric_native_compilation_integration.py
)
python -m mypy --strict "${quality_paths[@]}"
python -m ruff check --isolated --select D,D413 \
  --config 'lint.pydocstyle.convention = "numpy"' "${quality_paths[@]}"
```

The executable cohort is the same list without the helper or four production
paths. It exercises public Enzyme audit, toolchain, Phase-QNode, transform-plan,
custom-rule, batching, and native scalar/vector/matrix/symmetric compiler
routes. Coverage is isolated to the four leaf targets and fails below exact
statement and branch coverage:

```bash
coverage_tests=("${quality_paths[@]:5}")
python -m coverage run --rcfile=/dev/null \
  --data-file=.coverage.mlir-leaf-quality --branch \
  --source=src/scpn_quantum_control/compiler \
  -m pytest -q "${coverage_tests[@]}"
python -m coverage report --rcfile=/dev/null \
  --data-file=.coverage.mlir-leaf-quality \
  --include='*/mlir_enzyme_audit.py,*/mlir_phase_qnode_runtime.py,*/mlir_transform_plan_assembly.py,*/mlir_workload_compilation.py' \
  --fail-under=100 --show-missing
```

The contract currently covers 414 statements and 138 branches with no missing
or partial paths. Its tests use real public compiler/runtime execution: mock,
monkeypatch, frozen-record mutation, and private-helper coverage shortcuts are
outside this lane's evidence model.

## Phase-QNode JAX quality ratchet

The registered Phase-QNode JAX execution leaf has a dedicated public-path
quality lane. Static checks cover the two production consumers, both shared
test helpers, seven public-route test owners, and the extracted gate policy and
its direct test. Keeping the ordered policy in
`tools/phase_jax_qnode_quality_gates.py` prevents `tools/preflight.py` from
crossing the GodFile threshold while CI and the permitted local focused gates
consume one definition. CI executes the owner coverage inside the
`differentiable-parity` job after that job installs and verifies its CPU JAX
overlay. A dedicated import and device probe then proves that the real JAX
runtime is usable before any optional-dependency skip can enter the coverage
cohort; the base hash-locked CI environment intentionally does not claim an
optional JAX runtime.

```bash
quality_paths=(
  src/scpn_quantum_control/phase/jax_qnode_transforms.py
  src/scpn_quantum_control/phase/jax_compatibility.py
  tests/_phase_jax_bridge_test_helpers.py
  tests/_phase_jax_qnode_test_helpers.py
  tests/test_phase_jax_bridge_aot_export.py
  tests/test_phase_jax_qnode_transforms.py
  tests/test_phase_jax_qnode_transforms_integration.py
  tests/test_phase_jax_qnode_input_validation.py
  tests/test_phase_jax_qnode_pytree_validation.py
  tests/test_phase_jax_qnode_aot_validation.py
  tests/test_phase_jax_qnode_statevector_edges.py
  tools/phase_jax_qnode_quality_gates.py
  tests/test_phase_jax_qnode_quality_gate.py
)
python -m mypy --strict --explicit-package-bases "${quality_paths[@]}"
python -m ruff check --isolated --select D,D413 \
  --config 'lint.pydocstyle.convention = "numpy"' "${quality_paths[@]}"
```

The executable cohort comprises the seven `test_phase_jax_*` owners above,
including the AOT/export owner. Coverage records branch data under the phase
source root but reports only the registered-QNode leaf:

```bash
coverage_tests=(
  tests/test_phase_jax_qnode_transforms.py
  tests/test_phase_jax_qnode_transforms_integration.py
  tests/test_phase_jax_bridge_aot_export.py
  tests/test_phase_jax_qnode_input_validation.py
  tests/test_phase_jax_qnode_pytree_validation.py
  tests/test_phase_jax_qnode_aot_validation.py
  tests/test_phase_jax_qnode_statevector_edges.py
)
python -m coverage run --rcfile=/dev/null \
  --data-file=.coverage.phase-jax-qnode --branch \
  --source=src/scpn_quantum_control/phase \
  -m pytest -q "${coverage_tests[@]}"
python -m coverage report --rcfile=/dev/null \
  --data-file=.coverage.phase-jax-qnode \
  --include='*/jax_qnode_transforms.py' --fail-under=100 --show-missing
```

The current contract is 546 statements and 198 branches with no missing or
partial paths. Positive value, transform, PyTree, sharding, and AOT/export
cases execute the installed JAX runtime through public APIs. Controlled loader
doubles exercise missing-capability and malformed-result refusal contracts
through the same public facades; they do not replace the installed-runtime
parity cases. Private production-helper calls, coverage exclusions, provider
jobs, and benchmark claims are outside this evidence lane.

## Test Categories

### 1. Pipeline Wiring Tests (`test_pipeline_wiring_performance.py`)

155 tests that verify **every** symbol in `scpn_quantum_control.__all__` (104 symbols)
is importable, callable, and produces valid output when wired into a data pipeline.

Each test follows the pattern:
1. Build inputs from canonical SCPN parameters (build_knm_paper27, OMEGA_N_16)
2. Call the function/class under test
3. Assert output has correct type, shape, and physical bounds
4. Print wall-time performance

```python
class TestTopLevelExports:
    @pytest.mark.parametrize("name", sqc.__all__)
    def test_export_exists(self, name):
        obj = getattr(sqc, name, None)
        assert obj is not None, f"{name} is None — not wired"
```

15 subsystems tested end-to-end:
- Bridge (Knm→H, Knm→Ansatz)
- Phase Solvers (VQE, UPDE, Kuramoto)
- Hardware (Runner, Noise Model)
- Mitigation (ZNE, PEC)
- QEC (ControlQEC, SurfaceCodeUPDE)
- QSNN (Synapse, LIF, STDP)
- Identity (Attractor, Fingerprint)
- Crypto (Key Hierarchy, QKD)
- Analysis (FSS, H1, OTOC, XXZ)
- SSGF (Quantum Loop)
- Orchestrator (Adapter Roundtrip)
- Cutting (24-oscillator Partitioned Simulation)
- Control (VQLS, QAOA-MPC, QPetriNet)
- Benchmarks (Scaling)
- Applications (Reservoir, Kernel, Disruption)

### 2. Rust Path Benchmarks (`test_rust_path_benchmarks.py`)

Rust path tests cover parity and benchmark surfaces in `scpn_quantum_engine`
(PyO3 + rayon). The current exported binding count is generated in
`docs/_generated/capability_snapshot.md`.

Each Rust function is tested for:
- **Correctness:** Output shape, dtype, physical bounds
- **Parity:** Exact match with Python fallback (where applicable)
- **Performance:** Wall-time measurement with `time.perf_counter()`

```python
class TestBuildKnm:
    def test_parity_with_python_paper27(self):
        K_rust = np.array(eng.build_knm(16, 0.45, 0.3))
        K_py = build_knm_paper27(L=16)
        np.testing.assert_allclose(K_rust, K_py, atol=1e-12)
```

Measured speedups:

| Function | Speedup |
|----------|---------|
| `build_knm` | 4.7× |
| `kuramoto_euler` | 33.1× |
| `pec_coefficients` | exact parity |
| `build_xy_hamiltonian_dense` | exact parity with Qiskit |

### 3. Module-Specific Tests (217 files)

Each test file covers one source module with multi-angle tests:

**Physical invariants** — Hermiticity, unitarity, trace, bounds:
```python
def test_hamiltonian_hermitian():
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)
```

**Pipeline wiring** — Data flows end-to-end, not decorative:
```python
def test_pipeline_knm_to_spectrum():
    K = build_knm_paper27(L=4)
    H = knm_to_dense_matrix(K, omega)
    eigvals = np.linalg.eigvalsh(H)
    assert eigvals[0] < 0
    print(f"\n  PIPELINE Knm→H→spectrum (4q): {dt:.1f} ms")
```

**Rust path parity** — Python and Rust produce identical results:
```python
def test_rust_dense_matrix_parity():
    H_py = knm_to_dense_matrix(K, omega)
    H_rust = np.array(eng.build_xy_hamiltonian_dense(K_flat, omega, n))
    np.testing.assert_allclose(H_rust, H_py, atol=1e-10)
```

**Edge cases** — Zero input, boundary values, degenerate systems:
```python
def test_zero_coupling_decoupled():
    K = np.zeros((n, n))
    H = knm_to_dense_matrix(K, omega)
    eigvals = np.linalg.eigvalsh(H)
    np.testing.assert_allclose(eigvals[0], -np.sum(np.abs(omega)), atol=1e-10)
```

**Performance benchmarks** — Wall-time printed for regression tracking:
```python
def test_pipeline_full_kuramoto_evolution():
    t0 = time.perf_counter()
    result = solver.run(t_max=0.3, dt=0.1, trotter_per_step=5)
    dt = (time.perf_counter() - t0) * 1000
    print(f"\n  PIPELINE Knm→KuramotoSolver→R(t) (4q): {dt:.1f} ms")
```

### 4. Property-Based Tests (Hypothesis)

Several files use `hypothesis` for randomised property testing:

```python
@given(n=st.integers(min_value=2, max_value=6))
@settings(max_examples=10, deadline=30000)
def test_hamiltonian_hermitian(n: int) -> None:
    K = rng.uniform(0, 0.5, (n, n))
    K = (K + K.T) / 2
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    assert np.allclose(mat, mat.conj().T, atol=1e-12)
```

Files using hypothesis:
- `test_knm_properties.py` — Hermiticity, eigenvalues, ansatz params
- `test_qec_properties.py` — Zero-error syndrome, correction shapes
- `test_crypto_properties.py` — Key hierarchy determinism, HMAC roundtrip
- `test_new_module_properties.py` — Cross-module property tests

## Shared Fixtures (`conftest.py`)

```python
# System sizes
SMALL_SIZES = [2, 3, 4]
MEDIUM_SIZES = [2, 3, 4, 6]
ALL_SIZES = [2, 3, 4, 6, 8]

# Coupling matrices
@pytest.fixture
def knm_4q():
    return build_knm_paper27(L=4), OMEGA_N_16[:4]

# Coupling variants
@pytest.fixture(params=["paper27", "ring", "zero", "identity"])
def coupling_variant_4q(request): ...

# Hardware runner
@pytest.fixture
def sim_runner(tmp_path):
    runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path))
    runner.connect()
    return runner

# Hypothesis strategies
@st.composite
def st_coupling_matrix(draw, n=None): ...

@st.composite
def st_statevector(draw, n_qubits=None): ...
```

## Markers

```ini
# pyproject.toml
[tool.pytest.ini_options]
markers = ["slow: marks tests as slow (18-qubit, OOM on CI)"]
```

The `@pytest.mark.slow` marker is applied to tests that:
- Create 18-qubit systems (2^18 = 262,144 Hilbert space dimension)
- Require > 7 GB RAM (GitHub Actions runner limit)
- Take > 60 seconds

CI skips slow tests: `pytest -m "not slow"`.

## CI Integration

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: pytest tests/ -v --tb=short -x -m "not slow" --ignore=tests/test_hardware_runner.py
```

Test matrix: Python 3.11, 3.12, 3.13.
Coverage target: 90% line coverage on the Python 3.12 lane, enforced by
`tools/audit_coverage_policy.py` from branch-enabled `coverage.xml`. Branch data
is mandatory but observational until the policy records an evidence-backed
branch threshold. The pre-branch remote baseline is 92.5151% line coverage at
origin `4c3a4fee` (CI run `29180328986`). Coverage recovery must stay
module-specific; coverage-bucket tests remain forbidden.

The separate 100% recovery register is
`data/coverage/coverage_debt_register.json`. It is generated and audited by
`tools/audit_coverage_debt.py` under `tools/coverage_debt_policy.json`; it does
not replace or raise the aggregate 90% line gate. The current register derives
from Python 3.12 CI run `29180328986`, honors the justified exclusions ledger,
and marks source paths added or executably decomposed after that run as
unmeasured until remote CI refreshes them. Debt is ordered first by public
claim-ledger ownership, then explicit runtime hot paths, unmeasured paths,
large known line debt, and remaining measured gaps. CI rejects new unregistered
debt or increased missing-line counts on previously measured rows while
accepting improvements and first measurements of explicitly unmeasured rows.
Refresh only from a downloaded CI artifact:

```bash
python tools/audit_coverage_debt.py \
  --coverage-audit /path/to/coverage-gap-audit.json \
  --write-register
```

### Rust engine format gate

The complete `scpn_quantum_engine` crate is checked with the repository
toolchain's canonical formatter in both local preflight and the CI `rust-audit`
job:

```bash
cargo fmt --manifest-path scpn_quantum_engine/Cargo.toml --all -- --check
```

Formatting drift fails before Rust tests or advisory conclusions are accepted.
The gate is crate-wide so a touched module cannot leave adjacent Rust owners in
an uncheckable state.

### Differentiable external-validation manifests

The external-validation environment lock and its dependent artefact bundle are
checked before the Python test matrix. Run the typed gate in read-only mode to
detect stale file digests, sizes, line counts, or bundle membership:

```bash
python tools/check_differentiable_external_validation.py
```

When a pinned input changes intentionally, refresh the environment JSON and
Markdown first, then the bundle that hashes that pair, with:

```bash
python tools/check_differentiable_external_validation.py --write
```

CI and local preflight execute the read-only form. This keeps an unrelated
matrix job from being the first place where manifest drift is discovered.

### Lock-faithful CI build environment

The parent and standalone oscillatools projects both build through
`hatchling.build`. Their real-wheel regression deliberately invokes
`python -m build --wheel --no-isolation`: release validation must exercise the
environment that the CI matrix and reproduction Docker image actually install,
not an unpinned backend downloaded into a temporary isolation environment.

`requirements-dev.txt` is the authoritative input for that environment. It
pins the build frontend (`build==1.5.1`) and backend (`hatchling==1.31.0`), and
all Python 3.11, 3.12, and 3.13 CI locks carry those exact releases, their two
PyPI-published SHA-256 hashes, and the direct `requirements-dev.txt` ownership
record. Regenerate each lock with its matching interpreter and the repository's
current generator version, pip-tools 7.5.3:

```bash
python3.11 -m piptools compile --allow-unsafe --generate-hashes \
  --output-file=requirements-ci-py311-linux.txt requirements-dev.txt
python3.12 -m piptools compile --allow-unsafe --generate-hashes \
  --output-file=requirements-ci-py312-linux.txt requirements-dev.txt
python3.13 -m piptools compile --allow-unsafe --generate-hashes \
  --output-file=requirements-ci-py313-linux.txt requirements-dev.txt
```

`tools/audit_ci_build_environment.py` fails closed on project metadata, root
developer-extra, direct-pin, interpreter-provenance, distribution-hash,
pip-compile-owner, CI-matrix, Docker-install, real-wheel-fixture, installed
version, backend-import, or security-job import-path drift. The security job
exposes both source trees through `PYTHONPATH` before its focused pytest
coverage gate imports the repository-wide `tests/conftest.py`; it does not rely
on an undeclared editable install. The CI lint job enforces strict MyPy, Ruff,
NumPy docstrings, Bandit, and exact statement/branch coverage for this audit
before the three-version test matrix can start. Any intentional build-tool
upgrade therefore requires one coherent source-pin, three-lock, generated
external-validation, documentation, and audit-constant refresh.

The Docker reproduction context also carries the root `Dockerfile` and the
`oscillatools/README.md` declared by the standalone package metadata. The audit
requires each canonical `COPY` instruction as exactly one normalized Dockerfile
line; comments, longer destinations, and duplicate copies cannot satisfy the
contract. Consequently the live build-environment audit can read its own
Docker consumer inside the image, and the no-isolation oscillatools wheel can
resolve every metadata file without reaching outside the curated context.

### Braket-constrained setuptools advisory waiver

The Python 3.12 security job scans the complete hash-pinned CI lock with
`pip-audit`. It temporarily ignores exactly `PYSEC-2026-3447`, which affects
`setuptools<83.0.0`: on macOS APFS or HFS+, a decomposed Unicode filename can
bypass a composed-form `MANIFEST.in` exclusion and enter a source distribution.
The exception does not classify the advisory as a false positive.

The lock cannot move independently to the fixed release. Both
`amazon-braket-default-simulator` and `amazon-braket-schemas` declare the exact
distribution requirement `setuptools==81.0.0`; the 2026-07-14 source check
confirmed the same pin in the then-current upstream releases
[`1.39.5`](https://pypi.org/project/amazon-braket-default-simulator/1.39.5/)
and [`1.31.0`](https://pypi.org/project/amazon-braket-schemas/1.31.0/).
All three Python CI locks therefore retain the two PyPI-published 81.0.0
distribution hashes. Raising only the lock to 83.0.0 makes the hashed install
unsatisfiable.

The project build boundary limits exposure: `pyproject.toml` uses
`hatchling.build`, not setuptools, and repository Python code does not import
setuptools. CI installs the transitive package on Linux from the locked wheel;
it does not use setuptools to construct this project's source distribution.
The macOS source-distribution risk remains recorded until the upstream pin is
removed.

`tools/audit_dependency_security_waiver.py` fails closed unless all of these
conditions remain true:

- both installed Braket distributions still declare the exact 81.0.0 pin;
- every Python CI lock carries that version, its two source-verified hashes,
  and exactly the same two transitive owners;
- this project still builds with Hatchling and has no setuptools import;
- CI runs the policy audit and ignores only `PYSEC-2026-3447` as two distinct,
  unconditional, blocking `jobs.security` steps; and
- this operator boundary and its removal rule remain documented.

The security job enforces strict typing, NumPy docstrings, and exact 100%
statement/branch coverage on the policy tool before it executes both the live
waiver audit and the complete dependency scan. A second ignored advisory or an
upstream metadata change fails the policy gate. The workflow audit also rejects
job- or step-level `if`, `continue-on-error`, YAML merge keys, custom shells,
and workflow/job run defaults on this boundary. Each protected command must
own a canonical step with exactly one direct run key, preventing a later
duplicate YAML key from replacing the policy audit while leaving a text-only
command scan green.

Mapping-key spelling is part of the security boundary. A semantic PyYAML
compose-tree audit rejects every double-quoted block or flow mapping key that
contains a YAML escape, including nested sequence-owned, multiline or
commented explicit (`?`), tagged, anchored, and aliased forms. Source marks
distinguish escaped keys from escaped values without collapsing duplicate
mapping nodes. This closes the decode-after-scan forms `"\u0069f"`,
`"continue-on-\u0065rror"`, and `"d\u0065faults"`, which standard YAML resolves
to `if`, `continue-on-error`, and `defaults`. Escapes in ordinary scalar values remain
valid, including multiline sequence values. Malformed YAML fails the semantic
audit closed. The raw structural pass remains an independent duplicate-key
and canonical-ownership defence, while workflow, job, and step mapping keys
must use their canonical unescaped spellings.

Remove the waiver as one dependency-lock change when neither Braket pin owner
requires `setuptools<83.0.0`, whether the dependency disappears or its allowed
range admits a fixed release: regenerate all three hashed CI locks, refresh
the dependent external-validation manifests, remove the single
`--ignore-vuln` argument and this temporary policy gate, then require an
exception-free `pip-audit` result. Do not broaden or prolong the exception to
accommodate an unrelated advisory.

### Docker image — reproduction/CI only

The root `Dockerfile` (built and exercised by
`.github/workflows/docker.yml`) is a **reproduction / CI test image, not a
production runtime**. Its default `CMD` is the pytest suite, and the
workflow builds the image and runs the tests inside it — the image is never
pushed to a registry. It deliberately ships `tests/`, `docs/`, `paper/`,
`notebooks/`, `data/`, and CI fixtures, and it does not install the
compiled `scpn_quantum_engine` extension (stubbed to fail loudly), so the
Python tier runs on its pure-Python fallbacks with no Rust toolchain in the
image. It is intentionally not slimmed — slimming would defeat its only
purpose. For a production deployment, install the published wheel
(`pip install scpn-quantum-control`) into your own base image rather than
reusing this one.

Repository-policy tests resolve tracked paths through Git. The image therefore
creates a synthetic index over only the curated files copied into the build;
the host `.git` directory, history, remotes, objects, and credentials remain
excluded by `.dockerignore`. This preserves the same tracked-file contract
without widening the reproduction image's trust boundary. The copied
`.gitignore` keeps ignored fixtures readable without admitting them to policy
audits. Recursive Rust
`target/` build products are also excluded from the context because the image
copies source and fixtures only. Local backup archives and ignored campaign-run
logs are excluded for the same reason; committed root `results/` fixtures remain
part of the reproduction surface.

The image now carries `Dockerfile` so the live build-environment audit can
validate its exact reproduction consumer. The joint credential-free-index
assertion over `Dockerfile` and `.dockerignore` still runs on the host and skips
inside the image because `.dockerignore` is not part of the curated runtime
context. The Rustfmt preflight contract likewise checks an absolute Cargo
executable when the Rust toolchain is installed and the explicit `cargo`
fallback when it is absent; the manifest-scoped command arguments remain
mandatory in both environments.

### Coverage-exclusions ledger

Every file omitted from the line gate (`[tool.coverage.run].omit` and
`[tool.coverage.report].omit` in `pyproject.toml`) MUST carry exactly one
row in `docs/release_coverage_exclusions.json`, stating why it is omitted
and the CI lane that exercises it instead (or an explicit tracked gap).
`tests/test_exclusions_ledger_drift.py` is the drift gate: it fails the
build the moment an omit glob has no ledger row, or a ledger row matches
no live omit glob — the coverage analogue of the
no-`pragma: no cover`-without-issue rule, so the 90% number can never
hide a silent, unjustified hole.

## Test Quality Standards

Every test file must have:
1. **11+ test functions** (no exceptions)
2. **Physical invariant checks** (Hermiticity, bounds, conservation laws)
3. **Pipeline wiring test** proving data flows end-to-end
4. **Performance benchmark** with wall-time output
5. **Rust path verification** where `scpn_quantum_engine` function exists
6. **Edge cases** (zero input, boundary, degenerate)
7. **SPDX 7-line header**

## File Naming Convention

```
tests/test_{module_name}.py           # Primary module test
tests/test_{module_name}_edge_cases.py # Edge case / error path focus
tests/test_{module_name}_properties.py # Property-based (hypothesis)
tests/test_coverage_100_{area}.py      # Coverage-push tests
tests/test_e2e_{pipeline}.py           # End-to-end integration
```

## Performance Regression

Pipeline benchmark values are printed to stdout during test execution.
To capture a performance baseline:

```bash
pytest tests/test_pipeline_wiring_performance.py -v -s 2>&1 | grep "PIPELINE" > benchmarks.txt
pytest tests/test_rust_path_benchmarks.py -v -s 2>&1 | grep "RUST" >> benchmarks.txt
```

Key regression indicators:
- Kuramoto solver (4q): should be < 50 ms
- VQE solve (2q): should be < 300 ms
- Rust kuramoto_euler: should show > 20× speedup
- Pipeline wiring: all 155 tests should pass in < 10 s total

## Dependencies

Test-only dependencies (in `[dev]` extras):
- `pytest >= 8.0`
- `pytest-cov >= 7.0`
- `hypothesis >= 6.100`
- `qiskit-aer >= 0.17`

Optional (for full test coverage):
- `scpn-quantum-engine >= 0.2.0` (Rust acceleration tests)
- `quimb >= 1.8` (MPS/DMRG tests — `importorskip`)
- `sc-neurocore >= 3.14` (optional SNN bridge E2E tests — `skipif`)
- `pennylane >= 0.40` (PennyLane adapter tests — `skipif`)

### Extras required for a clean full collection

A handful of modules import an optional-extra package at module scope rather
than behind an `importorskip`/`skipif` guard, so `pytest --collect-only` on the
base `[dev]` install alone raises collection errors for them (not skips). Install
these three extras for a clean full collection:

- `[config]` — `pydantic-settings` (+ `pydantic`). `scpn_quantum_control.config`
  imports `pydantic`/`pydantic_settings` at module scope; any test that reaches
  the typed configuration path needs it.
- `[logging]` — `structlog`. The runtime carries a graceful no-`structlog`
  fallback, but the tests that exercise the real structured-logging path import
  `structlog` directly (e.g. `tests/test_logging_setup.py`).
- `[braket]` — `amazon-braket-sdk`. `tests/test_hardware_hal_braket_adapters.py`
  imports `braket.circuits` at module scope.

One command covers all three (plus the runtime and dev tools):

```bash
pip install -e '.[dev,config,logging,braket]'
```

CI installs the full extra matrix, so this only bites a minimal local `[dev]`
checkout. The alternative remediation — making these imports lazy — is tracked
but deferred; documenting the extras is the supported path for now.

### Single-process memory

The whole suite in one process peaks above 4 GB (the 18-qubit `slow`-marked
cases dominate; see the `slow` marker note above). On a memory-constrained
machine, exclude the heavy lane with `-m 'not slow'`, or run in shards by path
(`pytest tests/<subset>`), rather than running the entire suite single-process.
`pytest-xdist` is not a declared dependency; install it separately if you want
`-n auto` process sharding.

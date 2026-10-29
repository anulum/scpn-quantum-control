#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Original Studio browser journeys
set -euo pipefail

browser_coverage_command=(python -m coverage run --rcfile=/dev/null --append --branch --concurrency=greenlet
  --data-file="$RUNNER_TEMP/studio-workspace-browser.coverage"
  --include="*/tools/studio_browser_journey.py,*/tools/studio_local_experiment_browser.py,*/tools/studio_result_browser.py,*/tools/studio_comparison_browser.py,*/tools/studio_workflow_browser.py,*/tools/studio_owned_worker_journey.py,*/tools/studio_resource_browser_journey.py,*/tools/studio_accessibility_browser.py,*/tools/studio_accessibility_checks.py,*/tools/studio_parameter_browser_journey.py,*/tools/studio_workbench_browser_journey.py,*/tools/studio_workspace_browser_journey.py,*/tools/studio_workspace_browser_coverage.py,*/tools/studio_workspace_browser_ui.py")
python -m http.server 4173 --bind 127.0.0.1 --directory "$RUNNER_TEMP/studio-browser-preview" > "$RUNNER_TEMP/studio-preview.log" 2>&1 &
preview_pid=$!
node studio-web/node_modules/vite/bin/vite.js "$RUNNER_TEMP/studio-workbench-source/studio-web" --host 127.0.0.1 --port 4174 --strictPort > "$RUNNER_TEMP/studio-workspace-source.log" 2>&1 &
source_pid=$!
trap 'kill "$preview_pid" "$source_pid" 2>/dev/null || true; wait "$preview_pid" || true; wait "$source_pid" || true' EXIT
for ((attempt = 1; attempt <= 5; attempt++)); do
  if curl --fail --silent http://127.0.0.1:4173/ > /dev/null; then break; fi
  sleep 1
done
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario capability_catalogue \
  --base-url http://127.0.0.1:4173/ --output "$RUNNER_TEMP/studio-catalogue-journey.json"
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario evidence_inspector \
  --base-url http://127.0.0.1:4173/ --output "$RUNNER_TEMP/studio-evidence-journey.json"
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario resource_plan_projection \
  --base-url http://127.0.0.1:4173/ --output "$RUNNER_TEMP/studio-resource-journey.json"
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario owned_kernel_worker \
  --base-url http://127.0.0.1:4173/ --output "$RUNNER_TEMP/studio-owned-worker-journey.json"
for ((attempt = 1; attempt <= 5; attempt++)); do
  if curl --fail --silent http://127.0.0.1:4174/browser-tests/workspace.html > /dev/null; then break; fi
  sleep 1
done
STUDIO_WORKSPACE_SOURCE_URL=http://127.0.0.1:4174/ "${browser_coverage_command[@]}" \
  -m pytest --noconftest -q tools/tests/test_studio_workspace_browser_coverage.py -k 'not archive'
STUDIO_WORKSPACE_SOURCE_URL=http://127.0.0.1:4174/ STUDIO_PREVIEW_DIR="$RUNNER_TEMP/studio-browser-preview" \
  bash tools/run_ci_test_shards.sh studio-browser 2 "$RUNNER_TEMP/studio-browser-shards" \
  "$RUNNER_TEMP/studio-browser-duration-hints.json" "$RUNNER_TEMP/studio-workspace-browser.coverage" \
  "${browser_coverage_command[-1]#--include=}" greenlet \
  tools/tests/test_studio_browser_runtime_refusal.py tools/tests/test_studio_owned_worker_browser_runtime.py tools/tests/test_studio_workbench_browser_journey.py tools/tests/test_studio_parameter_browser_journey.py tools/tests/test_studio_local_experiment_browser_runtime.py tools/tests/test_studio_result_browser_runtime.py tools/tests/test_studio_comparison_browser_runtime.py tools/tests/test_studio_workflow_browser_runtime.py
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario workspace_recovery \
  --base-url http://127.0.0.1:4173/ --workspace-source-url http://127.0.0.1:4174/ \
  --output "$RUNNER_TEMP/studio-workspace-journey.json"
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario workbench_navigation \
  --base-url http://127.0.0.1:4173/ --workspace-source-url http://127.0.0.1:4174/ \
  --output "$RUNNER_TEMP/studio-workbench-journey.json"
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario parameter_graph_editor \
  --base-url http://127.0.0.1:4173/ --workspace-source-url http://127.0.0.1:4174/ \
  --output "$RUNNER_TEMP/studio-parameter-journey.json"
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario result_value_inspector \
  --base-url http://127.0.0.1:4173/ --workspace-source-url http://127.0.0.1:4174/ \
  --output "$RUNNER_TEMP/studio-result-journey.json"
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario immutable_run_comparison \
  --base-url http://127.0.0.1:4173/ --workspace-source-url http://127.0.0.1:4174/ \
  --output "$RUNNER_TEMP/studio-comparison-journey.json"
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario experiment_workflow_runner \
  --base-url http://127.0.0.1:4173/ --workspace-source-url http://127.0.0.1:4174/ \
  --output "$RUNNER_TEMP/studio-workflow-journey.json"
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario local_experiment_journey \
  --base-url http://127.0.0.1:4173/ --workspace-source-url http://127.0.0.1:4174/ \
  --output "$RUNNER_TEMP/studio-experiment-journey.json"
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario workbench_accessibility \
  --base-url http://127.0.0.1:4173/ --workspace-source-url http://127.0.0.1:4174/ \
  --output "$RUNNER_TEMP/studio-accessibility-journey.json"
STUDIO_WORKSPACE_SOURCE_URL=http://127.0.0.1:4174/ STUDIO_PREVIEW_DIR="$RUNNER_TEMP/studio-browser-preview" \
  "${browser_coverage_command[@]}" -m pytest --noconftest -q \
  tools/tests/test_studio_accessibility_browser_runtime.py tools/tests/test_studio_accessibility_checks.py
python -m coverage run --rcfile=/dev/null --branch --concurrency=greenlet \
  --data-file="$RUNNER_TEMP/studio-program-browser.coverage" \
  --include="*/tools/studio_program_authoring_browser.py,*/tools/studio_browser_journey.py" \
  tools/studio_browser_journey.py --scenario program_authoring \
  --base-url http://127.0.0.1:4173/ --output "$RUNNER_TEMP/studio-program-journey.json"
STUDIO_PREVIEW_DIR="$RUNNER_TEMP/studio-browser-preview" \
  python -m coverage run --rcfile=/dev/null --append --branch --concurrency=greenlet \
  --data-file="$RUNNER_TEMP/studio-program-browser.coverage" \
  --include="*/tools/studio_program_authoring_browser.py" \
  -m pytest --noconftest -q tools/tests/test_studio_program_authoring_browser.py
python -m coverage report --rcfile=/dev/null \
  --data-file="$RUNNER_TEMP/studio-program-browser.coverage" \
  --include="*/tools/studio_program_authoring_browser.py" --precision=2 --show-missing --fail-under=100
python -m coverage combine --rcfile=/dev/null --append --keep \
  --data-file="$RUNNER_TEMP/studio-workspace-browser.coverage" "$RUNNER_TEMP/studio-program-browser.coverage"
"${browser_coverage_command[@]}" tools/studio_browser_journey.py --scenario compiler_trace_inspector \
  --base-url http://127.0.0.1:4173/ --output "$RUNNER_TEMP/studio-compiler-journey.json"
STUDIO_PREVIEW_DIR="$RUNNER_TEMP/studio-browser-preview" \
  python -m coverage run --rcfile=/dev/null --branch --concurrency=greenlet \
  --data-file="$RUNNER_TEMP/studio-compiler-browser.coverage" \
  --include="*/tools/studio_compiler_trace_browser.py" \
  -m pytest --noconftest -q tools/tests/test_studio_compiler_trace_browser.py
python -m coverage report --rcfile=/dev/null \
  --data-file="$RUNNER_TEMP/studio-compiler-browser.coverage" --precision=2 --show-missing --fail-under=100
python -m coverage run --rcfile=/dev/null --branch --concurrency=greenlet \
  --data-file="$RUNNER_TEMP/studio-profiles-browser.coverage" \
  --include="*/tools/studio_backend_profiles_browser.py,*/tools/studio_browser_journey.py" \
  tools/studio_browser_journey.py --scenario operator_backend_profiles \
  --base-url http://127.0.0.1:4173/ --output "$RUNNER_TEMP/studio-profiles-journey.json"
STUDIO_PREVIEW_DIR="$RUNNER_TEMP/studio-browser-preview" \
  python -m coverage run --rcfile=/dev/null --append --branch --concurrency=greenlet \
  --data-file="$RUNNER_TEMP/studio-profiles-browser.coverage" \
  --include="*/tools/studio_backend_profiles_browser.py" \
  -m pytest --noconftest -q tools/tests/test_studio_backend_profiles_browser.py
python -m coverage report --rcfile=/dev/null \
  --data-file="$RUNNER_TEMP/studio-profiles-browser.coverage" \
  --include="*/tools/studio_backend_profiles_browser.py" --precision=2 --show-missing --fail-under=100
python -m coverage combine --rcfile=/dev/null --append --keep \
  --data-file="$RUNNER_TEMP/studio-workspace-browser.coverage" "$RUNNER_TEMP/studio-profiles-browser.coverage"
python -m coverage run --rcfile=/dev/null --branch --concurrency=greenlet \
  --data-file="$RUNNER_TEMP/studio-policy-browser.coverage" \
  --include="*/tools/studio_operator_policy_browser.py,*/tools/studio_browser_journey.py" \
  tools/studio_browser_journey.py --scenario operator_policy_decisions \
  --base-url http://127.0.0.1:4173/ --output "$RUNNER_TEMP/studio-policy-journey.json"
STUDIO_PREVIEW_DIR="$RUNNER_TEMP/studio-browser-preview" \
  python -m coverage run --rcfile=/dev/null --append --branch --concurrency=greenlet \
  --data-file="$RUNNER_TEMP/studio-policy-browser.coverage" \
  --include="*/tools/studio_operator_policy_browser.py" \
  -m pytest --noconftest -q tools/tests/test_studio_operator_policy_browser.py
python -m coverage report --rcfile=/dev/null \
  --data-file="$RUNNER_TEMP/studio-policy-browser.coverage" \
  --include="*/tools/studio_operator_policy_browser.py" --precision=2 --show-missing --fail-under=100
python -m coverage combine --rcfile=/dev/null --append --keep \
  --data-file="$RUNNER_TEMP/studio-workspace-browser.coverage" "$RUNNER_TEMP/studio-policy-browser.coverage"
python -m coverage run --rcfile=/dev/null --branch --concurrency=greenlet \
  --data-file="$RUNNER_TEMP/studio-dossier-browser.coverage" \
  --include="*/tools/studio_operator_dossier_browser.py,*/tools/studio_browser_journey.py" \
  tools/studio_browser_journey.py --scenario operator_review_dossiers \
  --base-url http://127.0.0.1:4173/ --output "$RUNNER_TEMP/studio-dossier-journey.json"
STUDIO_PREVIEW_DIR="$RUNNER_TEMP/studio-browser-preview" \
  python -m coverage run --rcfile=/dev/null --append --branch --concurrency=greenlet \
  --data-file="$RUNNER_TEMP/studio-dossier-browser.coverage" \
  --include="*/tools/studio_operator_dossier_browser.py" \
  -m pytest --noconftest -q tools/tests/test_studio_operator_dossier_browser.py
python -m coverage report --rcfile=/dev/null \
  --data-file="$RUNNER_TEMP/studio-dossier-browser.coverage" \
  --include="*/tools/studio_operator_dossier_browser.py" --precision=2 --show-missing --fail-under=100
python -m coverage combine --rcfile=/dev/null --append --keep \
  --data-file="$RUNNER_TEMP/studio-workspace-browser.coverage" "$RUNNER_TEMP/studio-dossier-browser.coverage"

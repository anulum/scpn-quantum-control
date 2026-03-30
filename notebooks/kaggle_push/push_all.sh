#!/usr/bin/env bash
# Push remaining SCPN notebooks to Kaggle — run 2 at a time, wait 5 min between
set -euo pipefail

PUSH_DIR="$(cd "$(dirname "$0")" && pwd)"
DELAY=300  # 5 minutes between pushes

NOTEBOOKS=(
  kaggle_firefly_sync "SCPN QC Firefly Kuramoto"
  kaggle_cardiac_sa_node "SCPN QC Cardiac SA Node"
  kaggle_calcium_waves "SCPN QC Calcium Waves"
  kaggle_cancer_hypersync "SCPN QC Cancer Hypersync"
  kaggle_cilia_metachronal "SCPN QC Cilia Metachronal"
  kaggle_circadian_scn "SCPN QC Circadian SCN"
  kaggle_cochlear_mechanics "SCPN QC Cochlear Mechanics"
  kaggle_csf_metals_chain "SCPN QC CSF Metals Chain"
  kaggle_dna_breathing "SCPN QC DNA Breathing"
  kaggle_entropy_production "SCPN QC Entropy Production"
  kaggle_enzyme_extended_validation "SCPN QC Enzyme Extended Validation"
  kaggle_glycolytic_nfkb "SCPN QC Glycolytic NFkB"
  kaggle_huygens_metronomes "SCPN QC Huygens Metronomes"
  kaggle_jax_gpu_validation "SCPN QC JAX GPU Validation"
  kaggle_josephson_qubits "SCPN QC Josephson Qubits"
  kaggle_laser_modelocking "SCPN QC Laser Modelocking"
  kaggle_magic_berry_qsl "SCPN QC Magic Berry QSL"
  kaggle_morphogenesis_clock "SCPN QC Morphogenesis Clock"
  kaggle_neural_oscillations "SCPN QC Neural Oscillations"
  kaggle_orbital_resonances "SCPN QC Orbital Resonances"
  kaggle_anaesthesia_consciousness "SCPN QC Anaesthesia Consciousness"
  kaggle_bz_reaction "SCPN QC BZ Reaction"
  kaggle_photosynthesis_fmo "SCPN QC Photosynthesis FMO"
  kaggle_piezo_geometry_bio "SCPN QC Piezo Geometry Bio"
  kaggle_power_grid_sync "SCPN QC Power Grid Sync"
  kaggle_protein_folding_kuramoto "SCPN QC Protein Folding Kuramoto"
  kaggle_reservoir_computing "SCPN QC Reservoir Computing"
  kaggle_schumann_theta "SCPN QC Schumann Theta"
  kaggle_sleep_spindles "SCPN QC Sleep Spindles"
  kaggle_universal_correlations "SCPN QC Universal Correlations"
  kaggle_water_coupling "SCPN QC Water Coupling"
  kaggle_p4_decoherence_cascade_plasticity "SCPN QC P4 Decoherence Cascade"
  kaggle_p4_ephaptic_mechanotransduction "SCPN QC P4 Ephaptic Mechano"
  kaggle_p4_glial_slow_control "SCPN QC P4 Glial Slow Control"
  kaggle_p4_lunar_temperature_oscillation_death "SCPN QC P4 Lunar Temperature"
  kaggle_p4_organ_harmonic_chambers "SCPN QC P4 Organ Harmonic"
  kaggle_p4_stochastic_resonance_adler "SCPN QC P4 Stochastic Resonance"
  kaggle_p4_synchronopathies "SCPN QC P4 Synchronopathies"
  kaggle_p4_topological_metastability_info "SCPN QC P4 Topological Metastability"
  kaggle_paper4_griffiths_metastability "SCPN QC Paper4 Griffiths"
  kaggle_upde_tuning "SCPN QC UPDE Tuning"
)

count=0
for ((i=0; i<${#NOTEBOOKS[@]}; i+=2)); do
  file="${NOTEBOOKS[$i]}"
  title="${NOTEBOOKS[$i+1]}"
  slug=$(echo "$title" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')

  tmpdir=$(mktemp -d)
  cp "$PUSH_DIR/${file}.py" "$tmpdir/${file}.py"
  cat > "$tmpdir/kernel-metadata.json" << METADATA
{
  "id": "anulum/${slug}",
  "title": "${title}",
  "code_file": "${file}.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": false,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
METADATA

  echo "[$(date -u +%H:%M:%S)] Pushing ${file} as ${slug}..."
  if kaggle kernels push -p "$tmpdir" 2>&1; then
    echo "  OK"
    count=$((count + 1))
  else
    echo "  FAILED — stopping (rate limit?)"
    rm -rf "$tmpdir"
    break
  fi
  rm -rf "$tmpdir"

  # Every 2 pushes, wait
  if ((count % 2 == 0)); then
    echo "  Waiting ${DELAY}s for rate limit..."
    sleep "$DELAY"
  fi
done

echo "Pushed ${count} notebooks total."

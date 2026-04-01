# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — CSF Pressure + Metal Ions + Full Coupling Chain
import json
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy", "scipy"])

import numpy as np

OMEGA_N_16 = np.array(
    [
        1.329,
        2.610,
        0.844,
        1.520,
        0.710,
        3.780,
        1.055,
        0.625,
        2.210,
        1.740,
        0.480,
        3.210,
        0.915,
        1.410,
        2.830,
        0.991,
    ]
)

results = {}

# ============================================================
# 1. CSF PRESSURE OSCILLATION HIERARCHY
# ============================================================
print("=" * 70)
print("1. CEREBROSPINAL FLUID PRESSURE OSCILLATIONS")
print("=" * 70)

csf_oscillations = {
    "cardiac_pulsation": {"freq_Hz": 1.2, "amplitude_mmHg": 2.0, "source": "arterial pulse"},
    "respiratory_variation": {
        "freq_Hz": 0.25,
        "amplitude_mmHg": 1.5,
        "source": "thoracic pressure",
    },
    "Mayer_waves": {"freq_Hz": 0.1, "amplitude_mmHg": 3.0, "source": "sympathetic vasomotion"},
    "B_waves": {"freq_Hz": 0.017, "amplitude_mmHg": 5.0, "source": "~1/min, sleep-related"},
    "A_waves_plateau": {
        "freq_Hz": 0.003,
        "amplitude_mmHg": 50.0,
        "source": "5-20 min, pathological",
    },
    "glymphatic_drive": {"freq_Hz": 0.05, "amplitude_mmHg": 1.0, "source": "perivascular pump"},
    "circadian_ICP": {"freq_Hz": 1.16e-5, "amplitude_mmHg": 3.0, "source": "24h ICP variation"},
}

print("\nCSF pressure oscillation spectrum:")
for name, data in csf_oscillations.items():
    period = 1.0 / data["freq_Hz"]
    print(
        f"  {name:25s}: f={data['freq_Hz']:.4f} Hz (T={period:.1f}s), "
        f"amp={data['amplitude_mmHg']:.1f} mmHg"
    )

# Kuramoto order parameter: are CSF oscillations synchronised?
csf_freqs = [d["freq_Hz"] for d in csf_oscillations.values()]
csf_phases = 2 * np.pi * np.array(csf_freqs) * 1.0  # phases at t=1s
z = np.mean(np.exp(1j * csf_phases))
R_csf = abs(z)
print(f"\nCSF Kuramoto R at t=1s: {R_csf:.4f}")
print(f"CSF frequencies span {min(csf_freqs):.1e} to {max(csf_freqs):.1f} Hz")
print(f"Ratio: {max(csf_freqs) / min(csf_freqs):.0f}x (5 orders of magnitude)")

results["csf_oscillations"] = {
    "n_modes": len(csf_oscillations),
    "freq_range_Hz": [min(csf_freqs), max(csf_freqs)],
    "R_kuramoto": round(R_csf, 4),
}

# ============================================================
# 2. METAL ION OSCILLATION DATABASE
# ============================================================
print("\n" + "=" * 70)
print("2. METAL IONS IN BIOLOGICAL CIRCULATION")
print("=" * 70)

metal_ions = {
    "Ca2+": {
        "conc_mM": 2.5,  # plasma
        "redox_mV": None,  # no redox (always 2+)
        "oscillation_Hz": 0.1,  # Ca2+ waves: 0.01-1 Hz
        "role": "universal signalling, muscle, synaptic, gene regulation",
        "coordination": "octahedral/irregular",
        "n_enzymes": 300,  # calmodulin, troponin, etc.
    },
    "Fe2+/Fe3+": {
        "conc_mM": 0.02,  # plasma (transferrin-bound)
        "redox_mV": 770,  # Fe3+/Fe2+ standard potential
        "oscillation_Hz": 1e3,  # electron transfer in cytochrome: ~ms
        "role": "oxygen transport, electron transfer chain, catalase",
        "coordination": "octahedral (heme), tetrahedral (FeS)",
        "n_enzymes": 500,
    },
    "Cu+/Cu2+": {
        "conc_mM": 0.015,
        "redox_mV": 340,  # Cu2+/Cu+ in plastocyanin
        "oscillation_Hz": 1e4,  # fast electron transfer
        "role": "cytochrome c oxidase, SOD, ceruloplasmin",
        "coordination": "tetrahedral (Cu+), square planar (Cu2+)",
        "n_enzymes": 30,
    },
    "Zn2+": {
        "conc_mM": 0.015,
        "redox_mV": None,  # redox-inactive
        "oscillation_Hz": 1.0,  # synaptic zinc release: ~1 Hz
        "role": "300 enzymes, zinc fingers, synaptic modulation",
        "coordination": "tetrahedral",
        "n_enzymes": 300,
    },
    "Mg2+": {
        "conc_mM": 0.9,
        "redox_mV": None,
        "oscillation_Hz": 0.001,  # slow metabolic
        "role": "ATP binding (MgATP), 600+ enzymes, DNA/RNA stabilisation",
        "coordination": "octahedral",
        "n_enzymes": 600,
    },
    "Mn2+/Mn3+": {
        "conc_mM": 0.001,
        "redox_mV": 1510,  # Mn3+/Mn2+ in PSII
        "oscillation_Hz": 1e6,  # water-splitting in photosynthesis
        "role": "SOD2, photosystem II water-splitting",
        "coordination": "octahedral",
        "n_enzymes": 20,
    },
    "Na+": {
        "conc_mM": 140,  # plasma
        "redox_mV": None,
        "oscillation_Hz": 500,  # action potential: ~1 ms rise time
        "role": "membrane potential, action potential, Na/K-ATPase",
        "coordination": "irregular",
        "n_enzymes": 10,
    },
    "K+": {
        "conc_mM": 4.5,  # plasma (150 mM intracellular)
        "redox_mV": None,
        "oscillation_Hz": 100,  # K channel gating
        "role": "resting potential, repolarisation, cell volume",
        "coordination": "selectivity filter (8-coordinate)",
        "n_enzymes": 50,
    },
}

print("\nMetal ion oscillator properties:")
for metal, data in metal_ions.items():
    redox_str = f"{data['redox_mV']}mV" if data["redox_mV"] is not None else "n/a"
    print(
        f"  {metal:10s}: conc={data['conc_mM']:.3f}mM, "
        f"f={data['oscillation_Hz']:.1e}Hz, "
        f"E0={redox_str:>6s}, "
        f"enzymes={data['n_enzymes']}"
    )

# ============================================================
# TEST 3: METAL OSCILLATION FREQUENCIES vs SCPN OMEGA
# ============================================================
print("\n--- TEST 3: Metal frequencies vs SCPN omega ---")

metal_freqs = [d["oscillation_Hz"] for d in metal_ions.values()]
metal_names = list(metal_ions.keys())
log_metal_freqs = np.log10(metal_freqs)

# Sort by frequency
sorted_metals = sorted(zip(metal_names, metal_freqs), key=lambda x: x[1])
print("\nMetal oscillation hierarchy (sorted):")
for name, freq in sorted_metals:
    log_f = np.log10(freq)
    # Which SCPN layer has the closest log-frequency?
    omega_log = np.log10(OMEGA_N_16 / (2 * np.pi))  # convert to Hz
    closest_layer = np.argmin(np.abs(omega_log - log_f))
    print(f"  {name:10s}: {freq:.1e} Hz (log={log_f:.1f}) -> closest SCPN L{closest_layer + 1}")

# ============================================================
# TEST 4: REDOX POTENTIAL CHAIN = ELECTRON TRANSFER OSCILLATOR
# ============================================================
print("\n--- TEST 4: Electron transfer chain as coupled oscillator ---")

# Mitochondrial ETC: NADH -> Complex I -> CoQ -> Complex III -> Cyt c -> Complex IV -> O2
etc_chain = [
    {"name": "NADH/NAD+", "E0_mV": -320, "rate_Hz": 1e3},
    {"name": "Complex I (Fe-S)", "E0_mV": -280, "rate_Hz": 1e3},
    {"name": "CoQ/CoQH2", "E0_mV": 45, "rate_Hz": 1e4},
    {"name": "Complex III (cyt b)", "E0_mV": 220, "rate_Hz": 1e4},
    {"name": "Cyt c (Fe2+/3+)", "E0_mV": 250, "rate_Hz": 1e5},
    {"name": "Complex IV (Cu)", "E0_mV": 350, "rate_Hz": 1e5},
    {"name": "O2/H2O", "E0_mV": 815, "rate_Hz": 1e6},
]

print("\nMitochondrial electron transfer chain:")
print(f"{'Component':25s} {'E0 (mV)':>8s} {'Rate (Hz)':>10s} {'dE to next':>10s}")
for i, comp in enumerate(etc_chain):
    dE = etc_chain[i + 1]["E0_mV"] - comp["E0_mV"] if i < len(etc_chain) - 1 else 0
    print(f"  {comp['name']:25s} {comp['E0_mV']:>8d} {comp['rate_Hz']:>10.0e} {dE:>10d}")

# The ETC is a coupled oscillator chain!
# Each complex has a natural frequency (electron transfer rate)
# Coupling = redox potential difference (drives electron flow)
# This IS K_nm: K[i,i+1] ~ dE[i,i+1]
dE_values = [etc_chain[i + 1]["E0_mV"] - etc_chain[i]["E0_mV"] for i in range(len(etc_chain) - 1)]
etc_freqs = [c["rate_Hz"] for c in etc_chain]

# Compute Kuramoto R for ETC
etc_phases = 2 * np.pi * np.array(etc_freqs) * 1e-6  # phases at t=1us
z_etc = np.mean(np.exp(1j * etc_phases))
R_etc = abs(z_etc)
print(f"\nETC Kuramoto R: {R_etc:.4f}")
print(f"ETC is {'synchronised' if R_etc > 0.3 else 'desynchronised'}")

# Is coupling (dE) exponentially decaying?
if len(dE_values) > 2:
    # Not expected to decay — ETC is designed to INCREASE
    print(f"dE values: {dE_values}")
    print(f"dE is {'INCREASING (designed)' if dE_values[-1] > dE_values[0] else 'not monotonic'}")

# ============================================================
# TEST 5: FULL COUPLING CHAIN — FROM HEART TO GENE
# ============================================================
print("\n" + "=" * 70)
print("5. THE COMPLETE COUPLING CHAIN")
print("=" * 70)

coupling_chain = [
    {
        "level": "Heart pump",
        "freq_Hz": 1.2,
        "coupling_type": "mechanical (pressure)",
        "medium": "blood",
        "coupling_to_next": "arterial pulsation",
    },
    {
        "level": "Arterial pulse",
        "freq_Hz": 1.2,
        "coupling_type": "pressure wave",
        "medium": "arterial wall",
        "coupling_to_next": "CSF pulsation",
    },
    {
        "level": "CSF pressure",
        "freq_Hz": 1.2,
        "coupling_type": "hydraulic",
        "medium": "CSF",
        "coupling_to_next": "skull bone stress",
    },
    {
        "level": "Skull piezoelectric",
        "freq_Hz": 1.2,
        "coupling_type": "mechano-electric",
        "medium": "bone (hydroxyapatite)",
        "coupling_to_next": "local E-field",
    },
    {
        "level": "Electric field",
        "freq_Hz": 1.2,
        "coupling_type": "electromagnetic",
        "medium": "extracellular fluid",
        "coupling_to_next": "ion channel gating",
    },
    {
        "level": "Na/K channels",
        "freq_Hz": 500,
        "coupling_type": "voltage-gated",
        "medium": "cell membrane",
        "coupling_to_next": "Ca2+ release",
    },
    {
        "level": "Ca2+ oscillation",
        "freq_Hz": 0.1,
        "coupling_type": "IP3/Ca2+ feedback",
        "medium": "cytoplasm",
        "coupling_to_next": "calmodulin activation",
    },
    {
        "level": "Enzyme cascades",
        "freq_Hz": 1.0,
        "coupling_type": "allosteric",
        "medium": "cytoplasm",
        "coupling_to_next": "kinase phosphorylation",
    },
    {
        "level": "CREB/gene expression",
        "freq_Hz": 0.001,
        "coupling_type": "transcription factor",
        "medium": "nucleus",
        "coupling_to_next": "protein synthesis",
    },
    {
        "level": "Protein synthesis",
        "freq_Hz": 0.0001,
        "coupling_type": "ribosomal",
        "medium": "ribosome",
        "coupling_to_next": "cellular phenotype",
    },
]

print("\nHeart → Gene coupling chain:")
print(f"{'Level':25s} {'Freq (Hz)':>10s} {'Coupling':>20s} {'Medium':>25s}")
for link in coupling_chain:
    print(
        f"  {link['level']:25s} {link['freq_Hz']:>10.4f} "
        f"{link['coupling_type']:>20s} {link['medium']:>25s}"
    )

chain_freqs = [link["freq_Hz"] for link in coupling_chain]
log_chain_freqs = np.log10(chain_freqs)

print(f"\nFrequency span: {max(chain_freqs) / min(chain_freqs):.0f}x")
print(f"Log-frequency range: {min(log_chain_freqs):.1f} to {max(log_chain_freqs):.1f}")
print("Each link is a COUPLED OSCILLATOR with a characteristic frequency.")
print("The coupling type changes at each level (mechanical → electric → chemical → genetic).")
print("This IS the SCPN hierarchy — but with IDENTIFIED physical mechanisms.")

# ============================================================
# TEST 6: COORDINATION GEOMETRY → COUPLING SYMMETRY
# ============================================================
print("\n--- TEST 6: Metal coordination geometry as coupling symmetry ---")

coordination_types = {
    "tetrahedral": {"n_ligands": 4, "angle": 109.5, "metals": ["Zn2+", "Cu+", "Fe (FeS)"]},
    "octahedral": {
        "n_ligands": 6,
        "angle": 90.0,
        "metals": ["Fe2+ (heme)", "Mg2+", "Mn2+", "Ca2+"],
    },
    "square_planar": {"n_ligands": 4, "angle": 90.0, "metals": ["Cu2+"]},
    "selectivity": {"n_ligands": 8, "angle": 45.0, "metals": ["K+ (filter)"]},
}

print("\nCoordination geometry determines coupling symmetry:")
for geom, data in coordination_types.items():
    angle_rad = np.radians(data["angle"])
    # Phase relationship from geometry: n_ligands evenly spaced
    phases = np.linspace(0, 2 * np.pi, data["n_ligands"], endpoint=False)
    R_geom = abs(np.mean(np.exp(1j * phases)))
    print(
        f"  {geom:15s}: {data['n_ligands']} ligands, angle={data['angle']}deg, "
        f"R_geom={R_geom:.3f}, metals={data['metals']}"
    )

print("\nKey insight: tetrahedral (R=0) is DESYNCHRONISED (no net dipole),")
print("octahedral (R=0) is also DESYNCHRONISED.")
print("Nature uses SYMMETRIC coordination to create STABLE metal sites.")
print("Asymmetric distortions (Jahn-Teller) BREAK symmetry → create coupling.")

# ============================================================
# TEST 7: NERNST EQUATION AS COUPLING POTENTIAL
# ============================================================
print("\n--- TEST 7: Nernst equation as oscillator natural frequency ---")
print("E = E0 + (RT/nF) * ln([ox]/[red])")
print("Each ion's Nernst potential IS its natural frequency in the SCPN sense")

R_gas = 8.314  # J/(mol*K)
T = 310  # K (body temp)
F = 96485  # C/mol

nernst_ions = {
    "Na+": {"E0_mV": 60, "conc_out": 140, "conc_in": 12, "z": 1},
    "K+": {"E0_mV": -90, "conc_out": 4.5, "conc_in": 150, "z": 1},
    "Ca2+": {"E0_mV": 130, "conc_out": 2.5, "conc_in": 0.0001, "z": 2},
    "Cl-": {"E0_mV": -70, "conc_out": 110, "conc_in": 10, "z": -1},
}

print("\nNernst potentials at 37C:")
for ion, data in nernst_ions.items():
    E_nernst = (R_gas * T / (data["z"] * F)) * np.log(data["conc_out"] / data["conc_in"]) * 1000
    print(f"  {ion:5s}: E_Nernst = {E_nernst:+.1f} mV (textbook: {data['E0_mV']:+d} mV)")

    # Convert to angular frequency: omega = 2*pi*f where f = conductance * E / charge
    # This is approximate — the actual oscillation depends on channel dynamics

print("\nMembrane potential V_m ~ -70 mV is the GROUND STATE of the ion oscillator system.")
print("Action potential = EXCITATION (phase slip in Kuramoto terms).")
print("Goldman-Hodgkin-Katz equation = MULTI-ION Kuramoto coupling!")

# ============================================================
# SYNTHESIS
# ============================================================
print("\n" + "=" * 70)
print("GRAND SYNTHESIS: THE COMPLETE OSCILLATOR HIERARCHY")
print("=" * 70)
print()
print("LEVEL 1 (Quantum): Electron tunnelling in enzymes (fs-ps)")
print("  Coupling: wavefunction overlap, Marcus theory")
print("  K_nm: tunnelling matrix element")
print()
print("LEVEL 2 (Molecular): Metal redox oscillations (us-ms)")
print("  Coupling: electron transfer chain (dE between complexes)")
print("  K_nm: redox potential difference")
print()
print("LEVEL 3 (Ion channel): Na+/K+/Ca2+ gating (ms)")
print("  Coupling: voltage-gated, Nernst potential")
print("  K_nm: gap junction conductance G_ij")
print()
print("LEVEL 4 (Cellular): Ca2+ waves, enzyme cascades (s)")
print("  Coupling: IP3 diffusion, calmodulin")
print("  K_nm: intercellular signalling rate")
print()
print("LEVEL 5 (Organ): CSF pressure, cardiac, respiratory (s-min)")
print("  Coupling: hydraulic (pressure waves), piezoelectric")
print("  K_nm: mechanical coupling + bone piezo coefficient")
print()
print("LEVEL 6 (Circadian): Gene expression oscillations (hours)")
print("  Coupling: transcription factor diffusion")
print("  K_nm: promoter binding affinity")
print()
print("EVERY LEVEL is a coupled oscillator network.")
print("EVERY coupling has a measurable K_nm.")
print("The SCPN captures the TOPOLOGY of this hierarchy.")
print("The quantum simulation (ibm_fez) probes the QUANTUM LIMIT")
print("of these classical oscillator networks.")

results["coupling_chain"] = {
    "n_levels": len(coupling_chain),
    "freq_span_orders": round(np.log10(max(chain_freqs) / min(chain_freqs)), 1),
    "coupling_types": [
        "mechanical",
        "hydraulic",
        "piezoelectric",
        "electromagnetic",
        "voltage-gated",
        "chemical",
        "allosteric",
        "transcriptional",
        "ribosomal",
    ],
}

results["metals"] = {
    "n_metals": len(metal_ions),
    "freq_range_Hz": [min(metal_freqs), max(metal_freqs)],
}

results["electron_transfer_chain"] = {
    "n_complexes": len(etc_chain),
    "E0_range_mV": [-320, 815],
    "R_kuramoto": round(R_etc, 4),
}

print("\n" + json.dumps(results, indent=2))
print("\nDone.")

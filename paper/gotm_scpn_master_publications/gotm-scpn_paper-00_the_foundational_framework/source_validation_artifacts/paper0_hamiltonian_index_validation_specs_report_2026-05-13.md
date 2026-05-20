# Paper 0 Hamiltonian Index Specs

- Source span: P0R06878 - P0R06915
- Source records consumed: 38
- Coverage match: True
- Operator count: 9
- Hardware status: operator_index_no_execution
- Claim boundary: source-bounded Hamiltonian/operator index; not empirical evidence

## Specs

### appendix_c.hamiltonian_index.appendix_boundary
- Protocol: paper0.appendix_c.hamiltonian_index.boundary
- Statement: Appendix C is introduced as a mathematical reference for later papers.
- Operators: none
- Source equations: P0R06880:appendix_location, P0R06881:appendix_title
- Null controls: 3

### appendix_c.hamiltonian_index.master_lagrangian
- Protocol: paper0.appendix_c.hamiltonian_index.master_lagrangian
- Statement: The master Lagrangian is indexed as the fundamental action principle.
- Operators: L_Anulum
- Source equations: P0R06885:master_lagrangian
- Null controls: 3

### appendix_c.hamiltonian_index.microtubule_layer1
- Protocol: paper0.appendix_c.hamiltonian_index.microtubule_layer1
- Statement: Layer 1 microscopic operators index microtubule, transduction, and isotopic-spin terms.
- Operators: H_MT, H_PQT, H_iso
- Source equations: P0R06889:microtubule_frohlich_hamiltonian, P0R06892:piezo_quantum_transduction_hamiltonian, P0R06895:isotopic_spin_interaction
- Null controls: 3

### appendix_c.hamiltonian_index.neuroimmune_mesoscopic
- Protocol: paper0.appendix_c.hamiltonian_index.neuroimmune_mesoscopic
- Statement: Mesoscopic operators index neuro-immune and synaptic Hamiltonian entries.
- Operators: H_NI, H_syn
- Source equations: P0R06899:neuroimmune_hamiltonian, P0R06902:synaptic_location
- Null controls: 3

### appendix_c.hamiltonian_index.radical_pair_macro
- Protocol: paper0.appendix_c.hamiltonian_index.radical_pair_macro
- Statement: Macroscopic Layer 6-8 entries index radical-pair and stochastic-resonance operators.
- Operators: H_RP, H_QSR
- Source equations: P0R06905:radical_pair_hamiltonian
- Null controls: 3

### appendix_c.hamiltonian_index.informational_operators
- Protocol: paper0.appendix_c.hamiltonian_index.informational_operators
- Statement: Informational entries index the phase-curvature tensor and semiotic operator.
- Operators: R_Psi, O_sem
- Source equations: P0R06910:phase_curvature_tensor, P0R06912:semiotic_operator
- Null controls: 3

### appendix_c.hamiltonian_index.structural_separators
- Protocol: paper0.appendix_c.hamiltonian_index.structural_separators
- Statement: Blank structural records after Appendix C are consumed to preserve source contiguity.
- Operators: none
- Source equations: P0R06914:blank_structural_separator, P0R06915:blank_structural_separator
- Null controls: 3

# Contribution

## QuantumGraph.py - Changes Summary

### Imports and Library Adjustments

**Qiskit Version Comptability**

- Previous:
  `from qiskit import QuantumCircuit, execute, Aer`
- Corrected:
  `from qiskit import transpile, QuantumCircuit`
  `from qiskit_aer import AerSimulator`
> Explanation: Updated to match Qiskit 2.0.2, replacing deprecated imports with the current supported simulator (AerSimulator).

### Decomposition Libraries
- Previous
  `from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer`
  `from qiskit.quantum_info.synthesis.two_qubit_decompose import two_qubit_cnot_decompose`
- Corrected
  `from qiskit.synthesis import OneQubitEulerDecomposer`
  `from qiskit.synthesis import TwoQubitBasisDecomposer`
  `from qiskit.circuit.library import CXGate`
> Explanation: Adjusted imports to use TwoQubitBasisDecomposer and CXGate for compatibility.

### Tomography Changes
- Previous:
  `from pairwise_tomography.pairwise_state_tomography_circuits import pairwise_state_tomography_circuits`
  `from pairwise_tomography.pairwise_fitter import PairwiseStateTomographyFitter`
- Corrected:
  `from qiskit_experiments.library.tomography import StateTomography`
> Explanation: qiskit-ignis dependencies replaced with qiskit-experiments library to align with newer Qiskit versions.

### Backend Handling
- Previous:
  `self.backend = Aer.get_backend('qasm_simulator')`
- Corrected:
  `self.backend = AerSimulator()`
> Explanation: Updated backend initialization to reflect the new method in qiskit-aer.

### Execution Method Updates

- Previous:
  `Used execute() method.`
- Corrected:
  `Directly used backend’s .run() method (recommended practice for newer Qiskit versions).`

### State Tomography

- Previous:
  `Custom pairwise tomography implementation.`
- Corrected:
  `Utilized built-in StateTomography from qiskit-experiments.`

### Gate Implementation Adjustments

- Previous: Used deprecated u3 gate:
  `self.qc.u3(the, phi, lam, qubit)`
- Corrected: Replaced with universal gate (u):
  `self.qc.u(the, phi, lam, qubit)`

### Two-Qubit Decomposition
- Previous: 
  `two_qubit_cnot_decompose(U)`
- Corrected:
  `TwoQubitBasisDecomposer(CXGate())`
> Explanation: Updated method to use the new decomposer explicitly specifying CXGate.

### Result Handling

- Added: Custom FakeResult class used as a workaround for qiskit-aer issue.

### Miscellaneous Changes

- Added extensive debug print statements to monitor the internal state of computations.

### Overall Summary

> The changes enhance compatibility with Qiskit 2.0.2 and improve stability, clarity, and maintainability of the quantum circuit implementations, especially in tomography and gate decomposition areas.

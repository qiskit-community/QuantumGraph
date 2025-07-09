from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate

from qiskit.quantum_info import partial_trace, DensityMatrix

from qiskit_ibm_runtime import  EstimatorV2 as Estimator

from qiskit_experiments.library.tomography import StateTomography
from qiskit.result import Result
from qiskit_experiments.library.tomography import *

from quantumgraph.ExpectationValue import ExpectationValue

from numpy import pi, cos, sin, sqrt, exp, arccos, arctan2, conj, array, kron, dot, outer, nan, isnan
import numpy as np
from numpy.random import normal

from scipy import linalg as la
from scipy.linalg import fractional_matrix_power as pwr

from random import random, choice

# define the Pauli matrices in a dictionary
matrices = {}
matrices['I'] = np.identity(2,dtype='complex')
matrices['X'] = np.array([[0,1+0j],[1+0j,0]])
matrices['Y'] = np.array([[0,-1j],[1j,0]])
matrices['Z'] = np.array([[1+0j,0],[0,-1+0j]])
for pauli1 in ['I','X','Y','Z']:
    for pauli2 in ['I','X','Y','Z']:
        matrices[pauli1+pauli2] = kron(matrices[pauli2],matrices[pauli1])

class QuantumGraph ():
    
    def __init__ (self,num_qubits,coupling_map=[],backend=None):
        '''
        Args:
            num_qubits: The number of qubits, and hence the number of nodes in the graph.
            coupling_map: A list of pairs of qubits, corresponding to edges in the graph.
                If none is given, a fully connected graph is used.
            backend: The backend on which the graph will be run as a Qiskit backend object.
            If none is given, a local simulator is used.
        '''
        
        self.num_qubits = num_qubits
        
        # the coupling map consists of pairs [j,k] with the convention j<k
        self.coupling_map = []
        for j in range(self.num_qubits-1):
            for k in range(j+1,self.num_qubits):
                if ([j,k] in coupling_map) or ([j,k] in coupling_map) or (not coupling_map):
                    self.coupling_map.append([j,k])
              
        # construct all two qubit paulis
        self.observables = self._get_observables()

        # set backend
        if backend==None:
            self.backend = AerSimulator()
        else:
            self.backend = backend
        
        # create the quantum circuit, and initialize the tomography
        self.qc = QuantumCircuit(self.num_qubits)
        self.update_tomography()
        
    def _get_observables(self):
        observables = []
        for qubit0 in range(self.num_qubits):
            for pauli in ['X','Y','Z']:
                full_pauli = ['I']*self.num_qubits
                full_pauli[qubit0] = pauli
                obs = ''.join(full_pauli)
                observables.append(obs)
            for qubit1 in range(qubit0+1,self.num_qubits):
                full_pauli = ['I']*self.num_qubits
                for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
                    full_pauli[qubit0] = pauli[0]
                    full_pauli[qubit1] = pauli[1]
                    obs = ''.join(full_pauli)
                    observables.append(obs)
        return observables

    def update_tomography(self, shots=8192):
        '''
        Updates the `tomography` attribute of the `QuantumGraph` object, which contain the two qubit tomography.
        The update is persomed by running the current circuit on the backend.
        
        Args:
            shots: Number of shots to use.
        '''   

        if type(self.backend)==ExpectationValue:
            self.backend = ExpectationValue(self.backend.n,
                                            k=self.backend.k,
                                            pairs=self.backend.pairs)
            self.backend.apply_circuit(self.qc)
        else:
            estimator = Estimator(mode=self.backend)
            estimator.options.default_shots = shots
            job = estimator.run([(transpile(self.qc,self.backend), self.observables)])
            result = job.result()[0]
            self.pauli_decomp = {}
            for obs, val in zip (self.observables, result.data.evs):
                self.pauli_decomp[obs[::-1]] = val
    
    def get_bloch(self,qubit):
        '''
        Returns the X, Y and Z expectation values for the given qubit.
        '''
        if type(self.backend) == ExpectationValue:
            pauli_decomp = self.backend.pauli_decomp
        else:
            pauli_decomp = self.pauli_decomp
        expect = {}
        full_pauli = ['I'] * self.num_qubits
        for pauli in ['X', 'Y', 'Z']:
            full_pauli[qubit] = pauli
            expect[pauli] = pauli_decomp[''.join(full_pauli)]
        return expect
    
    def get_relationship(self, qubit0, qubit1):
        '''
        Returns the two qubit pauli expectation values for a given pair of qubits.
        '''
        if type(self.backend) == ExpectationValue:
            pauli_decomp = self.backend.pauli_decomp
        else:
            pauli_decomp = self.pauli_decomp
        relationship = {}
        full_pauli = ['I']*self.num_qubits
        for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
            full_pauli[qubit0] = pauli[0]
            full_pauli[qubit1] = pauli[1]
            full_pauli_string = ''.join(full_pauli)
            if full_pauli_string in pauli_decomp:
                val = pauli_decomp[full_pauli_string]
            else:
                # Use product of marginals if not present
                bloch0 = self.get_bloch(qubit0)
                bloch1 = self.get_bloch(qubit1)
                val = bloch0[pauli[0]] * bloch1[pauli[1]]
            relationship[pauli] = val
            # Reset for next loop
            full_pauli[qubit0] = 'I'
            full_pauli[qubit1] = 'I'
        return relationship


    def set_bloch(self,target_expect,qubit,fraction=1,update=True):
        '''
        Rotates the given qubit towards the given target state.
        
        Args:
            target_state: Expectation values of the target state.
            qubit: Qubit on which the operation is applied
            fraction: fraction of the rotation toward the target state to apply.
            update: whether to update the tomography after the rotation is added to the circuit.
        '''
                    
        def normalize(expect):
            '''
            Returns the given expectation values after normalization.
            '''
            R = sqrt( expect['X']**2 + expect['Y']**2 + expect['Z']**2 )
            return {pauli:expect[pauli]/R for pauli in expect}
        
        def get_basis(expect):
            '''
            Get the eigenbasis of the density matrix for a the given expectation values.
            '''
            normalized_expect = normalize(expect)
            
            theta = arccos(normalized_expect['Z'])
            phi = arctan2(normalized_expect['Y'],normalized_expect['X'])
            
            state0 = [cos(theta/2),exp(1j*phi)*sin(theta/2)]
            state1 = [conj(state0[1]),-conj(state0[0])]
            
            return [state0,state1]

        # add in missing zeros
        for pauli in ['X', 'Y', 'Z']:
            if pauli not in target_expect:
                target_expect[pauli] = 0
        
        # determine the unitary which rotates as close to the target state as possible
        current_basis = get_basis(self.get_bloch(qubit))
        target_basis = get_basis(target_expect)
        U = array([ [0 for _ in range(2)] for _ in range(2) ], dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[j][k] += target_basis[i][j]*conj(current_basis[i][k])

        # get the unitary for the desired fraction of the rotation
        if fraction!=1:
            U = pwr(U, fraction)
        
        # apply the corresponding gate
        the,phi,lam = OneQubitEulerDecomposer().angles(U)

        self.qc.u(the, phi, lam, qubit)

        if update:
            self.update_tomography()
             
    def set_relationship(self,relationships,qubit0,qubit1,fraction=1, update=True):
        '''
        Rotates the given pair of qubits towards the given target expectation values.
        
        Args:
            target_state: Target expectation values.
            qubit0, qubit1: Qubits on which the operation is applied
            fraction: fraction of the rotation toward the target state to apply.
            update: whether to update the tomography after the rotation is added to the circuit.
        '''
        zero = 0.001

        def inner(vec1,vec2):
            inner = 0
            for j in range(len(vec1)):
                inner += conj(vec1[j])*vec2[j]
            return inner

        def normalize(vec):
            renorm = sqrt(inner(vec,vec))
            if abs((renorm*conj(renorm)))>zero:
                return np.copy(vec)/renorm
            else:
                return [nan for amp in vec]
            
        def random_vector(ortho_vecs=[]):
            vec = np.array([ 2*random()-1 for _ in range(4) ],dtype='complex')
            vec[0] = abs(vec[0])
            for ortho_vec in ortho_vecs:
                vec -= inner(ortho_vec,vec)*ortho_vec
            return normalize(vec)
        
        def get_rho(qubit0,qubit1):
            rel = self.get_relationship(qubit0,qubit1)
            b0 = self.get_bloch(qubit0)
            b1 = self.get_bloch(qubit1)
            rho = np.identity(4,dtype='complex128')
            for pauli in ['X','Y','Z']:
                rho += b0[pauli]*matrices[pauli+'I']
                rho += b1[pauli]*matrices['I'+pauli]
            for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
                rho += rel[pauli]*matrices[pauli]
            return rho/4

        raw_vals,raw_vecs = la.eigh( get_rho(qubit0,qubit1) )
        vals = sorted([(val,k) for k,val in enumerate(raw_vals)], reverse=True)
        vecs = [[ raw_vecs[j][k] for j in range(4)] for (val,k) in vals]
        
        Pup = np.identity(4,dtype='complex')
        for (pauli,sign) in relationships.items():
            Pup = dot(Pup, (matrices['II']+sign*matrices[pauli])/2)
        Pdown = (matrices['II'] - Pup)

        new_vecs = [[nan for _ in range(4)] for _ in range(4)]
        valid = [False for _ in range(4)]

        # the first new vector comes from projecting the first eigenvector
        vec = np.copy(vecs[0])
        while not valid[0]:
            new_vecs[0] = normalize(dot(Pup,vec))
            valid[0] = True not in [isnan(new_vecs[0][j]) for j in range(4)]
            # if that doesn't work, a random vector is projected instead
            vec = random_vector()
        
        # ensure that the new vectors are orthogonal to each other
        for j in range(1, 4):
            vec = dot(Pup, vecs[j])
            for k in range(j):
                vec -= inner(new_vecs[k], vec) * new_vecs[k]
            while not valid[j]:
                new_vecs[j] = normalize(vec)
                valid[j] = not any(isnan(new_vecs[j][k]) for k in range(4))
                vec = random_vector(ortho_vecs=new_vecs[:j])

            
        # a unitary is then constructed to rotate the old basis into the new
        U = np.zeros((4, 4), dtype=complex)
        for j in range(4):
            U += outer(new_vecs[j],conj(vecs[j]))

        if fraction!=1:
            U = pwr(U, fraction)
            
        try:
            decomposer = TwoQubitBasisDecomposer(CXGate())
            circuit = decomposer(U)
            gate = circuit.to_instruction()
        except Exception as e:
            print(e)
            gate = None

        if gate:
            self.qc.append(gate,[qubit0,qubit1])

        if update:
            self.update_tomography()

        return gate
        
        

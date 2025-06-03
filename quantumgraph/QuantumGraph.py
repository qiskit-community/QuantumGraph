#from qiskit import QuantumCircuit, execute, Aer #(not supported in qiskit 2.x.x)
from qiskit import transpile, QuantumCircuit #(corrected to qiskit 2.0.2 version)
from qiskit_aer import AerSimulator #(corrected to qiskit-aer version 0.17.x compatible with qiskit 2.0.2)


'''
previously written
# from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
# from qiskit.quantum_info.synthesis.two_qubit_decompose import two_qubit_cnot_decompose
'''

# corrected to qiskit 2.0.2 version 
from qiskit.synthesis import OneQubitEulerDecomposer #(corrected to qiskit 2.0.2 version)
from qiskit.synthesis import TwoQubitBasisDecomposer #(corrected to qiskit 2.0.2 version)
from qiskit.circuit.library import CXGate

from qiskit.quantum_info import partial_trace, DensityMatrix # (corrected to qiskit 2.0.2 version)

#from pairwise_tomography.pairwise_state_tomography_circuits import pairwise_state_tomography_circuits
#from pairwise_tomography.pairwise_fitter import PairwiseStateTomographyFitter
#the above imports is dependent on qiskit-ignis and ignis is not supported in qiskit 2.x.x

from qiskit_experiments.library.tomography import StateTomography # (corrected to qiskit-experiments 0.6.0 compatible with qiskit 2.0.2)
from qiskit.result import Result # (corrected to qiskit 2.0.2 version)
from qiskit_experiments.library.tomography import *

from quantumgraph.ExpectationValue import ExpectationValue

from numpy import pi, cos, sin, sqrt, exp, arccos, arctan2, conj, array, kron, dot, outer, nan, isnan
import numpy as np
from numpy.random import normal

from scipy import linalg as la
from scipy.linalg import fractional_matrix_power as pwr

from random import random, choice

import time

try:
    from qiskit import IBMQ
    IBMQ.load_account()
except:
    print('An IBMQ account could not be loaded')


# used in workaround for qiskit-aer issue 1015
class FakeResult():
    def __init__(self):
        self.circuits = []
        self.counts = []
    def set_counts(self,circ,counts):
        self.circuits.append(circ)
        self.counts.append(counts)
    def get_counts(self,circ):
        return self.counts[self.circuits.index(circ)]
    
    
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
    
    def __init__ (self,num_qubits,coupling_map=[],device='simulator'):
        '''
        Args:
            num_qubits: The number of qubits, and hence the number of nodes in the graph.
            coupling_map: A list of pairs of qubits, corresponding to edges in the graph.
                If none is given, a fully connected graph is used.
            device: The device on which the graph will be run. Can be given as a Qiskit backend object
                or a description of the device as a string. If none is given, a local simulator is used.
        '''
        
        self.num_qubits = num_qubits
        
        # the coupling map consists of pairs [j,k] with the convention j<k
        self.coupling_map = []
        for j in range(self.num_qubits-1):
            for k in range(j+1,self.num_qubits):
                if ([j,k] in coupling_map) or ([j,k] in coupling_map) or (not coupling_map):
                    self.coupling_map.append([j,k])
              
        # use the `device` input to make a Qiskit backend object
        if type(device) is str:
            if device=='simulator':
                self.backend = AerSimulator() #You dont need to specify the backend as 'qasm_simulator', because in qiskit 2.x.x AerSimulator() is the default backend for qasm_simulation.
            else:
                try:
                    if device[0:4]=='ibmq':
                        backend_name = device
                    else:
                        backend_name = 'ibmq_' + device # fixed to work with the IBMQ backend names
                    for provider in IBMQ.providers(): 
                        for potential_backend in provider.backends():
                            if potential_backend.name()==backend_name:
                                self.backend = potential_backend
                    self.coupling_map = self.backend.configuration().coupling_map
                except:
                    print('The given device does not correspond to a valid IBMQ backend')
        else:
            self.backend = device
        
        # create the quantum circuit, and initialize the tomography
        self.qc = QuantumCircuit(self.num_qubits)
        self.update_tomography()
        
    def update_tomography(self, shots=8192):
        '''
        Updates the `tomography` attribute of the `QuantumGraph` object, which contain the two qubit tomography.
        The update is persomed by running the current circuit on the backend.
        
        Args:
            shots: Number of shots to use.
        '''
        
        def get_status(job):
            '''
            Get the status of a submitted job, mitigating for the fact that there may be a disconnection.
            '''
            try:
                job_status = job.status().value
            except:
                job_status = 'Something is wrong. Perhaps disconnected.'
            return job_status
        
        def submit_job(circs):
            '''
            Submit a job until it has been verified to have been submitted.
            If the circuit is empty, this is done using the 'qasm_simulator' rather than the specified backend.
            
            Args:
                circs: A list of circuits to run.
            
            Returns:
                The job object of the submitted job.
            '''
            if len(self.qc.data) == 0:
                sim = AerSimulator()
                job = sim.run(circs, shots=shots)
            else:
                submitted = False
                while not submitted==False:
                    try:
                        job = self.backend.run(circs, shots=shots)
                        submitted = True
                    except:
                        print('Submission failed. Trying again in a minute')
                        time.sleep(60)
            return job
        
        def get_result(circs):
            '''
            Submits a list of circuits, waits until they run and then returns the results object.
            
            Args:
                circs: A list of circuits to run.
            
            Returns:
                The results object for the circuits that have been run.
            '''
            if self.backend.name() == 'qasm_simulator' or len(self.qc.data) == 0:
                result = FakeResult()
                sim = AerSimulator()
                for circ in circs:
                    counts = sim.run(circ, shots=shots).result().get_counts()
                    result.set_counts(circ,counts)
                return result
            else:
                job = submit_job(circs)
                if job.backend().name()!='qasm_simulator':
                    time.sleep(1)
                    m = 0
                    while job.status().value != 'job has successfully run' and m < 60:
                        time.sleep(60)
                        print(get_status(job))
                        m += 1
                    if get_status(job)!='job has successfully run':
                        print('After 1 hour, job status is ' + job.status().value + '. Another job will be submitted')
                        job = submit_job(circs)
                return job.result()

        if type(self.backend)==ExpectationValue:
            self.backend = ExpectationValue(self.backend.n,
                                            k=self.backend.k,
                                            pairs=self.backend.pairs)
            self.backend.apply_circuit(self.qc)
        else:
            tomo_exp = StateTomography(self.qc)
            exp_data = tomo_exp.run(self.backend, shots=shots)
            exp_data.block_for_results()
            self.tomography = exp_data.analysis_results(0).value  # This is the density matrix
    
    def get_bloch(self,qubit):
        '''
        Returns the X, Y and Z expectation values for the given qubit.
        '''

        expect = {}
        if type(self.backend) == ExpectationValue:
            full_pauli = ['I'] * self.num_qubits
            for pauli in ['X', 'Y', 'Z']:
                full_pauli[qubit] = pauli
                expect[pauli] = self.backend.pauli_decomp[''.join(full_pauli)]
        else:
            # --- REPLACE EVERYTHING BELOW THIS LINE ---
            #if qubit==self.num_qubits-1:
            #    q0,q1 = qubit-1,qubit
            #else:
            rho = self.tomography  # This is the full density matrix
            # Build the single-qubit Pauli operator for the target qubit
            for pauli in ['X', 'Y', 'Z']:
                op = 1
                n = self.num_qubits
                for i in range(n):
                    if i == qubit:
                        op = np.kron(matrices[pauli], op)
                    else:
                        op = np.kron(matrices['I'], op)
                expect[pauli] = np.trace(rho @ op).real
        return expect
    
    def get_relationship(self, qubit0, qubit1):
        '''
        Returns the two qubit pauli expectation values for a given pair of qubits.
        '''
        relationship = {}
        if type(self.backend) == ExpectationValue:
            full_pauli = ['I']*self.num_qubits
            for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
                full_pauli[qubit0] = pauli[0]
                full_pauli[qubit1] = pauli[1]
                key = ''.join(full_pauli)
                if key in self.backend.pauli_decomp:
                    val = self.backend.pauli_decomp[key]
                else:
                    # Use product of marginals if not present
                    bloch0 = self.get_bloch(qubit0)
                    bloch1 = self.get_bloch(qubit1)
                    val = bloch0[pauli[0]] * bloch1[pauli[1]]
                relationship[pauli] = val
                # Reset for next loop
                full_pauli[qubit0] = 'I'
                full_pauli[qubit1] = 'I'
        else:
            rho = self.tomography  # Full density matrix
            n = self.num_qubits
            for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
                op = 1
                for i in range(n):
                    if i == qubit0:
                        op = np.kron(op, matrices[pauli[0]])
                    elif i == qubit1:
                        op = np.kron(op, matrices[pauli[1]])
                    else:
                        op = np.kron(op, matrices['I'])
                relationship[pauli] = np.trace(rho @ op).real
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
        def basis_change(pole,basis,qubit,dagger=False):
            '''
            Returns the circuit required to change from the Z basis to the eigenbasis
            of a particular Pauli. The opposite is done when `dagger=True`.
            '''
            if pole=='+' and dagger==True:
                self.qc.x(qubit)
            
            if basis=='X':
                self.qc.h(qubit)
            elif basis=='Y':
                if dagger:
                    self.qc.rx(-pi/2,qubit)
                else:
                    self.qc.rx(pi/2,qubit)
                    
            if pole=='+' and dagger==False:
                self.qc.x(qubit)
                    
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
        
        # Debug: Print the current and target Bloch vectors
        print(f"set_bloch called for qubit {qubit} with target_expect: {target_expect}, fraction: {fraction}, update: {update}")

        # add in missing zeros
        for pauli in ['X', 'Y', 'Z']:
            if pauli not in target_expect:
                target_expect[pauli] = 0

        # Debug: Print the current and target Bloch vectors
        print(f"Current Bloch vector for qubit {qubit}: {self.get_bloch(qubit)}")

        # normalize the target expectation values
        print(f"Target Bloch vector for qubit {qubit}: {target_expect}")
        
        # determine the unitary which rotates as close to the target state as possible
        current_basis = get_basis(self.get_bloch(qubit))
        target_basis = get_basis(target_expect)
        U = array([ [0 for _ in range(2)] for _ in range(2) ], dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[j][k] += target_basis[i][j]*conj(current_basis[i][k])

        # Debug: Print the computed unitary matrix
        print(f"Computed unitary matrix U for qubit {qubit}:\n{U}")

        # get the unitary for the desired fraction of the rotation
        if fraction!=1:
            U = pwr(U, fraction)
        
        # Debug: Print the fractional unitary matrix
        print(f"Fractional unitary matrix U for qubit {qubit} (fraction={fraction}):\n{U}")
        
        # apply the corresponding gate
        the,phi,lam = OneQubitEulerDecomposer().angles(U)

        # Debug: Print the Euler angles
        print(f"Euler angles for qubit {qubit}: theta={the}, phi={phi}, lambda={lam}")

        self.qc.u(the, phi, lam, qubit)

        if update:
            self.update_tomography()
            # Debug: Print confirmation of tomography update
            print("Tomography updated after applying set_bloch.")
             
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
            if type(self.backend)==ExpectationValue:
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
            else:
                # Extract the reduced density matrix for qubit0 and qubit1
                rho_full = DensityMatrix(self.tomography)
                reduced = partial_trace(rho_full, [i for i in range(self.num_qubits) if i not in [qubit0, qubit1]])
                # The result is a 2-qubit DensityMatrix object
                return reduced.data

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
            
        # the second is found by similarly projecting the second eigenvector
        # and then finding the component orthogonal to new_vecs[0]
        vec = dot(Pup,vecs[1])
        while not valid[1]:
            new_vecs[1] = vec - inner(new_vecs[0],vec)*new_vecs[0]
            new_vecs[1] = normalize(new_vecs[1])
            valid[1] = True not in [isnan(new_vecs[1][j]) for j in range(4)]
            # if that doesn't work, start with a random one instead
            vec = random_vector()

        # the third is the projection of the third eigenvector to the subpace orthogonal to the first two
        vec = np.copy(vecs[2])
        for j in range(2):
            vec -= inner(new_vecs[j],vec)*new_vecs[j]
        while not valid[2]:
            new_vecs[2] = normalize(vec)
            valid[2] = True not in [isnan(new_vecs[2][j]) for j in range(4)]
            # if that doesn't work, use a random vector orthogonal to the first two
            vec = random_vector(ortho_vecs=[new_vecs[0],new_vecs[1]])

        # the last is just orthogonal to the rest
        vec = normalize(dot(Pdown,vecs[3]))
        while not valid[3]:
            new_vecs[3] = random_vector(ortho_vecs=[new_vecs[0],new_vecs[1],new_vecs[2]])
            valid[3] = True not in [isnan(new_vecs[3][j]) for j in range(4)]
            
        # a unitary is then constructed to rotate the old basis into the new
        '''
        The unitary is constructed as follows:
        U = sum_j ( |new_vecs[j]><vecs[j]| )
        
        where |new_vecs[j]> is the j-th new vector and <vecs[j]| is the j-th old vector.
        If the fraction is not 1, then the unitary is raised to the power of `fraction`.
        The unitary is then decomposed into a circuit using the CX gate.
        The circuit is then appended to the quantum circuit.
        If the decomposition fails, None is returned.
        '''
        U = np.zeros((4, 4), dtype=complex) # What changed here is the initialization of U to a complex array because the outer product will produce complex numbers.
        for j in range(4):
            U += outer(new_vecs[j],conj(vecs[j]))

        if fraction!=1:
            U = pwr(U, fraction)
            
        try:
            decomposer = TwoQubitBasisDecomposer(CXGate())
            circuit = decomposer(U)
            gate = circuit.to_instruction()
            #done = True
        except Exception as e:
            print(e)
            gate = None

        if gate:
            self.qc.append(gate,[qubit0,qubit1])

        if update:
            self.update_tomography()

        return gate
        
        

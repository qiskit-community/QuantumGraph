from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import two_qubit_cnot_decompose

from pairwise_tomography.pairwise_state_tomography_circuits import pairwise_state_tomography_circuits
from pairwise_tomography.pairwise_fitter import PairwiseStateTomographyFitter

from numpy import pi, cos, sin, sqrt, exp, arccos, arctan2, conj, array, kron, dot, outer, nan, isnan
import numpy as np
from numpy.random import normal

from scipy import linalg as la
from scipy.linalg import fractional_matrix_power as pwr

from random import random, choice

try:
    IBMQ.load_account()
except:
    print('An IBMQ account could not be loaded')


import time

# define the Pauli matrices in a dictionary
matrices = {}
matrices['I'] = [[1,0],[0,1]]
matrices['X'] = [[0,1],[1,0]]
matrices['Y'] = [[0,-1j],[1j,0]]
matrices['Z'] = [[1,0],[0,-1]]
for pauli1 in ['I','X','Y','Z']:
    for pauli2 in ['I','X','Y','Z']:
        matrices[pauli1+pauli2] = kron(matrices[pauli1],matrices[pauli2])
        
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
                self.backend = Aer.get_backend('qasm_simulator')
            else:
                try:
                    if device[0:4]=='ibmq':
                        backend_name = device
                    else:
                        backend_name = 'ibmq_' + backend
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
            if len(self.qc.data)==0:
                job = execute(circs, Aer.get_backend('qasm_simulator'), shots=shots)
            else:
                submitted = False
                while submitted==False:
                    try:
                        job = execute(circs, self.backend, shots=shots)
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
            job = submit_job(circs)
            time.sleep(1)
            while get_status(job)!='job has successfully run':
                m = 0
                while m<60 and get_status(job)!='job has successfully run':
                    time.sleep(60)
                    print(get_status(job))
                    m += 1
                if get_status(job)!='job has successfully run':
                    print('After 1 hour, job status is ' + get_status(job) + '. Another job will be submitted')
                    job = submit_job(circs)
            return job.result()

        
        tomo_circs = pairwise_state_tomography_circuits(self.qc, self.qc.qregs[0])
        tomo_results = get_result(tomo_circs)
        self.tomography = PairwiseStateTomographyFitter(tomo_results, tomo_circs, self.qc.qregs[0]) 
        
    
    def get_bloch(self,qubit):
        '''
        Returns the X, Y and Z expectation values for the given qubit.
        '''
        expect = {'X':0, 'Y':0, 'Z':0}
        for q in range(self.num_qubits):
            if q!=qubit:
                (q0,q1) = sorted([q,qubit])
                pair_expect = self.tomography.fit(output='expectation')[q0,q1]
                for pauli in expect:
                    pauli_pair = (pauli,'I')
                    if q0!=qubit:
                        pauli_pair = tuple(list((pauli,'I'))[::-1])
                    expect[pauli] += pair_expect[pauli_pair]/(self.num_qubits-1)     
        return expect
    
    def get_relationship(self,qubit1,qubit2):  
        '''
        Returns the two qubit pauli expectation values for a given pair of qubits.
        '''
        (q0,q1) = sorted([qubit1,qubit2])
        reverse = (q0,q1)!=(qubit1,qubit2)
        pair_expect = self.tomography.fit(output='expectation')[q0,q1]
        relationship = {}
        for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
            if reverse:
                new_pauli = pauli[::-1]
            else:
                new_pauli = pauli
            relationship[new_pauli] = pair_expect[pauli[0],pauli[1]]
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
        self.qc.u3(the,phi,lam,qubit)

        if update:
            self.update_tomography()
             
    def set_relationship(self,relationships,qubit0,qubit1,fraction=1, update=True):
        '''
        
        Warning: This doesn't fully work yet!
        
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
                return vec/sqrt(inner(vec,vec))
            else:
                return [nan for amp in vec]
            
        def random_vector(ortho_vecs=[]):
            vec = np.array([ 2*random()-1 for _ in range(4) ],dtype='complex')
            vec[0] = abs(vec[0])
            for ortho_vec in ortho_vecs:
                vec -= inner(ortho_vec,vec)*ortho_vec
            return normalize(vec) 

        (q0,q1) = sorted([qubit0,qubit1])
        
        rho = self.tomography.fit(output='density_matrix')[q0,q1]#,pairs_list=[(q0,q1)])[q0,q1]
        raw_vals,raw_vecs = la.eigh(rho)

        vals = sorted([(val,k) for k,val in enumerate(raw_vals)], reverse=True)
        vecs = [[ raw_vecs[j][k] for j in range(4)] for (val,k) in vals]

        Pup = matrices['II']
        for (pauli,sign) in relationships:
            Pup = dot(Pup, (matrices['II']+sign*matrices[pauli])/2)
        Pdown = (matrices['II'] - Pup)

        new_vecs = [[nan for _ in range(4)] for _ in range(4)]
        valid = [False for _ in range(4)]

        # the first new vector comes from projecting the first eigenvector
        vec = vecs[0]
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
        vec = vecs[2]
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
        U = [[0 for _ in range(4)] for _ in range(4)]
        for j in range(4):
            U += outer(new_vecs[j],conj(vecs[j]))

        if fraction!=1:
            U = pwr(U, fraction)
            
        try:
            circuit = two_qubit_cnot_decompose(U)
            gate = circuit.to_instruction()
            done = True
        except Exception as e:
            print(e)
            gate = None

        if gate:
            self.qc.append(gate,[qubit0,qubit1])

        if update:
            self.update_tomography()

        return gate
        
        

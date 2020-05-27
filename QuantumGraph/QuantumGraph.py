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
        Updates the tomography information of the graph for its current circuit, by running on the backend.
        
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
        result = get_result(tomo_circs)
        
        self.tomography = {}
        for index, qc in enumerate(tomo_circs):
            basis = eval(qc.name)
            counts = result().get_counts(index)
            self.tomography[basis] = {}
            for string in counts:
                self.tomography[basis][string] = counts[string]/shots
                
    
    def get_state(self,qubit):
        state = {}
        for pauli in ['X','Y','Z']:
            for basis in self.tomography:
                if basis[qubit]==pauli:
                    probs = self.tomography[basis]
            state[pauli] = 0
            for string in probs:
                sign = 2*(string[-qubit-1]=='0')-1
                state[pauli[0]] += sign*probs[string]
        return state
    
    def get_relationship(self,qubit1,qubit2):  
        
        (j,k) = sorted([qubit1,qubit2])
        reverse = (j,k)!=(qubit1,qubit2)
        relationship = {}
        for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
            if reverse:
                new_pauli = pauli[::-1]
            else:
                new_pauli = pauli
            for basis in self.tomography:
                if basis[j]==pauli[0] and basis[k]==pauli[1]:
                    probs = self.tomography[basis]
            relationship[new_pauli] = 0
            for string in probs: 
                sign = 2*(string[-j-1]==string[-k-1])-1
                relationship[new_pauli] += sign*probs[string]
        return relationship
    
    def set_state(self,target_expect,qubit,fraction=1,update=True):
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
            Returns the given expectation values after normaliztion.
            '''
            R = sqrt( expect['X']**2 + expect['Y']**2 + expect['Z']**2 )
            return {pauli:expect[pauli]/R for pauli in expect}
        
        def get_basis(expect):
            '''
            Get the eigenbasis of the density matric for a the given expectation values.
            '''
            normalized_expect = normalize(expect)
            
            theta = arccos(normalized_expect['Z'])
            phi = arctan2(normalized_expect['Y'],normalized_expect['X'])
            
            state0 = [cos(theta/2),exp(1j*phi)*sin(theta/2)]
            state1 = [conj(state0[1]),-conj(state0[0])]
            
            return [state0,state1]

        # determine the unitary which rotates as close to the target state as possible
        current_basis = get_basis(self.get_state(qubit))
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
            
        expect = {'II':1.0}
        for pauli in self.expect[qubit0]:
            expect['I'+pauli] = self.expect[qubit0][pauli]
        for pauli in self.expect[qubit1]:
            expect[pauli+'I'] = self.expect[qubit1][pauli]
        pair = (min(qubit0,qubit1), max(qubit0,qubit1))
        for pauli in self.expect[pair]:
            if qubit0<qubit1:
                expect[pauli] = self.expect[pair][pauli]
            else:
                expect[pauli[::-1]] = self.expect[pair][pauli]

        rho = [[0 for _ in range(4)] for _ in range(4)]
        for pauli in expect:
            for r in range(4):
                for c in range(4):
                    rho[r][c] += expect[pauli]*matrices[pauli][r][c]/4

        raw_vals,raw_vecs = la.eig(rho)

        vals = sorted([(val,k) for k,val in enumerate(raw_vals)], reverse=True)
        vecs = [[ raw_vecs[j][k] for j in range(4)] for (val,k) in vals]

        Pup = matrices['II']
        for (pauli,sign) in relationships:
            Pup = dot(Pup, (matrices['II']+sign*matrices[pauli])/2)
        Pdown = (matrices['II'] - Pup)

        new_vecs = [[nan for _ in range(4)] for _ in range(4)]
        valid = [False for _ in range(4)]

        # the first new vector comes from projecting the first eigenvector
        new_vecs[0] = dot(Pup,vecs[0])
        new_vecs[0] = normalize(new_vecs[0])
        valid[0] = True not in [isnan(new_vecs[0][j]) for j in range(4)]

        # if it didn't project away to nothing, the second is found by similarly projecting
        # the second eigenvector and then finding the component orthogonal to new_vecs[0]
        if valid[0]:
            new_vecs[1] = dot(Pup,vecs[1])
            new_vecs[1] -= inner(new_vecs[0],new_vecs[1])*new_vecs[0]
            new_vecs[1] = normalize(new_vecs[1])
            valid[1] = True not in [isnan(new_vecs[1][j]) for j in range(4)]

        # the process repeats for the next two, bit with the opposite projection
        if valid[0] and valid[1]:
            new_vecs.append(dot(Pdown,vecs[2]))
            new_vecs[2] = normalize(new_vecs[2])
            valid[2] = True not in [isnan(new_vecs[2][j]) for j in range(4)]
        if valid[0] and valid[1] and valid[2]:    
            new_vecs.append(dot(Pdown,vecs[3]))
            new_vecs[3] -= inner(new_vecs[2],new_vecs[3])*new_vecs[2]
            new_vecs[3] = normalize(new_vecs[3])
            valid[3] = True not in [isnan(new_vecs[3][j]) for j in range(4)]

        # if the first succeeds but any of the last three fail,
        # replace them all with a random set of orthogonal gates
        if valid[0] and False in [valid[1], valid[2], valid[3]]:

            new_vecs[1] = [ random() for _ in range(4) ]
            new_vecs[1] -= inner(new_vecs[0],new_vecs[1])*new_vecs[0]
            new_vecs[1] = normalize(new_vecs[1])

            new_vecs[2] = [ random() for _ in range(4) ]
            new_vecs[2] -= inner(new_vecs[0],new_vecs[2])*new_vecs[0]
            new_vecs[2] -= inner(new_vecs[1],new_vecs[2])*new_vecs[1]
            new_vecs[2] = normalize(new_vecs[2])

            new_vecs[3] = [ random() for _ in range(4) ]
            new_vecs[3] -= inner(new_vecs[0],new_vecs[3])*new_vecs[0]
            new_vecs[3] -= inner(new_vecs[1],new_vecs[3])*new_vecs[1]
            new_vecs[3] -= inner(new_vecs[2],new_vecs[3])*new_vecs[2]
            new_vecs[3] = normalize(new_vecs[3])

        # a unitary is then construct to the old basis into the new
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
        
        

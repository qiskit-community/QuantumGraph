from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.quantum_info.synthesis import euler_angles_1q
from qiskit.quantum_info.synthesis.two_qubit_decompose import two_qubit_cnot_decompose

from pairwise_tomography.pairwise_state_tomography_circuits import pairwise_state_tomography_circuits
from pairwise_tomography.pairwise_fitter import PairwiseStateTomographyFitter

from numpy import pi, cos, sin, sqrt, exp, arccos, arctan2, conj, array, kron, dot, outer, nan, isnan
from numpy.random import normal

from scipy import linalg as la
from scipy.linalg import fractional_matrix_power as pwr

from random import random, choice

import time

matrices = {}
matrices['I'] = [[1,0],[0,1]]
matrices['X'] = [[0,1],[1,0]]
matrices['Y'] = [[0,-1j],[1j,0]]
matrices['Z'] = [[1,0],[0,-1]]
for pauli1 in ['I','X','Y','Z']:
    for pauli2 in ['I','X','Y','Z']:
        matrices[pauli1+pauli2] = kron(matrices[pauli1],matrices[pauli2])

class Job ():
    def __init__(self, circuits, counts):
        self._circuits = circuits
        self._counts = counts
        
    def result(self):
        return Results(self._circuits, self._counts)

class Results ():
    def __init__(self, circuits, counts):
        self._circuits = circuits
        self._counts = counts
        
    def get_counts(self,index=0):
        if type(index)==QuantumCircuit:
            index = self._circuits.index(index)
        return self._counts[index]

def fake_execute(circuits, shots=1024):
    
    if type(circuits)!=list:
        circuits = [circuits]
    
    counts = []
    for qc in circuits:

        n = qc.n_qubits
        m = len((qc).clbits)
        
        counts_qc = {}
        for _ in range(shots):
            string = ''
            for _ in range(m):
                string += choice(['0','1'])
            if string in counts_qc:
                counts_qc[string] += 1
            else:
                counts_qc[string] = 1
                        
        counts.append(counts_qc)
                
    return Job(circuits,counts)
        
class QuantumGraph ():
    
    def __init__ (self,num_qubits,coupling_map=[],backend='simulator'):
        
        self.num_qubits = num_qubits
        
        self.coupling_map = []
        for j in range(self.num_qubits-1):
            for k in range(j+1,self.num_qubits):
                if ([j,k] in coupling_map) or ([j,k] in coupling_map) or (not coupling_map):
                    self.coupling_map.append([j,k])
              
        if backend=='depolarized':
            self.backend = 'depolarized'
        elif backend=='simulator':
            self.backend = Aer.get_backend('qasm_simulator')
        elif backend=='rochester':
            IBMQ.load_account()
            for provider in IBMQ.providers():
                for potential_backend in provider.backends():
                    if potential_backend.name()=='ibmq_rochester':
                        self.backend = potential_backend
            self.coupling_map = self.backend.configuration().coupling_map
                                  
        self.qc = QuantumCircuit(self.num_qubits)
        
        self.update_tomography()
        
    def update_tomography(self, shots=8192):
        
        def get_status(job):
            try:
                job_status = job.status().value
            except:
                job_status = 'Something is wrong. Perhaps disconnected.'
            return job_status
        
        def submit_job(tomo_circs):
            submitted = False
            while submitted==False:
                try:
                    job = execute(tomo_circs, self.backend, shots=shots)
                    submitted = True
                except:
                    print('Submission failed. Trying again in a minute')
                    time.sleep(60)
            return job
            
        tomo_circs = pairwise_state_tomography_circuits(self.qc, self.qc.qregs[0])
        if self.backend=='depolarized':
            job = fake_execute(tomo_circs,shots=shots)
        elif self.backend==Aer.get_backend('qasm_simulator'):
            job = submit_job(tomo_circs)
        else:
            job = submit_job(tomo_circs)
            while get_status(job)!='job has successfully run':
                m = 0
                while m<60 and get_status(job)!='job has successfully run':
                    time.sleep(60)
                    print(get_status(job))
                    m += 1
                if get_status(job)!='job has successfully run':
                    print('After 1 hour, job status is ' + get_status(job) + '. Another job will be submitted')
                    job = submit_job(tomo_circs)
        
        self.tomography = {}
        for index, qc in enumerate(tomo_circs):
            basis = eval(qc.name)
            self.tomography[basis] = {}
            counts = job.result().get_counts(index)
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
            R = sqrt( expect['X']**2 + expect['Y']**2 + expect['Z']**2 )
            return {pauli:expect[pauli]/R for pauli in expect}
        
        def get_basis(expect):
            
            normalized_expect = normalize(expect)
            
            theta = arccos(normalized_expect['Z'])
            phi = arctan2(normalized_expect['Y'],normalized_expect['X'])
            
            state0 = [cos(theta/2),exp(1j*phi)*sin(theta/2)]
            state1 = [conj(state0[1]),-conj(state0[0])]
            
            return [state0,state1]

        if not self.backend=='depolarized':
        
            current_basis = get_basis(self.get_state(qubit))
            target_basis = get_basis(target_expect)

            U = array([ [0 for _ in range(2)] for _ in range(2) ], dtype=complex)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U[j][k] += target_basis[i][j]*conj(current_basis[i][k])

            if fraction!=1:
                U = pwr(U, fraction)

            the,phi,lam = euler_angles_1q(U)
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
            
        if not self.backend=='depolarized':

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
        
        

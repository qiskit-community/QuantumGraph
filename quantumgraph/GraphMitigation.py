import scipy.linalg as la
import numpy as np

from qiskit import ClassicalRegister
from qiskit.ignis.verification.tomography import StateTomographyFitter
from qiskit.ignis.verification.tomography.data import marginal_counts

'''
These tools are heavily inspired by
https://github.com/if-quantum/pairwise-tomography
'''

def pairwise_mitigation_circuits(circuit, meas_qubits=None):
    
    # if no `meas_qubits` is supplied, mitigation will be
    # for all qubits in the first (or only) qubit register
    if not meas_qubits:
        meas_qubits = circuit.qregs[0]
    cr = ClassicalRegister(len(meas_qubits))
    
    # create a blank version of the circuit
    blank_circuit = circuit.copy()
    blank_circuit.data = []

    # determine all circuit names
    
    N = len(meas_qubits)
    n = int(np.ceil(np.log(N)/np.log(2)))+1
    
    strings = []
    for j in range(N):
        strings.append( bin(j)[2::].zfill(n))

    circuit_names = []
    for k in range(n):
        circuit_0 = []
        circuit_1 = []
        for j in range(N):
            circuit_0.append(strings[j][k])
            circuit_1.append(str((int(strings[j][k])+1)%2))
        for circuit in [circuit_0,circuit_1]:
            circuit_names.append(tuple(circuit))

    # create the circuits
            
    output_circuit_list = []
    
    for circuit_name in circuit_names:
        meas_layout = blank_circuit.copy(name=str(circuit_name))
        meas_layout.add_register(cr)
        
        for bit_index, qubit in enumerate(meas_qubits):
            if circuit_name[bit_index]=='1':
                meas_layout.x(qubit)
            meas_layout.measure(qubit,cr[bit_index])
        
        output_circuit_list.append( meas_layout )
        
    return output_circuit_list


class PairwiseMitigationFitter():
    
    def __init__(self, miti_results, miti_circs, meas_qubits=None):
        
        self.miti_results = miti_results
        self.miti_circs = miti_circs
        
        if not meas_qubits:
            meas_qubits = miti_circs[0].qregs[0]
        self.meas_qubits = meas_qubits

        
    def fit(self,pairs_list=None):
        
        bits = ['00','01','10','11']
        
        if not pairs_list:
            N = len(self.meas_qubits)
            pairs_list = []
            for j in range(N-1):
                for k in range(j+1,N):
                    pairs_list.append( (j,k) )

        Minv = {}
        for pair in pairs_list:
            
            j,k = pair
            probs = {s:{ms:0 for ms in bits} for s in bits}
            for circuit in self.miti_circs:
                circuit_name = eval(circuit.name)
                s = circuit_name[k]+circuit_name[j]
                counts = marginal_counts(self.miti_results.get_counts(circuit), [j, k])
                for ms in counts:
                    probs[s][ms] += counts[ms]
            for s in probs:
                shots = sum(probs[s].values())
                for ms in probs[s]:
                    probs[s][ms] /= shots 
            
            M = [[ probs[s][ms] for s in bits ] for ms in bits]
            Minv[pair] = la.inv(M)
                    
        return Minv
    
    def mitigate_counts(self,counts,pair):
                
        bits = ['00','01','10','11']
        
        shots = sum(counts.values())
        
        for s in bits:
            if s not in counts:
                counts[s] = 0
                
        c = np.array([ counts[s] for s in bits])
        Minv = self.fit(pairs_list=[pair])[pair]
        c = np.dot(Minv,c)
        
        for j,s in enumerate(bits):
            counts[s] = max(c[j],0)
            
        new_shots = sum(counts.values())
        for j,s in enumerate(bits):
            counts[s] *= shots/new_shots
        
        
        return counts
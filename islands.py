from qiskit import QuantumCircuit, execute, Aer
from math import pi
import numpy as np
from opensimplex import OpenSimplex
import random

import matplotlib.pyplot as plt
from matplotlib import cm

def get_points (size):

    coupling_map = [[0, 1], [0, 5], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 6], [5, 0], [5, 9], [6, 4], [6, 13], [7, 8], [7, 16], [8, 7], [8, 9], [9, 5], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [11, 17], [12, 11], [12, 13], [13, 6], [13, 12], [13, 14], [14, 13], [14, 15], [15, 14], [15, 18], [16, 7], [16, 19], [17, 11], [17, 23], [18, 15], [18, 27], [19, 16], [19, 20], [20, 19], [20, 21], [21, 20], [21, 22], [21, 28], [22, 21], [22, 23], [23, 17], [23, 22], [23, 24], [24, 23], [24, 25], [25, 24], [25, 26], [25, 29], [26, 25], [26, 27], [27, 18], [27, 26], [28, 21], [28, 32], [29, 25], [29, 36], [30, 31], [30, 39], [31, 30], [31, 32], [32, 28], [32, 31], [32, 33], [33, 32], [33, 34], [34, 33], [34, 35], [34, 40], [35, 34], [35, 36], [36, 29], [36, 35], [36, 37], [37, 36], [37, 38], [38, 37], [38, 41], [39, 30], [39, 42], [40, 34], [40, 46], [41, 38], [41, 50], [42, 39], [42, 43], [43, 42], [43, 44], [44, 43], [44, 45], [44, 51], [45, 44], [45, 46], [46, 40], [46, 45], [46, 47], [47, 46], [47, 48], [48, 47], [48, 49], [48, 52], [49, 48], [49, 50], [50, 41], [50, 49], [51, 44], [52, 48]]

    half = {}
    half[0] = [0,2,4,7,9,11,13,15,19,21,23,25,27,30,32,34,36,38,42,44,46,48,50]
    half[1] = []
    for j in range(53):
        if j not in half[0]:
            half[1].append(j)
    
    v = 0.5
    diag_x = 1
    diag_y = 0.37
    tail = 1
    
    border = size/np.sqrt(53)

    hexagons = {
        (0,0):                     [2, 1, 3, 0, 4, 5, 6, 9, 13,10,12,11],
        (-2*diag_x,-2*diag_y-2*v): [9, 8, 10,7, 11,16,17,19,23,20,22,21],
        (2*diag_x,-2*diag_y-2*v):  [13,12,14,11,15,17,18,23,27,24,26,25],
        (0,-4*diag_y-4*v):         [23,22,24,21,25,28,29,32,36,33,35,34],
        (-2*diag_x,-6*diag_y-6*v): [32,31,33,30,34,39,40,42,46,43,45,44,51],
        (+2*diag_x,-6*diag_y-6*v): [36,35,37,34,38,40,41,46,50,47,49,48,52],
    }

    points = [ (-0.1,-0.1) for j in range(53)]

    for (diag_yx,diag_yy) in hexagons:
        points[hexagons[diag_yx,diag_yy][0]] = (diag_yx,diag_yy)
        points[hexagons[diag_yx,diag_yy][1]] = (diag_yx-diag_x,diag_yy-diag_y)
        points[hexagons[diag_yx,diag_yy][2]] = (diag_yx+diag_x,diag_yy-diag_y)
        points[hexagons[diag_yx,diag_yy][3]] = (diag_yx-2*diag_x,diag_yy-2*diag_y)
        points[hexagons[diag_yx,diag_yy][4]] = (diag_yx+2*diag_x,diag_yy-2*diag_y)
        points[hexagons[diag_yx,diag_yy][5]] = (diag_yx-2*diag_x,diag_yy-2*diag_y-v)
        points[hexagons[diag_yx,diag_yy][6]] = (diag_yx+2*diag_x,diag_yy-2*diag_y-v)
        points[hexagons[diag_yx,diag_yy][7]] = (diag_yx-2*diag_x,diag_yy-2*diag_y-2*v)
        points[hexagons[diag_yx,diag_yy][8]] = (diag_yx+2*diag_x,diag_yy-2*diag_y-2*v)
        points[hexagons[diag_yx,diag_yy][9]] = (diag_yx-diag_x,diag_yy-3*diag_y-2*v)
        points[hexagons[diag_yx,diag_yy][10]] = (diag_yx+diag_x,diag_yy-3*diag_y-2*v)
        points[hexagons[diag_yx,diag_yy][11]] = (diag_yx,diag_yy-4*diag_y-2*v)
        try:
            points[hexagons[diag_yx,diag_yy][12]] = (diag_yx,diag_yy-4*diag_y-2*v-tail)
        except:
            pass
        
    min_x = size
    min_y = size
    for pos in points:
        min_x = min(min_x,pos[0])
        min_y = min(min_y,pos[1])
    points = [ (x-min_x, y-min_y) for (x,y) in points ]
       
    max_x = -size
    max_y = -size
    for pos in points:
        max_x = max(max_x,pos[0])
        max_y = max(max_y,pos[1])
    points = [ (x/max_x, y/max_y) for (x,y) in points ]
    
    points = [ (int(border+x*(size-2*border)), int(border+y*(size-2*border))) for (x,y) in points ]
    
    return points, coupling_map, half

def plot_height(height,L=None,color_map='gray'):
    # note that this function produces an image, but does not return anything

    # if no L is supplied, set it to be large enough to fit all coordinates
    if not L:
        Lmax = max(max(height.keys()))+1
        Lmin = min(min(height.keys()))
    else:
        Lmax = L
        Lmin = 0
    
    # loop over all coordinates, and set any that are not present to be 0
    for x in range(Lmin,Lmax):
        for y in range(Lmin,Lmax):
            if (x,y) not in height:
                height[x,y] = 0
    
    # put the heights in a matplotlib-friendly form
    z = [ [ height[x,y] for x in range(Lmin,Lmax)] for y in range(Lmin,Lmax) ]
 
    # plot it as a contour plot, using the supplied colour map
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    cs = ax.contourf(z,25,vmin=0,vmax=1,cmap=cm.get_cmap(color_map))
    plt.axis('off')
    plt.show()

def make_line ( length ):
    # determine the number of bits required for at least `length` bit strings
    n = int(np.ceil(np.log(length)/np.log(2)))
    # start with the basic list of bit values
    line = ['0','1']
    # each application of the following process double the length of the list,
    # and of the bit strings it contains
    for j in range(n-1):
        # this is done in each step by first appending a reverse-ordered version of the current list
        line = line + line[::-1]
        # then adding a '0' onto the end of all bit strings in the first half
        for j in range(int(len(line)/2)):
            line[j] += '0'
        # and a '1' onto the end of all bit strings in the second half
        for j in range(int(len(line)/2),int(len(line))):
            line[j] += '1'
    return line

def make_grid(L):
    
    line = make_line( L )
    
    grid = {}
    for x in range(L):
        for y in range(L):
            grid[ line[x]+line[y] ] = (x,y)
    
    return grid

def height2circuit(height,grid):
    
    n = len( list(grid.keys())[0] )
        
    state = [0]*(2**n)
    
    H = 0
    for bitstring in grid:
        (x,y) = grid[bitstring]
        if (x,y) in height:
            h = height[x,y]
            state[ int(bitstring,2) ] = np.sqrt( h )
            H += h
        
    for j,amp in enumerate(state):
        state[ j ] = amp/np.sqrt(H)
                
    qc = QuantumCircuit(n,n)
    qc.initialize( state, qc.qregs )
        
    return qc

def circuit2height(qc,grid,backend,shots=None,log=False):
    
    # get the number of qubits from the circuit
    n = qc.n_qubits
    
    # construct a circuit to perform z measurements
    meas = QuantumCircuit(n,n)
    for j in range(n):
        meas.measure(j,j)
        
    # if no shots value is supplied use 4**n by default (unless that is too small)
    if not shots:
        shots = max(4**n,8192)

    #run the circuit on the supplied backend
    counts = execute(qc+meas,backend,shots=shots).result().get_counts() 
    
    # determine max and min counts values, to use in rescaling
    if log: # log=True uses the log of counts values, instead of the values themselves
        min_h = np.log( 1/10 ) # fake small counts value for results that didn't appear
        max_h = np.log( max( counts.values() ) )
    else:
        min_h = 0
        max_h = max( counts.values() )   
    
    # loop over all bit strings in `counts`, and set the corresponding value to be
    # the height for the corresponding coordinate. Values are rescaled to ensure
    # that the biggest height is 1, and that no height is less than zero.
    height = {}
    for bitstring in counts:
        if bitstring in grid:
            if log: # log=True uses the log of counts values, instead of the values themselves
                height[ grid[bitstring] ] = ( np.log(counts[bitstring]) - min_h ) / (max_h-min_h)
            else:
                height[ grid[bitstring] ] = ( counts[bitstring] - min_h ) / (max_h-min_h)
    
    return height

def generate_seed(L,num=5):
    # generate a height map of `num` randomly chosen points, each with randomly chosen values
    seed = {}
    for _ in range(num):
        x = random.randint(0,L-1)
        y = random.randint(0,L-1)
        seed[x,y] = random.random()
    # set one to have a height of exactly 1
    seed[random.choice(list(seed.keys()))] = 1
        
    return seed

def shuffle_height (height,grid):
    
    # determine the number of qubits
    n = int( np.log(len(grid))/np.log(2) )
    
    # randomly choose a way to shuffle the bit values in the string
    shuffle = [j for j in range(n)]
    random.shuffle(shuffle)
    
    # for each bit string, determine and record the pair of positions
    # * `pos`: the position correspoding to the bit string in the given `grid`
    # * `new_pos`: the position corresponding to the shuffled version of the bit string
    remap = {}
    for bitstring in grid:
        
        shuffledstring = ''.join([bitstring[j] for j in shuffle])

        pos = grid[bitstring]
        new_pos = grid[shuffledstring]
        
        remap[pos] = new_pos
        
    # create and return `new_height`, in which each point is moved from `pos` to `new_pos`
    new_height = {}
    for pos in height:
        new_height[remap[pos]] = height[pos]
        
    return new_height

def make_height(size):
    
    cities,_ = get_points(size)
    
    height = {} 
    for x in range(size):
        for y in range(size):
            height[x,y] = 0
            for pos in cities:
                d = np.sqrt( (x-pos[0])**2 + (y-pos[1])**2 )
                D = size/np.sqrt(len(cities))
                if d<D/2:
                    height[x,y] = max( height[x,y], 1 )
                elif d<D:
                    height[x,y] = max( height[x,y], (D-d)/(D/2) )
    
    return height

def make_quantum_island( size, L=16, period=4, border=4 ):

    island = make_height(size,period)
    #quantum_island = island

    grid = make_grid(L)
    seed = generate_seed(L)
    qc = height2circuit(seed,grid)
    qc.ry(pi/4,qc.qregs[0])
    tartan = circuit2height(qc,grid,Aer.get_backend('qasm_simulator'),log=True)
    tartans = [ shuffle_height(tartan,grid) for _ in range(int(size/L)) ]
    
    quantum_island = {}
    for x in range(size):
        for y in range(size):
            quantum_island[x,y] = 0
    
    for x0 in range(0,size+int(L/2),int(L/2)):
        for y0 in range(0,size+int(L/2),int(L/2)):
            tartan = random.choice(tartans) # choose a random tartan from the list
            for (x,y) in tartan:
                xx = x-int(L/2)+x0
                yy = y-int(L/2)+y0
                if (xx,yy) in island:
                    quantum_island[xx,yy] = (1+tartan[x,y])*(1+island[xx,yy])
    
    # renormalize
    max_height = max(quantum_island.values())
    for (x,y) in quantum_island:
        quantum_island[x,y] = quantum_island[x,y]/max_height
        
    # enforce sea level and ensure capitals are on land
    cities,_ = get_points(size)
    sea_level = 0
    for j in range(size):
        for k in list(range(border))+list(range(size-border,size)):
            sea_level = max(sea_level,quantum_island[j,k],quantum_island[k,j])
    for (x,y) in quantum_island:
        if quantum_island[x,y]-sea_level<0:
            quantum_island[x,y] = 0
        else:
            quantum_island[x,y] = (quantum_island[x,y]-sea_level)/(1-sea_level)
    for city in cities:
        if quantum_island[city]==0:
            for (dx,dy) in [(0,1),(1,0),(0,-1),(-1,0)]:
                if (city[0]+dx,city[1]+dy) in quantum_island:
                    quantum_island[city[0]+dx,city[1]+dy] = 0.5
            

        
    return quantum_island
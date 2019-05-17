import numpy as np
import scipy.sparse as sparse
from scipy.optimize import minimize
from scipy.linalg import toeplitz

N = 9

def sudoku_obj(x):
    return sum(x)-81

def constraint1(x):
    return A*x-1

"""
question = np.array(
    [[5, 3, 0, 0, 7, 0, 0, 0, 2],
    [6, 0, 0, 1, 9, 5, 0, 0, 8],
    [0, 9, 8, 0, 0, 0, 0, 6, 7],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]
)
"""
"""
question = np.array(
    [[0, 5, 0, 3, 1, 0, 8, 0, 0],
    [7, 1, 0, 0, 0, 0, 6, 3, 0],
    [0, 4, 0, 9, 0, 0, 7, 1, 0],
    [0, 8, 1, 0, 0, 0, 2, 0, 4],
    [3, 2, 0, 0, 0, 0, 0, 6, 8],
    [6, 0, 4, 0, 0, 0, 1, 9, 0],
    [0, 3, 7, 0, 0, 2, 0, 5, 0],
    [0, 6, 2, 0, 0, 0, 0, 8, 7],
    [0, 0, 5, 0, 6, 4, 0, 2, 0]]
)
"""

question = np.array(
    [[3, 0, 0, 0, 0, 0, 0, 6, 2],
    [0, 6, 0, 0, 0, 8, 0, 0, 0],
    [0, 0, 1, 0, 4, 0, 0, 0, 3],
    [1, 0, 9, 8, 0, 4, 0, 0, 0],
    [0, 7, 0, 1, 0, 6, 0, 4, 0],
    [0, 0, 0, 2, 0, 3, 1, 0, 8],
    [6, 0, 0, 0, 8, 0, 9, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 2, 0],
    [5, 9, 0, 0, 0, 0, 0, 0, 1]]
)


row_C = np.append(np.ones(1),np.zeros(N-1))
row_R = np.append(np.ones(1),np.zeros(N-1))
row_R = np.reshape(row_R,[N,1])
row = toeplitz(row_C,row_R)

ROW = np.kron(row,np.kron(np.ones(9),np.eye(N)))
ROW = sparse.csc_matrix(ROW)



col_R = np.kron(np.ones(9),np.append(np.ones(1),np.zeros(N-1)))
col = toeplitz(row_C,col_R)
COL = sparse.csc_matrix(np.kron(col, np.eye(N)))

sqrtN = int(np.sqrt(N))
tep = np.zeros(int(N/sqrtN)-1)


box_R = np.kron(np.ones(sqrtN),np.append(np.ones(1),tep))
box_C = np.append(np.ones(1),np.zeros(sqrtN-1))
box_R = np.reshape(box_R,[N,1])
box = toeplitz(box_C,box_R)
box2 = np.kron(np.eye(sqrtN),box)

eye3N = np.append(np.eye(N),np.eye(N),axis = 1)
eye3N = np.append(eye3N,np.eye(N),axis = 1)
BOX = sparse.csc_matrix(np.kron(box2,eye3N))


cell = np.eye(N**2)
CELL = sparse.csc_matrix(np.kron(cell,np.ones(N)))

ind = np.where(question > 0)
c = ind[0]
r = ind[1]
v = question[ind]



table = (c)*N + r
table = np.append(table,v,axis = 0)
size = int(table.size/2)

table = np.reshape(table,[2,size])

CLUE = []

for index in range(0,size,1):
    clue = np.zeros(N ** 3)
    clue[table[0][index]*N + table[1][index]-1] = 1
    if index == 0:
        CLUE = clue
    else :
        CLUE = np.vstack([CLUE,clue])

CLUE = sparse.csc_matrix(CLUE)

A = sparse.vstack((ROW,COL,BOX,CELL,CLUE),format = 'csc')
ind = sparse.find(A)
m = len(set(ind[0]))
n = len(set(ind[1]))
b = np.ones(m)
x0 = (1./N) * np.ones(n);


bnds = np.array([0,1])
bnds = np.tile(bnds,(N**3,1))

con1 = {'type': 'ineq', 'fun': constraint1}
solution = minimize(sudoku_obj,x0,method='SLSQP',bounds=bnds,constraints=con1)

y = (solution.x > 0.5)
X = np.reshape(y,[N**2,N])
r = np.where(X == True)
solution = np.reshape(r[1],[N,N])
solution = solution + 1

print("Solution:")
print(solution)
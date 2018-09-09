from __future__ import division
import numpy as np
import sys

#train_data = np.genfromtxt(sys.argv[1], delimiter = ",")
train_data = np.genfromtxt("ratings.csv", delimiter = ",")

lam = 2
sigma2 = 0.1
d = 5

# Implement function here
def PMF(train_data):
    num = 50
    user = train_data[:, 0].max()
    movie = train_data[:, 1].max()

    data = []
    
    for k in np.arange(user+1):
        index = train_data[:, 0] == k
        value = train_data[index][:, 1:]
        key = value[:, 0]
        for i in np.arange(movie+1):
            if i not in key:
                value = np.append(value, np.array([i, -1]).reshape(1,2), axis = 0)
        value = np.array(sorted(value, key = lambda x: x[0]))
        data.append(value[:, 1])
    train_data = np.array(data)
    Index = train_data != -1
    N1, N2 = train_data.shape
    I = np.eye(d)
    U = np.random.random((N1, d))
    V = np.random.random((d, N2))
    Ldata = []
    Udata = []
    Vdata = []
    for _ in range(num):
        for k in range(N1):
            index = train_data[k] != -1
            V1 = V[:, index]
            M = train_data[k, index]
            data = np.sum(M*V1, axis = 1)
            u = np.linalg.inv(lam*sigma2*I + V1.dot(V1.T)).dot(data)
            U[k] = u
        
        for k in range(N2):
            index = train_data[:, k] != -1
            U1 = U[index, :].T
            M = train_data[index, k]
            data = np.sum(M*U1, axis = 1)
            v = np.linalg.inv(lam*sigma2*I + U1.dot(U1.T)).dot(data)
            V[:, k] = v

        L = - 1/(2*sigma2) * np.sum((train_data[Index] - (U.dot(V))[Index])**2) - lam/2 * np.sum(V**2) - lam/2 * np.sum(U**2)
        
        Ldata.append(L)
        Udata.append(U)
        Vdata.append(V.T)
    return Ldata, Udata, Vdata

# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)

L, U_matrices, V_matrices = PMF(train_data)

np.savetxt("objective.csv", L, delimiter=",")
np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")


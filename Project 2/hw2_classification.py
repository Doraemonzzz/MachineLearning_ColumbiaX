from __future__ import division
import numpy as np
from scipy.linalg import det
from scipy.linalg import inv
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")
'''
X_train = np.loadtxt('X_train.csv', delimiter = ",")
y_train = np.loadtxt('y_train.csv', delimiter = ",")
X_test = np.loadtxt('X_test.csv', delimiter = ",")
y_test = np.loadtxt('y_test.csv', delimiter = ",")
'''

## can make more functions if required


def pluginClassifier(X_train, y_train, X_test):    
  # this function returns the required output 
  pass


num = 10
n = X_train.shape[0]
pi = np.array([])
N = np.array([])
for i in range(num):
    Ny = np.sum(y_train==i)
    N = np.append(N,Ny)
    piy = Ny/n
    pi = np.append(pi,piy)
    
mu = []
for i in range(num):
    temp1 = X_train[y_train==i]
    temp2 = temp1.sum(axis=0)/N[i]
    mu.append(temp2)
print(temp1.shape)
print(temp2.shape)
mu = np.array(mu)

sigma = []
de = []
sigma1 = []
for i in range(num):
    temp1 = X_train-mu[i]
    temp2 = temp1[y_train==i]
    sigmay = temp2.T.dot(temp2)/N[i]
#    sigma1.append(sigmay)
    sigmay_new = inv(sigmay)
    dety = np.sqrt(np.abs(det(sigmay_new)))
    sigma.append(sigmay_new)
    de.append(dety)
sigma = np.array(sigma)
de = np.array(de)
#sigma1 = np.array(sigma1)

k1 = pi*de
#k1 = np.log(k1)

result1 = np.array([])
result2 = []
n1 = X_test.shape[0]
for i in range(n1):
    r1 = np.array([])
    x = X_test[i]
    for j in range(num):
        x1 = x-mu[j]
        temp = -1/2*x1.dot(sigma[j]).dot(x1.T)
        temp1 = np.exp(temp)
        r1 = np.append(r1,temp1)
    r1 *= k1
    r2 = r1/np.sum(r1)
    result1 = np.append(result1,np.argmax(r1))
    result2.append(r2) 

final_outputs = np.array(result2)
#final_outputs = pluginClassifier(X_train, y_train, X_test) # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file


#print(np.sum(result1==y_test)/n1)
    

    

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 01:04:28 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import scipy as sp
import sys

#X = np.genfromtxt(sys.argv[1], delimiter = ",")
X = np.genfromtxt("X.csv", delimiter = ",")

def KMeans(data):
	#perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively
    k = 5
    n = 10
    N = data.shape[1]
    centerslist = np.random.random((5,N))
    for i in range(n):
        index = np.array([np.argmin(np.sum((centerslist - x)**2, axis = 1)) for x in X])
        for j in range(k):
            cnt = np.sum(index == j)
            if(cnt>0):
                centerslist[j] = np.mean(X[index == j], axis = 0)
        filename = "centroids-" + str(i+1) + ".csv" #"i" would be each iteration
        np.savetxt(filename, centerslist, delimiter=",")

  
def EMGMM(data):
    k = 5
    n = 10

    N = data.shape[1]
    mu = np.random.random((k, N))
    #sigma = np.random.random((k, N, N))
    sigma = np.array([np.eye(N) for _ in range(k)])
    pi = np.random.uniform(size = k)
    pi = pi/np.sum(pi)
    print(np.sum(pi))
    
    def f(x, mu, sigma):
        mu1 = np.linalg.inv(sigma)
        r1 = (x-mu).T.dot(mu1).dot(x-mu)
        result = np.linalg.det(mu1)*np.exp(-1/2*r1)
        
        return result
    
    for i in range(n):
        Phi = []
        for x in X:
            phi = pi * np.array([f(x, mu[t], sigma[t]) for t in range(k)])
            phi = phi/np.sum(phi)
            Phi.append(phi)
        Phi = np.array(Phi)
        nu = np.sum(Phi, axis=0)
        pi = nu/nu.sum()
        print(np.sum(pi),nu.shape)
        
        mu = Phi.T.dot(X)
        mu /= nu.reshape((-1,1))
        
        for j in range(k):
            xi = X - mu[j]
            diag = np.diag(Phi[:,j])
            s = xi.T.dot(diag).dot(xi)
            s /= nu[j]
            sigma[j] = s
        filename = "pi-" + str(i+1) + ".csv" 
        np.savetxt(filename, pi, delimiter=",") 
        filename = "mu-" + str(i+1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration
    
        for j in range(k): #k is the number of clusters 
            filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, sigma[j], delimiter=",")

KMeans(X)
EMGMM(X)
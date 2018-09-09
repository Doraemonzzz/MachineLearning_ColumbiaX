import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

'''
lambda_input=2
sigma2_input=3
X_train=np.loadtxt('X_train.csv', delimiter = ",")
y_train=np.loadtxt('y_train.csv', delimiter = ",")
X_test=np.loadtxt('X_test.csv', delimiter = ",")
'''

## Solution for Part 1
def part1():
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    n,m=X_train.shape
    m1=(lambda_input*np.eye(m))+X_train.T.dot(X_train)
    result=np.linalg.inv(m1).dot(X_train.T).dot(y_train)
    return result

wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    
    n,m=X_train.shape
    matrix1=np.linalg.inv(lambda_input*np.eye(m)+(X_train.T.dot(X_train))/sigma2_input)
    #mu=sigma2_input*np.linalg.inv(matrix1).dot(X_train.T).dot(y_train)

    sigma=[]
    for i in range(X_test.shape[0]):
        sigma.append((X_test[i,:]).dot(matrix1).dot(X_test[i,:].T))
    sigma=np.array(sigma)
    
    '''
    index=list(map(str,np.argsort(-1*sigma)[:10]))
    result=','.join(index)
    '''
    result=np.argsort(-1*sigma)[:10]

    #start from 1
    return result+1

active = part2()  # Assuming active is returned from the function
name="active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv"

'''
with open(name,'w') as file:
    file.write(active+'\n')
'''

with open(name,'w') as file:
    file.write(','.join([str(i) for i in active]))

#np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file

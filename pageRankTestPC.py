# Athor: stmatengss

import numpy as np
from numpy import dtype

EPS=1e-6
DI=0.25

def getPr(n):
    return np.ones(n,np.float32)/n

def pageRank(n,Matrix):
    pr=getPr(n)
    out_degree=np.empty(n,np.float32)
    for i in range(n):
        for j in range(n):
            out_degree[j]+=Matrix[i,j]
    for i in range(n):
        for j in range(n):
            Matrix[i,j]/=out_degree[j]
    for i in range(10):
        pr=np.dot(Matrix,pr)*(1-DI)+pr*DI
    return pr

n_point=4
Mat=np.array([
                 [0,1,1,0],
                 [1,0,0,1],
                 [1,0,0,1],
                 [1,1,0,0]],
                dtype=np.float32
                )   
pr=pageRank(n_point, Mat)
for a in pr:
    print a    
    
    
    

'''
Created on Dec 7, 2015

@author: stmatengss
'''
# tmp1 = pr/degree
# tmp2 = tmp1 * graph
# new_pr = (1-r) + r * tmp2
# _tmp = new_pr - pr;
# diff = sum(_tmp)
# pr = new_pr

import spartan as sp
import numpy as np
from spartan import expr


sp.initialize()

N_DIM=10
N_EXAMPLES=1000*1000
EPS=1e-6
DI=0.25


def makePr(n):
    return sp.ones(shape=2, dtype=np.float64, tile_hint=None) 

def pageRank(pr,graph): #sp.ones
    ont_mat=sp.rand(N_DIM,1)
    for i in range(50):
        tmp_mat=sp.dot(pr,graph)
        diff=pr*(tmp_mat-DI)
        grad=sp.sum(diff,axis=0).reshape((N_DIM,1))
                
a=expr.ones((2,2),float)
print a,type(a)

c=a.glom()
print c,type(c)

b=expr.eager(a)
print b,type(b)

c=b.glom()
print c,type(c)
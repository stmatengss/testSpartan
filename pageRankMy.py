import scipy.sparse
import spartan as sp
import numpy as np
import myTestCommon
import signal
import atexit
import os
import cProfile
import imp
import sys
import time
import types
import unittest

from os.path import basename, splitext
from spartan import expr, util, eager, array, config
from spartan.cluster import start_cluster
from spartan.config import FLAGS, StrFlag, BoolFlag
import test_common
from numpy import Inf, dtype
from sys import getsizeof
from Crypto.Random.random import shuffle

OUTLINKS_PER_PAGE = 10
PAGES_PER_WORKER = 1000 * 5
EPS=1e-8
DI=0.75
CO=10
N_DIM = 10
N_EXAMPLES = 1000 * 1000
count=0
N=80

FLAGS.add(StrFlag('worker_list', default='1,2,4,8,16'))
FLAGS.add(BoolFlag('test_optimizations', default=False))
file_out=open('data.out','w')

class BenchTimer(object):
  def __init__(self, num_workers):
    self.num_workers = num_workers
    self.prefix = ''

  def time_op(self, key, fn):
    st = time.time()
    result = fn()
    ed = time.time()
    self.log('%d,"%s",%f', self.num_workers, key, ed - st)
    return result

  def benchmark_op(self, op, min_time=1.0):
    '''Run ``op`` in a loop until ``min_time`` has passed.'''
    iters = 0
    from time import time
    st = time()

    while 1:
        iters += 1
        op()
        ed = time()
        if ed - st > min_time:
            break

    self.log('Ran %d ops, %.3f seconds, %f s/op, %f ops/s' % (iters, ed - st, (ed - st) / iters, iters / (ed - st)))


  def log(self, fmt, *args):
    msg = fmt % args if len(args) > 0 else fmt
    print self.prefix, msg

def readfile():
    pass

def record(x):
    print x
    file_out.write(str(x))

def PageRank(ntime):
#     mat=np.array([[0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1],
#                  [1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1],
#                  [0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,0,1],
#                  [1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0],
#                  [1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1],
#                  [0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1],
#                  [1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1],
#                  [0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,0,1],
#                  [1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0],
#                  [0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,0,1],
#                  [1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0],
#                  [0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1],
#                  [1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1],
#                  [0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,0,1],
#                  [1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0],
#                  [1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0],
#                  [1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0],
#                  [0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1],
#                  [1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,1],
#                  [0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,0,1]],dtype=float)
    source = np.random.randint(0, N, N)
    dest = np.random.randint(0, N, N)
    value = np.random.rand(N).astype(float)
    mat=scipy.sparse.coo_matrix((value,(source,dest)),shape=(N,N))  
#     print getsizeof(mat.data)
#     mat=mat.todense()
#     print getsizeof(mat.data)
    ######
    diff_sum=Inf
    n_point=mat.shape[0]
    pr=sp.ones((n_point,1),dtype=float)
    pr=pr/n_point
    
#     for i in range(n_point):
#         cal=0
#         for j in range(n_point):
#             cal+=mat[j,i]
#         for j in range(n_point):
#             mat[j,i]/=cal
    #record(mat)
    mat=expr.from_numpy(mat,tile_hint=(n_point,n_point/ntime))
    mat=eager(mat)
    #cal the time
    for cal_times in range(10):
        begin_time=time.time()
        new_pr=(expr.dot(mat,pr))*DI+pr*(1-DI)
        diff=sp.sub(new_pr, pr)
        diff_sum=sp.sum(diff)
        pr=new_pr
        res=pr.evaluate()
        end_time=time.time()
        record("It's the "+str(cal_times)+" times:"+str(end_time-begin_time)+"s\n")
    record(str(res)+"\n")

def debug(x):
    global count
    count=count+1
    print "##",count
    print type(x)
    print x.evaluate()
    print x.glom()

#Mat=expr.ndarray((4,4),dtype=np.float32)   
#Mat=expr.from_numpy(buff)

def run_benchmarks(module, benchmarks, master, timer):
  for benchname in benchmarks:
    getattr(module, benchname)(master, timer)

def run():
    workers = [1,2,4,8,16]
    sp.config.parse(sys.argv)
    for i in workers:
      # restart the cluster
      record("There are "+str(i)+" workers\n")
      FLAGS.num_workers = i
      sp.initialize()
      timer = BenchTimer(i)
      util.log_info('Running benchmarks on %d workers', i)
      PageRank(i)
      sp.shutdown()

run()








    
    


from spartan.expr import eager,sqrt, exp, norm_cdf, eager, log, abs, mean
from spartan import util
from spartan import expr
from spartan.expr import eager, ones, zeros, glom, evaluate, randn, from_numpy
from spartan.config import FLAGS
import scipy.sparse
import spartan as sp
import numpy as np
import math
import sys
from StringIO import StringIO
from datetime import datetime
import myTestCommon
from numpy import dtype, NaN
import spartan

EPS=0.01
MAXDIS=1000

def getxy(num):
#     print "#",num
#     if num==NaN:return 0,0
    return num-int(num)/MAXDIS*MAXDIS,int(num)/MAXDIS

def setxy(x,y):
    return x*MAXDIS+y

def dis(x,y,xx,yy):
    return math.sqrt((x-xx)*(x-xx)+(y-yy)*(y-yy))

def f2_id(pxv,idv,pyv,i,pz):
#     pz=pz.evaluate().glom()
    res=np.zeros(pxv.shape,int)
    count=0
    pz_x,pz_y=getxy(pz)
    for px,id,py in zip(pxv,idv,pyv):
        px_x,px_y=getxy(px)
        py_x,py_y=getxy(py)
        d=dis(px_x, px_y, py_x, py_y)
        d2=dis(px_x, px_y, pz_x, pz_y)
        if d>d2: res[count]=i
        else: res[count]=id
        count=count+1
    return res

def f2_py(pxv,idv,pyv,i,pz):
#     pz=pz.evaluate().glom()
    res=np.zeros(pxv.shape,float)
    count=0
    pz_x,pz_y=getxy(pz)
    for px,id,py in zip(pxv,idv,pyv):
        px_x,px_y=getxy(px)
        py_x,py_y=getxy(py)
        d=dis(px_x, px_y, py_x, py_y)
        d2=dis(px_x, px_y, pz_x, pz_y)
        if d>d2: res[count]=setxy(pz_x, pz_y)
        else: res[count]=setxy(py_x, py_y)
        count=count+1
    return res

def f3_cnt(pxv,idv,pyv,i):
    res=np.zeros(pxv.shape,float)
    count=0
    for px,id,py in zip(pxv,idv,pyv):
        if id==i: res[count]=1
        else: res[count]=0
        count=count+1
    return res  

def f3_sum(pxv,idv,pyv,i):
    res=np.zeros(pxv.shape,float)
    count=0
    for px,id,py in zip(pxv,idv,pyv):
        if id==i: res[count]=px
        else: res[count]=0
        count=count+1
    return res    

def kmeans(N,K,means,v):
    changed=True
    count=0
    means=means.evaluate().glom()
#     while changed:
    for run_times in range(4):
        changed=False
        time_begin=datetime.now()
        nearestid=sp.zeros((N,1), dtype=int)
        nearestpoint=sp.ones((N,1), dtype=float)*means[0]
        for i in range(1,K):
            nearestpoint=expr.map([v,nearestid,nearestpoint],f2_py,fn_kw={"i":i,"pz":means[i]})
            nearestid=expr.map([v,nearestid,nearestpoint],f2_id,fn_kw={"i":i,"pz":means[i]})
#             print nearestpoint
#             print nearestid
        for i in range(K):
            cnt=expr.sum(expr.map([v,nearestid,nearestpoint],f3_cnt,fn_kw={"i":i}))
            sum=expr.sum(expr.map([v,nearestid,nearestpoint],f3_sum,fn_kw={"i":i}))
            if cnt==0:cnt=1
            ki=(sum/cnt).evaluate().glom()
            kii=means[i]
#             print kii
            if abs(kii-ki)>EPS:
                changed=True
                means[i]=ki
        time_end=datetime.now()
        count=count+1
        print "Round is ",count," Time is ",(time_end-time_begin)

def benchmark_kmeans(ctx, timer):
    N=160*1000
    K=5
    count=0
    while count<4: 
        means=sp.from_numpy(np.random.random((K,1))*MAXDIS*MAXDIS,tile_hint=(K,1))
        v=sp.from_numpy(np.random.random((N,1))*MAXDIS*MAXDIS,tile_hint=(N/ctx.num_workers,1),)
        means=means.evaluate()
	v=v.evaluate()
	print "OVER!!!"
	time_begin=datetime.now()
        kmeans(N, K, means, v)
        time_end=datetime.now()
        print "When K is ",K,",N is ",N,",number is ",ctx.num_workers,",Total time is ",time_end-time_begin
        K=K*2
        count=count+1

def main(argv):  
  ip=["192.168.1.55","192.168.1.56","192.168.1.57","192.168.1.58","192.168.1.60","192.168.1.61","192.168.1.59","192.168.1.54"]
  s="--hosts="
  fi=open('data','w')
  #for n in argv:
  #  print n
  i=int(argv[1])
  if i<=8:
    for j in range(i):
      if j!=(i-1):
        s=s+ip[j]+":1,"
      else:
        s=s+ip[j]+":1" 
  elif i<=16:
    for j in range(i/2):
      if j!=(i/2-1):
        s=s+ip[j]+":2,"
      else:
        s=s+ip[j]+":2"
  elif i<24:
    for j in range(i/3):
      if j!=(i/3-1):
	s=s+ip[j]+":3,"
      else:
        s=s+ip[j]+":3"
  else:
    for j in range(i/4):
      if j!=(i/4-1):
	s=s+ip[j]+":4,"
      else:
        s=s+ip[j]+":4"
  print s 
  ctx=sp.initialize(['--cluster=1','--num_workers='+str(i),s])    
  benchmark_kmeans(ctx,fi)
  fi.close()
  sp.shutdown()    

if __name__ == '__main__':
  main(sys.argv)

#if __name__ == '__main__':
#    myTestCommon.run(__file__) 
# spartan.initialize()                
# kmeans(100,3,sp.from_numpy(np.random.random((3,1))*MAXDIS*MAXDIS),
#        sp.from_numpy(np.random.random((100,1))*MAXDIS*MAXDIS))




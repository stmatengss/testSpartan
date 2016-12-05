import scipy.sparse
import numpy as np
from StringIO import StringIO
from datetime import datetime
from spartan import expr, util, eager,FLAGS
#import test_common
#import myTestCommon
from locale import str
import spartan
import sys

OUTLINKS_PER_PAGE = 10
PAGES_PER_WORKER = 100 * 5
NODES= 875713
DIVS=128
LEN=NODES/DIVS
DI=0.8

def readfile(filename):
    a=open(filename,'r')
    txt=""
    for line in a:
        txt+=line   
    res=np.genfromtxt(StringIO(txt), dtype=np.int32, comments="#",delimiter="\t")
    res.sort(axis=0)
    return res

def calvalue():
    res=readfile("web-Google(dealing).txt")
#     res=readfile("test.txt")
    #print res
    pro=-1
    degree=np.ones((NODES,1),dtype=np.float32)
    store=[[] for i in range(NODES)]
    value=[]
    count=0
    t=0
    for a,b in res:
#         if store.has_key(b):
#             store[b].append(a)
#         else:
#             store[b]=[a]
        #print a,pro
#         store[b].append(a)
        if a!=pro and pro!=-1:
            value+=[1.0/count]*count
            degree[pro]=1.0*count
            count=1
        else:
            count=count+1
        pro=a
        t=t+1
    value+=[1.0/count]*count
    degree[pro]=count
#     return store,degree
#     source=[[] for i in range(DIVS+1)]
#     dest=[[] for i in range(DIVS+1)]
#     value=[[] for i in range(DIVS+1)]
#     for x,y in res:
#         pos=y/LEN
#         source[pos].append(x)
#         dest[pos].append(y%LEN)
#         value[pos].append(1.0)
    source=[x for x,y in res]
    dest=[y for x,y in res]
#     value=[1. for x,y in res]
    return source,dest,value,degree

source,dest,value,degree=calvalue()
# store,degree=calvalue()

def make_weights(tile, ex):
#     print type(ex)
#     print ex,ex.get_lr(),ex.get_ul()
    #print ex.shape
#     num_source = ex.shape[1]
#     num_dest = ex.shape[0]
#     
#     num_out = num_source * OUTLINKS_PER_PAGE
#     
#     util.log_info('%s %s %s %s', ex, num_source, num_dest, num_out)
#     
#     source = np.random.randint(0, ex.shape[0], num_out)
#     dest = np.random.randint(0, ex.shape[1], num_out)
#     value = np.random.rand(num_out).astype(np.float32)
    
    #util.log_info('%s %s %s', source.shape, dest.shape, value.shape)
    global dest,source,value
    dest_=[]
    source_=[]
    value_=[]
#     print ex
#     print ex.get_ul()[1],ex.get_lr()[1]
    lb=ex.get_ul()[1]
    ub=ex.get_lr()[1]
#     print lb/LEN,ub/LEN
    count=0
#     for y in store[ex.get_ul()[1]:ex.get_lr()[1]]:
#         if count%500==0:print count
#         dest_=dest_+([count]*len(y))
#         source_=source_+(y)
#         value_=value_+([1.0]*len(y))
#         count=count+1
#     print 'ok!!!!'
#########################
#     time_begin=datetime.now()
#     for x,y in zip(source,dest):
#         if lb<=y<ub:
#             source_.append(x)
#             dest_.append(y-lb)
#             value_.append(1.0)
#     time_end=datetime.now()
#     print "#1 ",time_end-time_begin," s"
#########################
    for i in range(lb/LEN,ub/LEN):
        source_=source_+source[i]
        dest_=dest_+dest[i]
        value_=value_+value[i]
    #print source_,dest_,value_    
    data = scipy.sparse.coo_matrix((value_, (source_, dest_)), shape=ex.shape)
    #print data
    return [(ex, data)]


def benchmark_pagerank(ctx, timer): 
    #num_pages = PAGES_PER_WORKER * ctx.num_workers
    global degree,value,dest,source
    num_pages=NODES
    PAGES_PER_WORKER=num_pages/ctx.num_workers
    util.log_info('Total pages: %s', num_pages)
#     wts = eager(
#       expr.shuffle(
#         expr.ndarray(
#           (num_pages, num_pages),
#           dtype=np.float32,
#           tile_hint=(num_pages, PAGES_PER_WORKER/8 )),
#         make_weights,
#       ))
#     print value,source,dest
    data=scipy.sparse.coo_matrix((value, (source, dest)), shape=(NODES,NODES))
    wts=eager(expr.from_numpy(data,tile_hint=(num_pages,PAGES_PER_WORKER)))
    #print wts.evaluate()
    #print wts.glom()
    ones = eager(expr.ones((num_pages, 1),
                        tile_hint=(PAGES_PER_WORKER/8 , 1),
                        dtype=np.float32))
    p=ones
    for i in range(4):
        time_begin=datetime.now()
#         print degree    
        tmp1=eager(p/degree)    #3
#          print tmp1
        #print tmp1.evaluate().glom()
        tmp2=eager(expr.dot(wts, tmp1))   #4
#         print tmp2
        #print tmp2.evaluate().glom()

        new_p=(DI*tmp2+(1-DI)*ones)  #5
        diff=eager(p-new_p).glom()
        p=new_p  #7
        p.evaluate().glom()  
        time_end=datetime.now()
        print "opt-enabled time is ",time_end-time_begin," s"
#         timer.time_op('pagerank', lambda: (DI*expr.dot(wts, p).optimized()+DI*p).evaluate())

#ctx=spartan.initialize()
# FLAGS.optimization = 1
def main(argv):  
  ip=["192.168.1.55","192.168.1.56","192.168.1.57","192.168.1.58","192.168.1.60","192.168.1.61","192.168.1.54","192.168.1.59"]
  s="--hosts="
  #fi=open('data','w')
  #for n in argv:
  #  print n\
  fi=1	
  i=int(argv[1])
  print "####"
  if i<=8:
    for j in range(i):
      if j!=(i-1):
        s=s+ip[j]+":1,"
      else:
        s=s+ip[j]+":1" 
  else :
    for j in range(i/2):
      if j!=(i/2-1):
        s=s+ip[j]+":2,"
      else:
        s=s+ip[j]+":2"
  print s 
  ctx=spartan.initialize(['--cluster=1','--num_workers='+str(i),s])    
  benchmark_pagerank(ctx,fi)
  #fi.close()
  spartan.shutdown()    
#benchmark_pagerank(ctx, timer=1)
if __name__ == '__main__':
  main(sys.argv)
#     myTestCommon.run(__file__) 

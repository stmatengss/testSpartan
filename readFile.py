import numpy as np
from StringIO import StringIO
import scipy.sparse
from numpy import dtype
from cups import Dest

NODES=875713

#a=open('web-Google.txt','r')
def readfile(filename):
    a=open(filename,'r')
    txt=""
    for line in a:
        txt+=line
    
    res=np.genfromtxt(StringIO(txt), dtype=np.int32, comments="#",delimiter="\t")
    res.sort(axis=0)
    return res

def calnodes():
    res=readfile("web-Google.txt")
    f=open("web-Google(dealing).txt",'w')
    count=0
    mp=dict()
    for a,b in res:
        if mp.get(a)==None:
            mp[a]=count
            count=count+1 
        if mp.get(b)==None:
            mp[b]=count
            count=count+1
        f.write(str(mp[a])+"\t"+str(mp[b])+"\n`1")
    f.close()

def calvalue():
    res=readfile("web-Google(dealing).txt")
    #print res
    pro=-1
    value=[]
    degree=np.zeros((NODES,1),dtype=np.float32)
    count=0
    print res
    for a,b in res:
        #print a,pro
        if a!=pro and pro!=-1:
            value+=[1.0/count]*count
            degree[pro,0]=count
            count=1
        else:
            count=count+1
        pro=a
    value+=[1.0/count]*count
    degree[pro]=count
    #print value
    source=[x for x,y in res]
    dest=[y for x,y in res]
    
    return source,dest,value,degree

print 'succees'


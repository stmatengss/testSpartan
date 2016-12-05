from spartan.expr import eager,sqrt, exp, norm_cdf, eager, log, abs, mean
from spartan import util
import spartan as sp
import scipy.sparse
from spartan.expr import eager, ones, zeros, glom, evaluate, randn, from_numpy
from spartan.config import FLAGS
import numpy as np
from StringIO import StringIO
from datetime import datetime
import myTestCommon
import sys

DATA_SIZE= 480*1000*1000
#S=current
#K=strike

def black_scholes(current, strike, maturity, rate, volatility):
    d1 = 1.0 / (volatility * sqrt(maturity)) * (
      log(current / strike) + (rate + volatility ** 2 / 2) * (maturity)
    )  #1
    d2 = d1 - volatility * sqrt(maturity)  #2
    tmp1 = norm_cdf(d2) * strike * exp(-rate * maturity) #4
    call = norm_cdf(d1) * current -  tmp1    #3
    call.evaluate().glom()
    tmp2 = norm_cdf(-d2) * strike * exp(-rate * maturity) - current  #5
    put = tmp2 + call   #5
    put.evaluate().glom() 
    
    
def benchmark_blackscholes(ctx, fi): 
#     FLAGS.optimization = 0
    #num_pages = PAGES_PER_WORKER * ctx.num_workers
    SCLICES=1	
    for i in range(0,4):
      current = eager(zeros((DATA_SIZE ,1),
      dtype=np.float32, tile_hint=(DATA_SIZE/ctx.num_workers/SCLICES,1)))
#     current = eager(randn((DATA_SIZE ,1)))
      strike = eager(ones((DATA_SIZE,1),
      dtype=np.float32, tile_hint=(DATA_SIZE/ctx.num_workers/SCLICES,1)))
      maturity = eager(strike * 12)
      rate = eager(strike * 0.05)
      volatility = eager(strike * 0.01)
    #
      FLAGS.optimization = 0
      time_begin=datetime.now()
      black_scholes(current, strike, maturity, rate, volatility)
      time_end=datetime.now()
      fi.write("Slices is "+str(SCLICES)+",Number is "+str(ctx.num_workers)+",Time is "+str(time_end-time_begin))
      print "Slices is ", SCLICES,",Number is ",ctx.num_workers,",Time is ",time_end-time_begin
      
      #FLAGS.optimization = 1
# #     FLAGS.opt_parakeet_gen = 0
#     FLAGS.opt_map_fusion = 1
      #time_begin=datetime.now()
      #black_scholes(current, strike, maturity, rate, volatility)
      #time_end=datetime.now()
      #print "Time is ",time_end-time_begin
#  	
      SCLICES=SCLICES*2	
    #
#     FLAGS.optimization = 1
# #     FLAGS.opt_parakeet_gen = 0
#     FLAGS.opt_map_fusion = 1
#     time_begin=datetime.now()
#     black_scholes(current, strike, maturity, rate, volatility)
#     time_end=datetime.now()
#     print "Time is ",time_end-time_begin
#     #
# #     FLAGS.opt_parakeet_gen = 0
#     FLAGS.opt_auto_tiling = 0
#     time_begin=datetime.now()
#     black_scholes(current, strike, maturity, rate, volatility)
#     time_end=datetime.now()
#     print "Time is ",time_end-time_begin
#     #
# #     FLAGS.opt_parakeet_gen = 1
#     time_begin=datetime.now()
#     black_scholes(current, strike, maturity, rate, volatility)
#     time_end=datetime.now()
#     print "Time is ",time_end-time_begin

def main(argv):  
  ip=["192.168.1.54","192.168.1.55","192.168.1.56","192.168.1.57","192.168.1.58","192.168.1.60","192.168.1.61","192.168.1.59"]
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
  benchmark_blackscholes(ctx,fi)
  fi.close()
  sp.shutdown()    

if __name__ == '__main__':
  main(sys.argv)
 #   myTestCommon.run(__file__) 
    
    

    

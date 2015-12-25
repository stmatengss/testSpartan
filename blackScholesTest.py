from spartan.expr import eager,sqrt, exp, norm_cdf, eager, log, abs, mean
from spartan import util
import scipy.sparse
from spartan.expr import eager, ones, zeros, glom, evaluate, randn, from_numpy
from spartan.config import FLAGS
import numpy as np
from StringIO import StringIO
from datetime import datetime
import myTestCommon


DATA_SIZE= 1600*1000
#S=current
#K=strike

def black_scholes(current, strike, maturity, rate, volatility):
    d1 = 1.0 / (volatility * sqrt(maturity)) * (
      log(current / strike) + (rate + volatility ** 2 / 2) * (maturity)
    )  #1
    d2 = d1 - volatility * sqrt(maturity)  #2
    tmp1 = norm_cdf(d2) * strike * exp(-rate * maturity) #4
    call = norm_cdf(d1) * current -  tmp1    #3
    print call.evaluate().glom()
    tmp2 = norm_cdf(-d2) * strike * exp(-rate * maturity) - current  #5
    put = tmp2 + call   #5
    print put.evaluate().glom() 
    
    
def benchmark_blackscholes(ctx, timer): 
#     FLAGS.optimization = 0
    #num_pages = PAGES_PER_WORKER * ctx.num_workers
    current = eager(zeros((DATA_SIZE ,1),
    dtype=np.float32, tile_hint=(DATA_SIZE/ctx.num_workers,1)))
#     current = eager(randn((DATA_SIZE ,1)))
    strike = eager(ones((DATA_SIZE,1),
    dtype=np.float32, tile_hint=(DATA_SIZE/ctx.num_workers,1)))
    maturity = eager(strike * 12)
    rate = eager(strike * 0.05)
    volatility = eager(strike * 0.01)
    #
    time_begin=datetime.now()
    black_scholes(current, strike, maturity, rate, volatility)
    time_end=datetime.now()
    print "Time is ",time_end-time_begin
    #
    FLAGS.optimization = 1
#     FLAGS.opt_parakeet_gen = 0
    FLAGS.opt_map_fusion = 1
    time_begin=datetime.now()
    black_scholes(current, strike, maturity, rate, volatility)
    time_end=datetime.now()
    print "Time is ",time_end-time_begin
    #
#     FLAGS.opt_parakeet_gen = 0
    FLAGS.opt_auto_tiling = 0
    time_begin=datetime.now()
    black_scholes(current, strike, maturity, rate, volatility)
    time_end=datetime.now()
    print "Time is ",time_end-time_begin
    #
#     FLAGS.opt_parakeet_gen = 1
    time_begin=datetime.now()
    black_scholes(current, strike, maturity, rate, volatility)
    time_end=datetime.now()
    print "Time is ",time_end-time_begin
    
    
if __name__ == '__main__':
    myTestCommon.run(__file__) 
    
    

    
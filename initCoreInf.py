import scipy.sparse
import spartan as sp
import numpy as np
import test_common

from spartan import expr, util, eager, array, master
from numpy import Inf, dtype
from spartan.master import MASTER

sp.initialize()

a=sp.master.get()

print a.get_available_workers()
print a.get_worker_scores()



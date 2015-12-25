import spartan as sp
sp.initialize()

N_DIM = 10
N_EXAMPLES = 1000 * 1000
EPSILON = 1e-6
count=0

def debug(x):
    global count
    count=count+1
    print "##",count
    print type(x)
    print x.evaluate()
    print x.glom()

x = 100 * sp.ones((N_EXAMPLES, N_DIM)) + sp.rand(N_EXAMPLES, N_DIM)
y = sp.ones((N_EXAMPLES, 1))
debug(x)
debug(y)

# put weights on one server
w = sp.rand(N_DIM, 1)
debug(w)

for i in range(50):
    yp = sp.dot(x, w)
    debug(yp)
    diff = x * (yp - y)
    debug(diff)
    grad = sp.sum(diff, axis=0).reshape((N_DIM, 1))
    debug(grad)
    w = w - (grad / N_EXAMPLES * EPSILON)

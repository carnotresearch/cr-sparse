import time
from cr.sparse.dict import simple
from cr.sparse.data.synthetic.random import SparseRepGenerator
from cr.sparse.pursuit.mp import MatchingPursuit
import tensorflow as tf

# Signal dimension
N = 1024
# Number of atoms
D = 2*N
# Sparsity level
K = N//32
# Number of signals
S = 160

gen = SparseRepGenerator(D, K, S)

tf.random.set_seed(1234)

# Dictionary
dict = simple.gaussian_mtx(N, D)


# Matching pursuit
mp = MatchingPursuit(dict)

start_time = time.time()
tf.random.set_seed(1234)
# Sparse Representation
representation = gen.gaussian()
# Signal
signal = tf.squeeze(representation @ dict)
# solve it
solution = mp(signal)
print("\n[%d] iters, --- time: %s seconds ---" % ( 
    solution.iterations, 
    time.time() - start_time))

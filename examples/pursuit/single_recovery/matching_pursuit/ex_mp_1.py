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
S = 1

gen = SparseRepGenerator(D, K, S)

tf.random.set_seed(1234)

# Dictionary
dict = simple.gaussian_mtx(N, D)


# Matching pursuit
mp = MatchingPursuit(dict)

start_time = time.time()
for i in range(20):
    tf.random.set_seed(i)
    # Sparse Representation
    representation = gen.gaussian()
    # Signal
    signal = tf.squeeze(representation @ dict)
    # solve it
    solution = mp(signal)
    print("[%d] [%d] iters  res_norm: %s, --- time: %s seconds ---" % (i+1, solution.iterations, solution.residual_norm.numpy(), time.time() - start_time))
import time
from jax import random
from cr.sparse.dict import simple
from cr.sparse.data.synthetic import sparse_normal_representations
from cr.sparse.pursuit.mp import solve_smv, solve_mmv

# Signal dimension
N = 1024
# Number of atoms
D = 2*N
# Sparsity level
K = N//32
# Number of signals
S = 64

key = random.PRNGKey(0)
# Dictionary
dictionary = simple.gaussian_mtx(key, N, D)
# Sparse Representation
representations, omega = sparse_normal_representations(key, D, K, S)

# Signal
signals = representations @ dictionary

# Matching pursuit
start_time = time.time()
for s in range(S):
    # solve it
    start = time.time()
    sol = solve_smv(dictionary, signals[s])
    duration = time.time() - start
    print(f"\n{sol.iterations} iters, time: {duration:.2f} sec, res norm: {sol.residual_norms:.2e}")
end_time = time.time()
total_duration = end_time - start_time
print(f'Problem size: D={D}, N={N}, K={K}, S={S}, Dict: {dictionary.shape}, Representations: {representations.shape}, Signals: {signals.shape}')
print(f'Total duration: {total_duration:.2f} sec')
print(f'Average duration: {total_duration/S:.2f} sec')
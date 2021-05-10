from jax import random
from cr.sparse.dict import simple
from cr.sparse.norm import norms_l2_rw
from cr.sparse.data.synthetic import sparse_normal_representations

D = 10
N = 6
K = 3
S = 6


key = random.PRNGKey(0)
dictionary = simple.gaussian_mtx(key, N, D)
print("Sample Dictionary:")
print(dictionary)
print (f"Dictionary shape: {dictionary.shape}")
print("Dictionary row wise norms:")
print (norms_l2_rw(dictionary))


representations, omega = sparse_normal_representations(key, D, K, S)

print("Sample Representations: ")
print(representations)
print(f"Rep shape: {representations.shape}")
print(f"Non zero indices: {omega}")

signals = representations @ dictionary


print("Sample Signals: ")
print(signals)
print(f"Signals shape {signals.shape}")

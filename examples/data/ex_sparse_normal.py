from jax import random
from cr.sparse.data.synthetic import sparse_normal_representations

D = 6
K = 2
S = 4

key = random.PRNGKey(1)

data, omega = sparse_normal_representations(key, D, K, S)
data = data.block_until_ready()
print(data)
print(omega)

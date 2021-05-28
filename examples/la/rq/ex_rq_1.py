import jax.numpy as jnp

from cr.sparse.la import rq

A = jnp.array([1, 0, 1, 1, 0, -1, 1, 1, 0], dtype=jnp.float32)
A = A.reshape([3,3])
print(A)

R, Q = rq.factor_mgs(A)

print("Q:")
print(Q)
print("R:")
print(R)

print ("Q Q'\n", Q @ Q.T)
#print ("Q' Q\n", Q.T @ Q)

I = jnp.eye(A.shape[0])

print("R Q\n", R @ Q)

print("A - R Q\n", A - R @ Q)

print ("max(abs(Q Q' - I)):", jnp.max(jnp.abs(Q @ Q.T - I)) )
print("max(abs(A - RQ))", jnp.max(jnp.abs(A - R @ Q)))
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

n = 400
A = jnp.ones((n, n))
x = jnp.ones(n)

import timeit

count = 1000
result  = timeit.timeit(lambda: A @ x, number=count)
print (f'{result*1000*1000/count:.3f} us')

import jax

result  = timeit.timeit(jax.jit(lambda: A @ x), number=count)
print (f'{result*1000*1000/count:.3f} us')

from typing import NamedTuple,Callable

class LinearOperator(NamedTuple):
    times : Callable[[jnp.ndarray], jnp.ndarray]
    trans : Callable[[jnp.ndarray], jnp.ndarray]
    m : int
    n : int

def matrix_operator(A):
    m, n = A.shape
    times = lambda x: A @ x
    trans = lambda x : (x.T @ A ).T
    return LinearOperator(times=times, trans=trans, m=m, n=n)

A_operator = matrix_operator(A)

result  = timeit.timeit(lambda: A_operator.times(x), number=count)
print (f'{result*1000*1000/count:.3f} us')


def jit_it(operator):
    times = jax.jit(operator.times)
    trans = jax.jit(operator.trans)
    return LinearOperator(times=times, trans=trans, m=operator.m, n=operator.n)

A_operator = jit_it(A_operator)
A_operator.times(x)

result  = timeit.timeit(lambda: A_operator.times(x).block_until_ready(), number=count)
print (f'{result*1000*1000/count:.3f} us')


def f(A, x, n):
    def init():
        return (jnp.zeros(x.shape), 0)

    def iter(state):
        result, c = state
        return (result + A @ x, c+1)

    def cond(state):
        result, c = state
        return c < n

    state = jax.lax.while_loop(cond, iter, init())
    return state[0]

count = 10
result  = timeit.timeit(lambda:f(A, x, 50).block_until_ready(), number=count)
print (f'f: {result*1000/count:.3f} ms')

f_jit = jax.jit(f, static_argnums=(2,))
f_jit(A, x, 50)

result  = timeit.timeit(lambda:f_jit(A, x, 50).block_until_ready(), number=count)
print (f'f_git: {result*1000/count:.3f} ms')



def g(operator, x, n):
    times = operator.times
    def init():
        return (jnp.zeros(x.shape), 0)

    def iter(state):
        result, c = state
        return (result + times(x), c+1)

    def cond(state):
        result, c = state
        return c < n

    state = jax.lax.while_loop(cond, iter, init())
    return state[0]


count = 10
result  = timeit.timeit(lambda:g(A_operator, x, 50).block_until_ready(), number=count)
print (f'g: {result*1000/count:.3f} ms')

g_jit = jax.jit(g, static_argnums=(0, 2,))
g_jit(A_operator, x, 50)

result  = timeit.timeit(lambda:g_jit(A_operator, x, 50).block_until_ready(), number=count)
print (f'g_jit: {result*1000/count:.3f} ms')




def ft(A, x, n):
    def init():
        return (jnp.zeros(x.shape), 0)

    def iter(state):
        result, c = state
        return (result + A.T @ x, c+1)

    def cond(state):
        result, c = state
        return c < n

    state = jax.lax.while_loop(cond, iter, init())
    return state[0]

count = 10
result  = timeit.timeit(lambda:ft(A, x, 50).block_until_ready(), number=count)
print (f'ft: {result*1000/count:.3f} ms')

ft_jit = jax.jit(ft, static_argnums=(2,))
ft_jit(A, x, 50)

result  = timeit.timeit(lambda:ft_jit(A, x, 50).block_until_ready(), number=count)
print (f'ft_jit: {result*1000/count:.3f} ms')



def gt(operator, x, n):
    trans = operator.trans
    def init():
        return (jnp.zeros(x.shape), 0)

    def iter(state):
        result, c = state
        return (result + trans(x), c+1)

    def cond(state):
        result, c = state
        return c < n

    state = jax.lax.while_loop(cond, iter, init())
    return state[0]


count = 10
result  = timeit.timeit(lambda:gt(A_operator, x, 50).block_until_ready(), number=count)
print (f'gt: {result*1000/count:.3f} ms')

gt_jit = jax.jit(gt, static_argnums=(0, 2,))
gt_jit(A_operator, x, 50)

result  = timeit.timeit(lambda:gt_jit(A_operator, x, 50).block_until_ready(), number=count)
print (f'gt_jit: {result*1000/count:.3f} ms')



def power_iteration(
    A,
    num_iters=100,
    error_tolerance=1e-6):
  n = A.shape[-1]

  def cond(state):
    i, unused_v, unused_s, unused_s_v, run_step = state
    return jnp.logical_and(i < num_iters, run_step)

  def body(state):
    i, new_v, s, s_v, unused_run_step = state
    new_v = new_v / jnp.linalg.norm(new_v)
    s_v = A @ new_v
    s_new = new_v @ s_v
    return (i + 1, s_v, s_new, s_v,
            jnp.greater(jnp.abs(s_new - s), error_tolerance))

  # Figure out how to use step as seed for random.
  v_0 = np.random.uniform(-1.0, 1.0, n)

  init_state = tuple([0, v_0, jnp.zeros([]), v_0, True])

  _, v_out, s_out, _, _ = lax.while_loop(cond, body, init_state)
  
  v_out = v_out / jnp.linalg.norm(v_out)
  
  return v_out, s_out

v_out, s_out = power_iteration(A)
print(v_out)
print(s_out)
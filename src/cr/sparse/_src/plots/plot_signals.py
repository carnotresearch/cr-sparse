import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt



def plot_sparse_signal(ax, x):
    if x.ndim == 1:
        t, = jnp.nonzero(x)
        ax.stem(t, x[t], markerfmt='.b', linefmt='k')
        return
    else:
        s = x.shape[1]
        for i in range(s):
            xx = x[:, i]
            t, = jnp.nonzero(xx)
            ax.stem(t, xx[t], markerfmt='.b', linefmt='k', label=f'{i+1}')
        ax.legend()


def plot_sparse_signals(ax, ref, rec):
    t, = jnp.nonzero(ref)
    ax[0].stem(t, ref[t], markerfmt='.b', linefmt='k')
    t, = jnp.nonzero(rec)
    ax[1].stem(t, rec[t], markerfmt='.b', linefmt='k')

def plot_sparse_n_signals(ax, x):
    if x.ndim == 1:
        return plot_sparse_signal(ax, x)
    for i in range(x.shape[1]):
        xx = x[:, i]
        t, = jnp.nonzero(xx)
        ax[i].stem(t, xx[t], markerfmt='.b', linefmt='k')

def plot_signal(ax, x):
    x = np.asarray(x)
    ax.plot(x)


def plot_n_signals(ax, x):
    # Assume that each row is a signal
    s, n = x.shape
    for i in range(s):
        ax[i].plot(x[i, :])
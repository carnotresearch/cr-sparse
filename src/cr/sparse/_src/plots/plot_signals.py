import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt



def plot_sparse_signal(ax, x):
    t, = jnp.nonzero(x)
    ax.stem(t, x[t], markerfmt='.b', linefmt='k')


def plot_sparse_signals(ax, ref, rec):
    t, = jnp.nonzero(ref)
    ax[0].stem(t, ref[t], markerfmt='.b', linefmt='k')
    t, = jnp.nonzero(rec)
    ax[1].stem(t, rec[t], markerfmt='.b', linefmt='k')


def plot_signal(ax, x):
    x = np.asarray(x)
    ax.plot(x)

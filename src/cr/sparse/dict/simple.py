import math
import tensorflow as tf
from cr.sparse.norm import normalize_l2_rw

def gaussian_mtx(dim_signal, num_atoms, normalize_atoms=True):
    shape = [num_atoms, dim_signal]
    sigma = math.sqrt(dim_signal)
    dict = tf.random.normal(shape, stddev=sigma)
    if normalize_atoms:
        dict = normalize_l2_rw(dict)
    return dict

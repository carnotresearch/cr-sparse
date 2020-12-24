import tensorflow as tf
import numpy as np
from .defs import SingleRecoverySolution
from cr.sparse.norm import *

class MatchingPursuit:

    def __init__(self, dict):
        # reach row is an atom
        # number of rows is number of atoms in dictionary
        self.dict = dict
        shape = dict.shape
        self.num_atoms = shape[0]
        self.dim_signal = shape[1]


    def __call__(self, signals, max_iters=None, max_res_norm=None):
        # initialize residual
        residuals = tf.Variable(signals)
        num_signals = signals.shape[0]
        # initialize solution vector
        # z = tf.Variable(tf.zeros(self.num_atoms))
        sol_shape = (num_signals, self.num_atoms)
        z = np.zeros(sol_shape)
        # iteration count
        t = 0
        dict = self.dict
        # compute the norm of original signal
        x_norms = norms_l2_rw(signals)
        # absolute limit on res norm
        upper_res_norm = tf.reduce_max(x_norms) * 1e-6
        # upper limit on number of iterations
        upper_iters = 4 * self.num_atoms
        while True:
            # Compute the inner product of residual with atoms
            correlations = tf.matmul(dict, residuals, transpose_b=True)
            #print(correlations.shape)
            # each correlation column is for one signal
            # take absolute values
            abs_corrs = tf.abs(correlations)
            # find the maximum in the column
            indices = tf.math.argmax(abs_corrs, axis=0)
            for i in range(num_signals):
                # best match atom index
                best_match_index = indices[i]
                # pick corresponding correlation value
                coeff = correlations[best_match_index, i]
                # update the representation
                z[i, best_match_index] += coeff
                # find the best match atom
                atom = dict[best_match_index]
                # update the residual
                residuals[i, :].assign(residuals[i, :] - coeff * atom) 
            t += 1
            # compute the updated residual norm
            r_norms = norms_l2_rw(residuals)
            max_r_norm = tf.reduce_max(r_norms)
            # print("[{}] norm: {}".format(t, r_norm))
            print('.', end="", flush=True)
            if max_iters is not None and t >= max_iters:
                break
            if max_res_norm is not None and max_r_norm < max_res_norm:
                break
            if max_r_norm < upper_res_norm:
                break
            if t >= upper_iters:
                break
            #print("[{}] res norm: {}".format(t, max_r_norm))
        solution = SingleRecoverySolution(signals=signals, 
            representations=z, 
            residuals=residuals, 
            residual_norms=r_norms,
            iterations=t)
        return solution


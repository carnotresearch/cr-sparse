import tensorflow as tf
import numpy as np
from .defs import SingleRecoverySolution

class MatchingPursuit:

    def __init__(self, dict):
        # reach row is an atom
        # number of rows is number of atoms in dictionary
        self.dict = dict
        shape = dict.shape
        self.num_atoms = shape[0]
        self.dim_signal = shape[1]


    def __call__(self, x, max_iters=None, max_res_norm=None):
        # initialize residual
        r = x
        # initialize solution vector
        # z = tf.Variable(tf.zeros(self.num_atoms))
        z = np.zeros(self.num_atoms)
        # iteration count
        t = 0
        dict = self.dict
        # compute the norm of original signal
        x_norm = tf.norm(x)
        # absolute limit on res norm
        upper_res_norm = x_norm * 1e-6
        # upper limit on number of iterations
        upper_iters = 4 * self.num_atoms
        while True:
            # Compute the inner product of residual with atoms
            h = dict @ tf.reshape(r, (self.dim_signal, 1))
            # h is a row vector squeeze it
            h = tf.squeeze(h)
            # find the atom with maximum correlation
            abs_h = tf.abs(h)
            # find the maximum in the vector
            index = tf.math.argmax(abs_h)
            # pick corresponding correlation value
            coeff = h[index]
            # update the representation
            # z[index].assign(z[index] + coeff)
            z[index] += coeff
            # update the residual
            atom = dict[index]
            r = r - coeff * atom
            t += 1
            # compute the updated residual norm
            r_norm = tf.norm(r)
            # print("[{}] norm: {}".format(t, r_norm))
            if max_iters is not None and t >= max_iters:
                break
            if max_res_norm is not None and r_norm < max_res_norm:
                break
            if r_norm < upper_res_norm:
                break
            if t >= upper_iters:
                break
        solution = SingleRecoverySolution(signal=x, 
            representation=z, 
            residual=r, 
            residual_norm=r_norm,
            iterations=t)
        return solution


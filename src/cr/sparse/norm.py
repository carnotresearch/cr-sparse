import numpy as np
import tensorflow as tf

EPS = np.finfo(np.float32).eps


def norms_l1_cw(X):
    """
    Computes the l_1 norm of each column of a matrix
    """
    return tf.norm(X, ord=1, axis=0)

def norms_l1_rw(X):
    """
    Computes the l_1 norm of each row of a matrix
    """
    return tf.norm(X, ord=1, axis=1)

def norms_l2_cw(X):
    """
    Computes the l_2 norm of each column of a matrix
    """
    return tf.norm(X, ord='euclidean', axis=0, keepdims=False)

def norms_l2_rw(X):
    """
    Computes the l_2 norm of each row of a matrix
    """
    return tf.norm(X, ord='euclidean', axis=1, keepdims=False)


def norms_linf_cw(X):
    """
    Computes the l_inf norm of each column of a matrix
    """
    return tf.norm(X, ord=np.inf, axis=0)

def norms_linf_rw(X):
    """
    Computes the l_inf norm of each row of a matrix
    """
    return tf.norm(X, ord=np.inf, axis=1)



######################################
# Normalization of rows and columns
######################################


def normalize_l1_cw(X):
    """
    Normalize each column of X per l_1-norm
    """
    X2 = tf.abs(X)
    sums = tf.reduce_sum(X2, axis=0) + EPS
    return tf.divide(X, sums)

def normalize_l1_rw(X):
    """
    Normalize each row of X per l_1-norm
    """
    X2 = tf.abs(X)
    sums = tf.reduce_sum(X2, axis=1) + EPS
    # row wise sum should be a column vector
    sums = tf.expand_dims(sums, axis=-1)
    # now broadcasting works well
    return tf.divide(X, sums)

def normalize_l2_cw(X):
    """
    Normalize each column of X per l_2-norm
    """
    X2 = tf.square(X)
    sums = tf.reduce_sum(X2, axis=0) 
    sums = tf.sqrt(sums)
    return tf.divide(X, sums)

def normalize_l2_rw(X):
    """
    Normalize each row of X per l_2-norm
    """
    X2 = tf.square(X)
    sums = tf.reduce_sum(X2, axis=1)
    sums = tf.sqrt(sums)
    # row wise sum should be a column vector
    sums = tf.expand_dims(sums, axis=-1)
    # now broadcasting works well
    return tf.divide(X, sums)


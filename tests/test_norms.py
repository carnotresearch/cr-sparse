from math import sqrt

from cr.sparse.norm import *
import tensorflow as tf

import pytest

# Integers
X = tf.constant([[1, 2],
             [3, 4]])

# Floats
X2 = tf.constant([[1, 2],
             [3, 4]], dtype=tf.float32)

def check_equal(actual, expected):
    print(actual, expected)
    x = tf.equal(actual, expected)
    print(x)
    x = tf.reduce_all(x)
    print(x)
    assert x


def check_approx_equal(actual, expected, abs_err=1e-6):
    diff = actual - expected
    abs_diff = tf.abs(diff)
    print(actual, expected, abs_diff)
    ok = tf.less(abs_diff, abs_err)
    ok = tf.reduce_all(ok)
    assert ok


def test_l1_norm_cw():
    check_equal(norms_l1_cw(X), tf.constant([4, 6]))

def test_l1_norm_rw():
    check_equal(norms_l1_rw(X), tf.constant([3, 7]))


def test_l2_norm_cw():
    check_equal(norms_l2_cw(X2), tf.constant([sqrt(10), sqrt(20)]))

def test_l2_norm_rw():
    check_equal(norms_l2_rw(X2), tf.constant([sqrt(5), sqrt(25)]))
 

def test_linf_norm_cw():
    check_equal(norms_linf_cw(X), tf.constant([3, 4]))

def test_linf_norm_rw():
    check_equal(norms_linf_rw(X), tf.constant([2, 4]))


def test_normalize_l1_cw():
    Y = normalize_l1_cw(X2)
    check_equal(norms_l1_cw(Y), tf.constant([1., 1.]))

def test_normalize_l1_rw():
    Y = normalize_l1_rw(X2)
    check_equal(norms_l1_rw(Y), tf.constant([1., 1.]))


def test_normalize_l2_cw():
    Y = normalize_l2_cw(X2)
    check_approx_equal(norms_l2_cw(Y), tf.constant([1., 1.]))

def test_normalize_l2_rw():
    Y = normalize_l2_rw(X2)
    check_approx_equal(norms_l2_rw(Y), tf.constant([1., 1.]))

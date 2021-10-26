import pytest

from cr.sparse import *
import cr.sparse as crs

import jax.numpy as jnp

M = 10
p = 3
N = 5

A = jnp.zeros([M, p])
B = jnp.ones([M, p])

C = A.T
D = B.T

def test_1():
    crs.pairwise_sqr_l2_distances_rw(A, B)

def test_2():
    crs.pairwise_sqr_l2_distances_cw(A, B)

def test_a():
    crs.pairwise_l2_distances_rw(A, B)

def test_b():
    crs.pairwise_l2_distances_cw(A, B)

def test_c():
    crs.pdist_sqr_l2_rw(A)

def test_d():
    crs.pdist_sqr_l2_cw(A)

def test_e():
    crs.pdist_l2_rw(A)

def test_f():
    crs.pdist_l2_cw(A)


def test_3():
    pairwise_l1_distances_rw(C, D)

def test_4():
    pairwise_l1_distances_cw(A, B)

def test_5():
    pdist_l1_rw(C)

def test_6():
    pdist_l1_cw(A)

def test_7():
    crs.pairwise_linf_distances_rw(C, D)


def test_8():
    crs.pairwise_linf_distances_cw(A, B)

def test_9():
    crs.pdist_linf_rw(C)

def test_10():
    crs.pdist_linf_cw(B)

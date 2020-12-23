
from cr.sparse.checks import *
import tensorflow as tf

import pytest


def test_vec():
    x = tf.constant([1,2,3])
    assert is_vec(x)


def test_row_vec():
    x = tf.constant([1,2,3])
    y = tf.expand_dims(x, axis=0)
    assert is_row_vec(y)
    assert not is_vec(y)

def test_col_vec():
    x = tf.constant([1,2,3])
    y = tf.expand_dims(x, axis=-1)
    assert is_col_vec(y)
    assert not is_vec(y)


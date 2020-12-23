

def is_scalar(x):
    return x.ndim == 0

def is_singleton(x):
    return tf.size(x)

def is_vec(x):
    return x.ndim == 1


def is_matrix(x):
    return x.ndim == 2

def is_row_vec(x):
    return x.ndim == 2 and x.shape[0] == 1

def is_col_vec(x):
    return x.ndim == 2 and x.shape[1] == 1

import tensorflow as tf

class SparseRepGenerator:

    def __init__(self, dim_rep, sparsity, count, dtype=tf.float32):
        self.dim_rep = dim_rep
        self.sparsity = sparsity
        self.count = count
        #  generate the sequence 0..N-1
        r = tf.range(dim_rep)
        r = tf.random.shuffle(r)
        # keep the first K
        r = r[:sparsity]
        self.omega = r
        # output
        shape = [count, dim_rep]
        self.output = tf.Variable(tf.zeros(shape, dtype=dtype))

    def gaussian(self):
        shape = [self.count, self.sparsity]
        data = tf.random.normal(shape)
        for (i, index) in enumerate(self.omega):
            self.output[:, index].assign(data[:, i])
        return self.output



from cr.sparse.dict import simple
from cr.sparse.norm import *
from cr.sparse.data.synthetic.random import *

dict = simple.gaussian_mtx(4,10)
print("Sample Dictionary:")
print(dict)
print (norms_l2_rw(dict))


N = 16
K = 3
S = 6
gen = SparseRepGenerator(N, K, S)

signals = gen.gaussian()

print("Sample Signals: ")
print(signals)
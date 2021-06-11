Computation Time Comparison of Sparse Recovery Methods
===============================================================

Performance on CPU
-------------------------------------------------

System configuration

* MacBook Pro 2019 Model
* Processor: 1.4 GHz Quad Core Intel Core i5
* Memory: 8 GB 2133 MHz LPDDR3

Problem Specification

* Gaussian sensing matrices (normalized to unit norm columns)
* Sparse vectors with non-zero entries drawn from Gaussian distributions
* M, N, K have been chosen so that all algorithms under comparison are known to converge to successful 
  recovery.


.. list-table:: Average time (msec) and speedups due to JIT compilation
    :header-rows: 1

    * - method
      - M
      - N
      - K
      - iterations
      - jit_off
      - jit_on
      - speedup
    * - OMP
      - 200
      - 1000
      - 20
      - 20
      - 105.78
      - 2.14
      - 49.48
    * - SP
      - 200
      - 1000
      - 20
      - 3
      - 1645.32
      - 2.73
      - 602.34
    * - CoSaMP
      - 200
      - 1000
      - 20
      - 4
      - 309.01
      - 6.20
      - 49.84
    * - IHT
      - 200
      - 1000
      - 20
      - 65
      - 232.99
      - 36.27
      - 6.42
    * - NIHT
      - 200
      - 1000
      - 20
      - 16
      - 240.96
      - 5.64
      - 42.72
    * - HTP
      - 200
      - 1000
      - 20
      - 5
      - 1491.00
      - 13.71
      - 108.76
    * - NHTP
      - 200
      - 1000
      - 20
      - 4
      - 1467.35
      - 1.98
      - 741.88

Some comments on the results are in order.

.. rubric:: Without JIT vs With JIT

* It is clear that all algorithms exhibit significant speedups with the introduction of 
  JIT compilation.
* The speedup is as low as 6x for IHT and as high as 740x in NHTP.
* After JIT compilation, IHT is the slowest algorithm while NIHT is the fastest.
* It appears that steps like dynamic step size computation and least squares tend to get
  aggressively optimized and lead to massive speed gains.

.. rubric:: OMP

* With JIT on, OMP is actually one of the fastest algorithms in the mix.
* In the current implementations, OMP is the only one in which the least squares step has
  been optimized using Cholesky updates. 
* This is possible as OMP structure allows for adding atoms one at a time to the mix.
* Other algorithms change several atoms [add / remove] in each iteration. Hence, such
  optimizations are not possible.
* The least squares steps in other algorithms can be accelerated using small number of conjugate gradients
  iterations. However, this hasn't been implemented yet.


.. rubric:: SP vs CoSaMP

* CoSaMP has one least squares step (on 3K indices) in each iteration.
* SP (Subspace Pursuit) has two least squares steps in each iteration.
* Without JIT, CoSaMP is 5x faster.
* With JIT, SP becomes 2.5x faster than CoSaMP.
* Thus, SP seems to provide more aggressive optimization opportunities.

.. rubric:: IHT vs NIHT

* IHT and NIHT are both simple algorithms. They don't involve a least squares step in their iterations.
* The main difference is that the step-size fixed for IHT and it is computed on every iteration in NIHT.
* The dynamic step size leads to reduction in the number of iterations for NIHT. From 65 to 16, 4x reduction.
* Without JIT, there is no significant difference between IHT and NIHT.
  Thus, step-size computation seems to contribute a lot to computation time without compilation.
* With JIT, step-size computation seems to be aggressively optimized.
  NIHT after JIT is 6x faster than IHT even though the number of iterations reduces by only 4 times
  and there is extra overhead of computing the step size.



.. rubric:: IHT vs HTP

* The major difference in the two algorithms is that HTP performs a least squares estimate
  on the current guess of signal support
* Although the number of iterations reduces 13 times, the computation speed up is not so
  high (due to extra overhead of least squares computation)
* Without JIT, HTP becomes much slower than IHT. Thus, overhead of a least squares step is quite high.
* HTP is about 3x faster than IHT with JIT. This makes sense. The number of iterations reduced by 13
  times and the overhead of least squares was added.


.. rubric:: HTP vs NHTP

* Just like NIHT, NHTP also introduces computing the step size dynamically in every iteration.
* It helps in reducing the number of iterations from 5 to 4.
* In this case, the benefit of dynamic step size is not visible much in terms of iterations.
* Without JIT, there is not much benefit either.
* However, with JIT, NHTP is 6x faster than HTP. This speedup is unusual as there is just
  20% reduction in number of iterations and there is the overhead of step size computation.

 


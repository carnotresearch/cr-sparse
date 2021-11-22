---
title: 'CR-Sparse: Hardware accelerated functional algorithms for sparse signal processing in Python using JAX'
tags:
    - Python
    - sparse and redundant representations
    - compressive sensing
    - wavelets
    - linear operators
    - sparse subspace clustering
    - functional programming
authors:
    - name: Shailesh Kumar
      affiliation: 1
      orcid: 0000-0003-2217-4768

affiliations:
    - name: Indian Institute of Technology, Delhi
      index: 1


date: 5 November 2021
bibliography: paper.bib
---

# Summary

We introduce [`CR-Sparse`](https://github.com/carnotresearch/cr-sparse), 
a Python library that enables efficiently solving
a wide variety of sparse representation based signal processing problems.
It is a cohesive collection of sub-libraries working together. Individual
sub-libraries provide functionalities for:
wavelets, linear operators, greedy and convex optimization 
based sparse recovery algorithms, subspace clustering, 
standard signal processing transforms,
and linear algebra subroutines for solving sparse linear systems. 
It has been built using Google JAX [@jax2018github], which enables the same high level
Python code to get efficiently compiled on CPU, GPU and TPU architectures
using XLA [@abadi2017computational]. 

![Sparse signal representations and compressive sensing](./srr_cs.png)

Traditional signal processing exploits the underlying structure in signals
by representing them using Fourier or wavelet orthonormal bases. 
In these representations,
most of the signal energy is concentrated in few coefficients allowing greater
flexibility in analysis and processing of signals. More flexibility can be
achieved by using overcomplete dictionaries [@mallat2008wavelet]
(e.g. unions of orthonormal bases). However, the construction of
sparse representations of signals in these overcomplete dictionaries 
is no longer straightforward and requires use of specialized sparse
coding algorithms like orthogonal matching pursuit [@pati1993orthogonal]
or basis pursuit [@chen2001atomic]. The key idea behind these algorithms 
is the fact that under-determined systems $A x = b$ can be solved efficiently
to provide sparse solutions $x$ if the matrix $A$ satisfies specific conditions
on its properties like coherence. Compressive sensing takes the same 
idea in the other direction and contends that signals having sparse representations
in suitable bases can be acquired by very few data-independent 
random measurements $y = \Phi x$ if the sensing or measurement system $\Phi$
satisfies certain conditions like restricted isometry property [@candes2008restricted].
The same sparse coding algorithms can be tailored for sparse signal recovery
from compressed measurements. 

A short mathematical introduction to compressive sensing and sparse representation problems 
is provided in [online documentation](https://cr-sparse.readthedocs.io/en/latest/intro.html).
For comprehensive introduction to sparse
representations and compressive sensing,
 please refer to excellent books [@mallat2008wavelet;@elad2010sparse;@foucart2013mathintro],
papers [@donoho2006compressed;@qaisar2013compressive;@marques2018review],
[Rice Compressive Sensing Resources](https://dsp.rice.edu/cs/) and references therein.

# Package Overview

The `cr.sparse.pursuit` package includes greedy and thresholding
based solvers for sparse recovery. It includes: 
`OMP`, `CoSaMP`, `HTP`, `IHT`, `SP` algorithms. Normalized versions
of `HTP` and `IHT` algorithms are also supported.
These algorithms can work with unstructured random and dense sensing matrices
as well as structured sensing matrices represented as linear operators
(provided in `cr.sparse.lop` package).

The `cr.sparse.cvx.adm.yall1` package includes a JAX version of 
`Your algorithms for l1` [@zhang2009user], an efficient solver
for l1-minimization problems using 
ADMM (alternating direction method of multipliers) approach.
The `cr.sparse.cvx.l1ls` includes 
*Truncated Newton Interior Points Method* for solving 
the l1-minimization problem. 


The `cr.sparse.sls` package provides JAX versions of
`LSQR`, `ISTA`, `FISTA`  algorithms for solving sparse linear 
systems. The linear system is represented by a linear operator
from `cr.sparse.lop`. It also includes a power iteration
algorithm to compute the largest eigen value of a 
symmetric linear operator.

The `cr.sparse.dict` package provides matrix versions of a variety
of random/structured sensing matrices and dictionaries. It
also provides some tools for dictionary analysis like:
coherence, babel function, frame bounds, etc..

The `cr.sparse.cluster` package includes JAX versions of 
K-means clustering, spectral clustering. It further
includes support for Sparse Subspace Clustering (SSC) where 
data points are assumed to lie in a small set of disjoint low dimensional
subspaces of an ambient space. Currently SSC is implemented using
OMP. 

The `cr.sparse.lop` package includes a collection of linear operators
influenced by `PyLops` [@ravasi2019pylops]. The design is different
following functional programming principles.
Supported operators include: 
`identity`, `block_diag`, `matrix`, `diagonal`, `zero`, `flipud`,
`sum`, `pad_zeros`, `symmetrize`, `restriction`, 
`running_average`, `fir_filter`, 
`convolve`, `convolve2D`, `convolveND`, 
`fourier_basis`, `dirac_fourier_basis`, `cosine_basis`, `walsh_hadamard_basis`,
`dwt`, `dwt2D`, 
`first_derivative`, `second_derivative`,
`circulant`, 
`gaussian_dict`, `rademacher_dict`, `random_onb_dict`.
All operator implementations can be jit compiled using `lop.jit`.
Operator algebra features allow one to combine different operators 
to form new operators. Following unary and binary functions are available:
`neg`, `scale`, `partial_op`, `add`, `subtract`, `compose`, 
`transpose`, `hermitian`, `hcat`, `power`, `gram`, `frame`.

Following sub-libraries in `cr.sparse` have been built to enrich the 
suite of linear operators in `cr.sparse.lop`.

`cr.sparse.wt` package includes a JAX version of major functionality
from `PyWavelets` [@lee2019pywavelets] making it a first major pure 
Python wavelets implementation which can work across CPUs, GPUs and TPUs.
The API includes:  `dwt`, `idwt`, `dwt2`, `idwt2`, 
`upcoef`, `downcoef`, `wavedec`, `waverec`, `cwt` functions. 
The discrete wavelets supported include: 
Haar, Daubechies, Symlets, Coiflets, Biorthogonal, 
Reverse biorthogonal, Discrete Meyer.
Continuous wavelets include: Complex Morlet and Ricker (or Mexican Hat).


The `cr.sparse.dsp` package includes JAX versions of 
`DCT`, `IDCT`, `WHT` transforms. These in turn are used
in the `lop` package. It also provides a number of utilities
to construct synthetic signals like chirps, pulses, Gaussian pulses,
etc..

The `cr.sparse.la` package provides JAX versions of a set of linear algebra
subroutines as required by other higher level modules. 
These are built on top of `jax.numpy` and bridge the necessary gap.
It also includes special routines for solving truncated SVD problems.
JAX doesn't include `eigs` and `svds` routines at the moment.
Several sparse problems require such functionality. We provide
an implementation of **Lanczos Bidiagonalization with Partial
Reorthogonalization** procedure which can be used to compute 
truncated SVD. 


# Statement of need

Currently, there is no single Package which provides a 
comprehensive set of tools for solving sparse recovery problems
in one place. Individual researchers provide their codes
along with their research paper only for the algorithms they have
developed. Most of this work is available in the form of MATLAB [@MATLAB:2018]
libraries. E.g.: [`YALL1`](http://yall1.blogs.rice.edu)is the original MATLAB implementation of the ADMM based sparse recovery algorithms. 
[`L1-LS`](https://web.stanford.edu/~boyd/l1_ls/) 
is the original MATLAB implementation of the
Truncated Newton Interior Points Method for solving the l1-minimization problem.
[`Sparsify`](https://www.southampton.ac.uk/engineering/about/staff/tb1m08.page#software) 
provides the MATLAB implementations of IHT, NIHT, AIHT algorithms.
[`aaren/wavelets`](https://github.com/aaren/wavelets) 
is a decent CWT implementation following
[@torrence1998practical]. 
[`HTP`](https://github.com/foucart/HTP) provides implementation of Hard Thresholding
Pursuit in MATLAB.
[`WaveLab`](https://github.com/gregfreeman/wavelab850) is the 
old reference open source wavelet implementation in MATLAB.
However, its API has largely been superceded by later libraries.
[`Sparse and Redundant Representations book code`](https://elad.cs.technion.ac.il/wp-content/uploads/2018/02/Matlab-Package-Book-1.zip) [@elad2010sparse]
provides basic implementations of a number of sparse recovery and related
algorithms.
Several of these libraries contain key 
performance critical sub-routines
in the form of C/C++ extensions making portability to GPUs harder. 

There are some Python libraries which focus on specific
areas however they are generally CPU based.
E.g., [`pyCSalgos`](https://github.com/nikcleju/pyCSalgos) is an
old Python implementation of various Compressed Sensing algorithms.
[`spgl1`](https://github.com/drrelyea/spgl1) is a `NumPy` based
implementation of spectral projected gradient for L1 minimization.
`c-lasso` [@simpson2021classo] is a Python package for constrained sparse regression 
and classification. This is also CPU only. 
[`PyWavelets`](https://github.com/PyWavelets/pywt) is an excellent 
CPU only wavelets implementation in Python closely following the API
of Wavelet toolbox in MATLAB. The performance critical parts have been
written entirely in C. There are several attempts to port it on GPU
using `PyTorch` ([PyTorch-Wavelet-Toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox)) 
or `Tensorflow` ([tf-wavelets](https://github.com/UiO-CS/tf-wavelets)) backends.
[`PyLops`](https://github.com/PyLops/pylops) includes GPU support. 
They have built a [`backend.py`](https://github.com/PyLops/pylops/blob/master/pylops/utils/backend.py) 
layer to switch explicitly between
`NumPy` and [`CuPy`](https://cupy.dev/) for GPU support. 
In contrast, 
our use of JAX enables us to perform jit compilation with 
abstracted out end-to-end XLA optimization to multiple backend.

The algorithms in this package have a wide variety of applications. We list
some: compressive imaging, medical imaging, compressive radar,


# Sparse signal processing problems and available solvers

A mathematical introduction to the problems supported by this library
is given in  the online [documentation](https://cr-sparse.readthedocs.io/en/latest/intro.html).

We provide JAX based implementations for the following greedy pursuit algorithms:

* `cr.sparse.pursuit.omp`: Orthogonal Matching Pursuit (OMP) [@pati1993orthogonal;@tropp2004greed;@davenport2010analysis] 
* `cr.sparse.pursuit.cosamp`: Compressive Sampling Matching Pursuit (CoSaMP) [@needell2009cosamp]
* `cr.sparse.pursuit.sp`: Subspace Pursuit (SP) [@dai2009subspace]
* `cr.sparse.pursuit.iht`: Iterative Hard Thresholding and its normalized version (IHT, NIHT) [@blumensath2009iterative;@blumensath2010normalized]
* `cr.sparse.pursuit.htp`: Hard Thresholding Pursuit and its normalized version (HTP, NHTP) [@foucart2011recovering]

For details, see the online [documentation](https://cr-sparse.readthedocs.io/en/latest/source/pursuit.html).

The `cr.sparse.cvx.admm` package includes solvers for basis pursuit (BP),
basis pursuit denoising (BPDN), basis pursuit with inequality constraints (BPIC),
and their nonnegative variants.

For details, see our online [tutorial](https://cr-sparse.readthedocs.io/en/latest/tutorials/admm_l1.html).

## Linear Operators

While major results on the recovery guarantees of sparse recovery algorithms focus on 
random matrices, actual applications prefer to use structured dictionaries and 
sensing matrices so that the operations $A x$ and $A^T x$ can be efficiently 
computed. These structured matrices fall under the more general class
of linear operators. In order to fully exploit the power of sparse 
recovery algorithms, it was deemed necessary to provide a complementary 
set of linear operators.

We provide a large collection of linear operators in `cr.sparse.lop`.
For the complete list, see the online [documentation](https://cr-sparse.readthedocs.io/en/latest/source/lop.html).

Although inspired by `PyLops` [@ravasi2019pylops], there are several
differences in our implementation. 

- We use `times` and `trans` functions to represent operators.
- The operators `+`, `-`, `@`, `**` etc. are overridden to provide operator algebra,
  i.e. ways to combine operators to generate new operators.
- All our operators can be JIT compiled. Hence, they can be sent as static
  arguments to other functions (like LSQR, FISTA, etc.) which can be JIT compiled.
- Our 2D and ND operators accept 2D/ND arrays as input and return 2D/ND arrays as
  output. We don't require callers to flatten input before applying the operator.
- Several operators are provided which are specifically meant for sparse
  representation and compressive sensing applications.
- We believe that our implementation is cleaner and simpler and yet gives better
  performance (thanks to JAX) on large size problems.

It is easy to extend the library with new operators.


## Sparse Subspace Clustering

As an application area, the library includes an implementation of
sparse subspace clustering (SSC) by orthogonal matching pursuit
[@vidal2010tutorial; @dyer2013greedy; @you2015sparse; @you2016scalable]
in the `cr.sparse.cluster.ssc` package.

This [section](https://cr-sparse.readthedocs.io/en/latest/ssc/intro.html) 
in docs provides a theoretical intro to SSC.
See [here](https://cr-sparse.readthedocs.io/en/latest/source/ssc.html) for API and examples.

The `cr.sparse.cluster.spectral` package provides a custom JAX
based implementation of spectral clustering.
In particular, it
uses our own Lanczos Bidiagonalization with Partial Orthogonalization (LANBPRO)
algorithm to compute the K largest singular values of the
normalized affinity matrix
in as few iterations as possible.


# Experimental Results

## Runtime Comparisons

We conducted a number of experiments to benchmark the runtime of 
`CR-Sparse` implementations viz. existing reference software
in Python or MATLAB. In this section, we present a small selection
of these results. Jupyter notebooks to reproduce these micro-benchmarks
are available on the 
[`cr-sparse-companion`](https://github.com/carnotresearch/cr-sparse-companion)
repository. Readers are encouraged to try them out.

All Python based benchmarks have been run
on the machine configuration: 
Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 16 Cores, 64 GB RAM, 
NVIDIA GeForce GTX 1060 6GB GPU, 
Ubuntu 18.04 64-Bit, Python 3.8.8, 
NVidia driver version 495.29.05,
CUDA version 11.5.

Versions for specific Python libraries are: 
`numpy=1.20.1`, `scipy=1.6.2`, `scikit-image=0.18.1`,
`scikit-learn=0.24.1`, `PyWavelets=1.1.1`, `pylops=1.15.0`,
`jax=0.2.24`, `jaxlib=0.1.73+cuda11.cudnn82`. 

MATLAB based benchmarks were run on the machine configuration:
Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz   2.30 GHz,
32 GB RAM, Windows 10 Pro, MATLAB R2020b.



The following table provides comparison of `CR-Sparse` against reference 
implementations on a set of representative problems:

\footnotesize

| Problem | Size | Ref tool | Ref time | Our time | Gain |
|:----:|:---:|:--:|:-:|:-:|:-:|
| Hard Thresholding Pursuit | M=2560, N=10240, K=200 | HTP (MATLAB) | 3.5687 s | 160 ms | 22x |  
| Orthogonal Matching Pursuit | M=2000, N=10000, K=100 | sckit-learn | 379 ms | 120 ms | 3.15x |  
| ADMM, BP | M=2000, N=20000, K=200 | YALL1 (MATLAB) | 1.542 sec | 445 ms | 3.46x |  
| ADMM, BPDN | M=2000, N=20000, K=200 | YALL1 (MATLAB) | 1.572.81 sec | 273 ms | 5.75x |  
| Image blurring | Image: 500x480, Kernel: 15x25 | Pylops | 6.63 ms | 1.64 ms | 4x |  
| Image deblurring using LSQR | Image: 500x480, Kernel: 15x25 | Pylops | 237 ms | 39.3 ms | 6x |  
| Image DWT2 | Image: 512x512 | PyWavelets | 4.48 ms | 656 µs | 6.83x |  
| Image IDWT2 | Image: 512x512 | PyWavelets | 3.4 ms | 614 µs | 5.54x |  
| OMP for SSC | 5 subspaces 50K points | SSCOMP_Code (MATLAB) | 52.5 s | 10.2 s | 4.6x |

\normalsize


We see significant gains achieved by `CR-Sparse` running on GPU although gain levels are not 
uniform. Few comments are in order. We have observed that gain tends to increase for larger
problem sizes. GPUs tend to perform better when problem size increases as the matrix/vector 
products become bigger. For smaller problems, we have seen that CPU implementations perform
quite well. Our focus so far has been to produce straightforward and 
faithful JAX based implementations of the
algorithms concerned. We believe with further research, more optimized implementations may 
be possible. JAX provides `vmap` which makes it very easy to vectorize an algorithm over
a number of similar problems. It also provides `pmap` which can be used to distribute 
a number of similar problems over multiple GPU devices. These tools make it extremely 
easy to parallelize the algorithms provided in `CR-Sparse` over multiple data.
The largest problem an algorithm in `CR-Sparse` can solve also depends on available GPU
memory. This should be considered carefully while planning.


### Linear operators 

Following table compares the runtime of linear operators in `CR-Sparse` on GPU vs `PyLops` on CPU for large size problems. 
Timings are measured for both forward and adjoint operations. For a linear operator $A$, 
the forward operation is the computation $y = A x$ 
and the adjoint operation is the computation $\hat{x} = A^H y$.
Linear operators from $\mathbb{R}^n$ to $\mathbb{R}^m$
can be represented as a matrix of size (m,n). For some linear operators,
$m=n$ by definition and we just mention $n$.

\footnotesize

| Operator | Size | Fwd ref | Fwd our | Gain | Adj ref | Adj our | Gain |
|:--------:|:----:|:-:|:-:|:-:|:-:|:-:|:-:|
| Diagonal matrix mult| n=1M | 966 µs | 95.7 µs | 10x | 992 µs | 96.3 µs | 10x | 
| Matrix mult | (m,n)=(10K,10K) | 11 ms | 2.51 ms | 4.37x | 11.6 ms | 2.51 ms | 4.63x |
| First derivative | n=1M | 2.15 ms | 71.1 µs | 30.2x | 2.97 ms | 186 µs | 15.97x |
| HAAR DWT2, level=8 | in=(4K,4K) | 981 ms | 34.4 ms | 28.5x | 713 ms | 60.8 ms | 11.7x | 

For HAAR DWT2, input image is of shape (4000, 4000) and the output
wavelet coefficients image is of shape (4096, 4096).

\normalsize

Notebooks for reproducing these micro-benchmarks are also available in the 
[companion](https://github.com/carnotresearch/cr-sparse-companion/tree/main/comparison/pylops) repository.

We would like to mention that for small sizes, `pylops` on CPU runs much quicker. 
Benefits of `CR-Sparse` on GPU can be seen on large sizes.

# Limitations

Some of the limitations in the library come from the underlying 
JAX library. 
JAX is relatively new and still hasn't 
reached `1.0` level maturity. 
The programming model chosen by JAX places
several restrictions on expressing the program logic. For example,
JAX does not have support for dynamic or data dependent shapes
in their [JIT compiler](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#to-jit-or-not-to-jit).
Thus, any algorithm parameter which determines the size/shape
of individual arrays in an algorithm must be statically provided.
E.g. for the greedy algorithms like OMP, 
the sparsity level $K$ must be known in advance
and provided as a static parameter to the API as the size of
output array depends on $K$. 

The control flow primitives like `lax.while_loop`, `lax.fori_loop`
etc. in JAX require that the algorithm state flowing between iterations
must not change shape and size. This makes coding of algorithms
like OMP or SVT (singular value thresholding) very difficult.
An incremental QR or Cholesky decomposition based implementation of OMP requires
growing algorithm state. We ended up using a standard Python `for` loop
for now but the JIT compiler simply unrolls it and doesn't allow for tolerance
based early termination in them. 

1D convolutions are slow in JAX on CPU 
[#7961](https://github.com/google/jax/discussions/7961). 
This affects the performance of DWT/IDWT in `cr.sparse.dwt`. 
We are working on exploring ways of making it more efficient 
while keeping the API intact.

Support for sparse array storage is still 
[experimental](https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html)
and is limited by static size requirements of JIT compiler. See [#8299](https://github.com/google/jax/issues/8299). 

These restrictions imply good amount of creativity and a very
disciplined coding style so that efficient JIT friendly 
solvers can be developed.

For more details, see the [limitations](https://cr-sparse.readthedocs.io/en/latest/dev/limitations.html) section in documentation.


# Future Work

Currently, work is underway to provide a JAX based
implementation of [`TFOCS`](http://cvxr.com/tfocs/) [@becker2011templates]
in the dev branch.
This will help us increase the coverage to a wider set of
problems (like total variation minimization, Dantzig selector,
l1-analysis, nuclear norm minimization, etc.). As part of this
effort, we are expanding our collection of linear operators and
building a set of indicator and projector functions on to
convex sets and proximal operators [@parikh2014proximal].
This will enable us to cover other applications such as
SSC-L1 [@pourkamali2020efficient]. 
In future, we intend to increase the coverage in following areas:
More recovery algorithms (OLS, Split Bergmann, SPGL1, etc.) 
and specialized cases (partial known support, );
Bayesian Compressive Sensing;
Dictionary learning (K-SVD, MOD, etc.);
Subspace clustering;
Image denoising, compression, etc. problems using sparse representation principles;
Matrix completion problems;
Matrix factorization problems;
Model based / Structured compressive sensing problems;
Joint recovery problems from multiple measurement vectors.

# Acknowledgements

Shailesh would like to thank his Ph.D. supervisors Prof. Surendra 
Prasad and Prof. Brejesh Lall to inculcate his interest in this area
and support him over the years in his exploration. He would also
like to thank his employers, Interra Systems Inc., for allowing him
to pursue his research interests along with his day job.


# References


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

header-includes:
    - \usepackage{setspace}
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
using XLA.  

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

There are some large libraries which focus on specific
areas however they are generally CPU based. Additional effort 
is spent in making the functionality available on GPUs.
[`PyWavelets`](https://github.com/PyWavelets/pywt) is an excellent 
CPU only wavelets implementation in Python closely following the API
of Wavelet toolbox in MATLAB. The performance critical parts have been
written entirely in C. There are several attempts to port it on GPU
using `PyTorch` or `Tensorflow` backends.
[`PyLops`](https://github.com/PyLops/pylops) includes a GPU backend
apart from its CPU implementation but the implementation appears to be
somewhat complex.

JAX is a new library which provides a `NumPy` [@oliphant2006guide] 
like API and provides
a just in time compiler to compile JAX based functions to a variety
of hardware architectures via XLA [@abadi2017computational],
a domain-specific compiler for linear algebra.
It enforces a functional programming paradigm on functions which
can be JIT compiled. This makes it very easy for the compiler 
to reason about the program and generate efficient code. 

JAX provides an excellent platform to build a comprehensive 
library of solvers which can work across a variety of hardware
platforms. It also frees users from the concern of supporting 
new hardware later as becomes the job of the JIT compiler.

At the same time, JAX is relatively new and still hasn't 
reached `1.0` level maturity. 
Besides the functional programming model places
several restrictions on expressing the program logic. For example,
JAX does not have support for dynamic or data dependent shapes.
Thus, any algorithm parameter which determines the size/shape
of individual arrays in an algorithm must be statically determined.
Support for sparse array storage is still experimental. 
The control flow primitives like `lax.while_loop`, `lax.fori_loop`
etc. require that the algorithm state flowing between iterations
must not change shape and size. 

These restrictions imply good amount of creativity and a very
disciplined coding style so that efficient JIT friendly 
solvers can be developed. This has been a key directive
in the development of `CR-Sparse`.


# Sparse signal processing problems and available solvers

We introduce the different problems that `CR-Sparse` library
can handle. 

## Sparse approximation and recovery problems

In the sparse approximation problems [@mallat2008wavelet; @elad2010sparse], we have a 
dictionary of atoms designed for a class of signals
such that the dictionary enables us to construct
a sparse representation of the signal. The
sparse and redundant representation model is:
$$
    x = \mathcal{D} \alpha + \eta
$$
where $x \in \mathbb{R}^M$ is a single from the given 
class of signals, $\mathcal{D} \in \mathbb{R}^{M \times N}$
is a dictionary consisting of $N$ atoms (column vectors) chosen
specifically for the class of signals, $\alpha$
is the sparse representation of $x$ in $\mathcal{D}$
giving us an approximation $\hat{x} = \mathcal{D} \alpha$
and $\eta$ is the approximation error. The
dictionary $\mathcal{D}$ is called the sparsifying dictionary.
The sparse approximation problem consists of finding
the best sparse $\alpha$ for a given $x$.


In the compressed sensing (CS) setting,
a sparse signal $x \in \mathbb{R}^N$ is captured 
through $M \ll N$ linear measurements which are
sufficient to recover $x$ from the measurements.
The model is given by:
$$
    y = \Phi x + e
$$
where $y \in \mathbb{R}^M$ is the vector of $M$ linear
measurements on $x$, $\Phi \in \mathbb{R}^{M \times N}$ 
is the sensing matrix [or measurement matrix] whose
rows represent the linear functionals on $x$, $x \in \mathbb{R}^N$
is the sparse signal being measured and $e$ is the measurement
noise. Typically, $x$ by itself is not sparse but it has
a sparse representation in a sparsifying basis $\Psi$
as $x = \Psi \alpha$. The model then becomes:
$$
    y = \Phi \Psi \alpha + e.
$$
Sparse recovery consists of finding $\alpha$ from
$y$ with minimum number of measurements possible.

Both sparse recovery and sparse approximation problems
can be addressed by same algorithms (though their 
performance analysis is different). To simplify the
notation, we will refer to $\mathcal{D}$ or $\Phi$ 
or $\Phi \Psi$ collectively as $A$ and attempt to
solve the under-determined system $y = A x + e$
with the prior on the solution that very few entries
in $x$ are non-zero. In general, we assume that
$A$ is full rank, unless otherwise specified.

The indices of non-zero
entry of $x$ form the support of $x$. Corresponding
columns in $A$ participate in the sparse
representation of $y$. We can call these columns
also as the support of $x$. 
$$
\mathop{\mathrm{supp}}(x) \triangleq \{i : x_i \neq 0 \}.
$$
Recovering the representation $x$
involves identifying its support $\Lambda = \mathop{\mathrm{supp}}(x)$
and identifying the non-zero entries over the support.
If the support has been 
correctly identified, a straight-forward
way to get the non-zero entries is to compute the
least squares solution $A_{\Lambda}^{\dag} y$.
The $\ell_0$ norm of $x$ denoted by $\| x\|_0$ 
is the number of non-zero entries in $x$.
A representation $y = A x$
is sparse if $\| x\|_0 \ll N$.
An algorithm which can
obtain such  a representation is called a *sparse coding
algorithm*.


### $\ell_0$ problems


The $K$-SPARSE approximation can be formally expressed as:
$$
\begin{aligned}
  & \underset{x}{\text{minimize}} 
  & &  \| y - A x \|_2 \\
  & \text{subject to}
  & &  \| x \|_0 \leq K.
\end{aligned}
$$
If the measurements are noiseless, we are interested in 
exact recovery. 
The $K$-EXACT-SPARSE approximation can be formally expressed as:
$$
\begin{aligned}
  & \underset{x}{\text{minimize}} 
  & &  \| x \|_0 \\
  & \text{subject to}
  & &  y = \Phi x\\
  & \text{and}
  & &  \| x \|_0 \leq K
\end{aligned}
$$

We need to discover both the sparse support for $x$ and
the non-zero values over this support. A greedy algorithm
attempts to guess the support incrementally and solves
a smaller (typically least squares) subproblem to estimate
the nonzero values on this support. It then computes the
residual $r = y - A x$ and analyzes the correlation of $r$
with the atoms in $A$, via the vector $h = A^T r$, to
improve its guess for the support and update $x$ accordingly.

We provide JAX based implementations for the following algorithms:

* `cr.sparse.pursuit.omp`: Orthogonal Matching Pursuit (OMP) [@pati1993orthogonal;@tropp2004greed;@davenport2010analysis] 
* `cr.sparse.pursuit.cosamp`: Compressive Sampling Matching Pursuit (CoSaMP) [@needell2009cosamp]
* `cr.sparse.pursuit.sp`: Subspace Pursuit (SP) [@dai2009subspace]
* `cr.sparse.pursuit.iht`: Iterative Hard Thresholding and its normalized version (IHT, NIHT) [@blumensath2009iterative;@blumensath2010normalized]
* `cr.sparse.pursuit.htp`: Hard Thresholding Pursuit and its normalized version (HTP, NHTP) [@foucart2011recovering]

For details, see the online [documentation](https://cr-sparse.readthedocs.io/en/latest/source/pursuit.html).

### $\ell_1$ problems

We introduce the different $\ell_1$ minimization problems supported by the
`cr.sparse.cvx.admm` package.

The $\ell_0$ problems are not convex. Obtaining a global minimizer 
is not feasible (NP hard). One way around is to use convex relaxation
where a cost function is replaced by its convex version. 
For $\| x \|_0$, the closest convex function is $\| x \|_1$ 
or $\ell_1$ norm. With this, the exact-sparse recovery problem becomes
$$
{\min}_{x} \| x\|_{1} \; \text{s.t.} \, A x = b
$$
This problem is known as Basis Pursuit (BP) in literature. It can be shown 
that under appropriate conditions on $A$, the basis pursuit solution
coincides with the exact sparse solution. In general, $\ell_1$-norm
minimization problems tend to give sparse solutions.

If $x$ is sparse in an sparsifying basis $\Psi$ as $x  = \Psi \alpha$
(i.e. $\alpha$ is sparse rather than $x$), then we can adapt the
BP formulation as
$$
{\min}_{x} \| W x\|_{1} \; \text{s.t.} \, A x = b
$$
where $W = \Psi^T$ and $A$ is the sensing matrix $\Phi$.

Finally, in specific problems, different atoms of $\Psi$ may 
have different importance. In this case, the $\ell_1$ norm
may be adapted to reflect this importance by a non-negative weight vector $w$:
$$
\| \alpha \|_{w,1} = \sum_{i=1}^{N} w_i | \alpha_i |.
$$
This is known as the weighted $\ell_1$ semi-norm.

This gives us the general form of the basis pursuit problem
$$
\tag{BP}
{\min}_{x} \| W x\|_{w,1} \; \text{s.t.} \, A x = b
$$

Usually, the measurement process introduces noise. Thus, 
a constraint $A x = b$ is too strict. We can relax this 
to allow for presence of noise as $\| A x - b \|_2 \leq \delta$
where $\delta$ is an upper bound on the norm of the measurement noise
or approximation error. 
This gives us the Basis Pursuit with Inequality Constraints (BPIC) problem:
$$
{\min}_{x} \| x\|_{1} \; \text{s.t.} \, \| A x - b \|_2 \leq \delta
$$
The more general form is the L1 minimization problem with L2 constraints:
$$
\tag{L1/L2con}
{\min}_{x} \| W x\|_{w,1} \; \text{s.t.} \, \| A x - b \|_2 \leq \delta
$$
The constrained BPIC problem can be transformed into an equivalent 
unconstrained convex problem:
$$
{\min}_{x} \| x\|_{1} + \frac{1}{2\rho}\| A x - b \|_2^2.
$$
This is known as Basis Pursuit Denoising (BPDN) in literature.
The more general form is the L1/L2 minimization:
$$
\tag{L1/L2}
{\min}_{x} \| W x\|_{w,1} + \frac{1}{2\rho}\| A x - b \|_2^2 
$$
We also support corresponding non-negative counter-parts.
The nonnegative basis pursuit problem:
$$
\tag{BP+}
{\min}_{x} \| W x\|_{w,1} \; \text{s.t.} \, A x = b \, \, \text{and} \, x \succeq 0
$$
The nonnegative L1/L2 minimization or basis pursuit denoising problem:
$$
\tag{L1/L2+}
{\min}_{x} \| W x\|_{w,1} + \frac{1}{2\rho}\| A x - b \|_2^2  \; \text{s.t.} \, x \succeq 0
$$
The nonnegative L1 minimization problem with L2 constraints:
$$
\tag{L1/L2con+}
{\min}_{x} \| W x\|_{w,1} \; \text{s.t.} \, \| A x - b \|_2 \leq \delta \, \, \text{and} \, x \succeq 0
$$
For details, see our online [tutorial](https://cr-sparse.readthedocs.io/en/latest/tutorials/admm_l1.html).

## Linear Operators

While major results on the recovery guarantees of sparse recovery algorithms focus on 
random matrices, actual applications prefer to use structured dictionaries and 
sensing matrices so that the operations $A x$ and $A^T x$ can be efficiently 
computed. These structured matrices fall under the more general class
of linear operators. In order to fully exploit the power of sparse 
recovery algorithms, it was deemed necessary to provide a complementary 
set of linear operators.

A linear operator $T : X \to Y$ connects a model space $X$ 
to a data space $Y$.

A linear operator satisfies following laws:
$$
    T (x + y) = T (x) + T (y)
$$
and
$$
    T (\alpha x) = \alpha T(x)
$$
Thus, for a general linear combination:
$$
    T (\alpha x + \beta y) = \alpha T (x) + \beta T (y)
$$
We are concerned with linear operators $T : \mathbb{F}^n \to \mathbb{F}^m$
where $\mathbb{F}$ is either the field of real numbers or 
complex numbers. 
$X = \mathbb{F}^n$ is the model space and 
$Y = \mathbb{F}^m$ is the data space.
Such a linear operator can be represented by a two dimensional matrix $A$.
The forward operation is given by:
$$
    y = A x.
$$
The corresponding adjoint operation is given by:
$$
    \hat{x} = A^H y
$$
We represent a linear operator by a pair of functions `times` and `trans`. 
The `times` function implements the forward operation while the `trans`
function implements the adjoint operation.

An inverse problem consists of computing $x$ given $y$ and $A$.

We provide a large collection of linear operators in `cr.sparse.lop`.

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

It is definitely quite easy to extend the library with new operators.

For details, see the online [documentation](https://cr-sparse.readthedocs.io/en/latest/source/lop.html).

## Sparse Subspace Clustering Problem

Consider a dataset of $S$ points in the ambient data space
$\mathbb{R}^M$ which have been assembled in a matrix $Y$ of shape $M \times S$.

In many applications, it often occurs that
if we *group* or *segment* the data set $Y$ into
multiple disjoint subsets (clusters): 
$Y = Y_1 \cup \dots \cup Y_K$,
then each subset can be modeled sufficiently well by a low dimensional subspace
$\mathbb{R}^D$ where $D \ll M$.
Some of the applications include:
motion segmentation [@tomasi1991detection; @tomasi1992shape; 
@boult1991factorization;
@poelman1997paraperspective;
@gear1998multibody;
@costeira1998multibody;
@kanatani2001motion], 
face clustering [@basri2003lambertian; ho2003clustering; lee2005acquiring]
and handwritten digit recognition [@zhang2012hybrid].

\emph{Subspace clustering} is a clustering framework which assumes
that the data-set can be segmented into clusters where points in
different clusters are drawn from different subspaces. Subspace clustering
algorithms are able to simultaneously segment the data into 
clusters corresponding to different subspaces as well as estimate
the subspaces from the data itself.
A comprehensive review of subspace clustering can be found in 
[@vidal2010tutorial].
Several state of the art algorithms are based on building
subspace preserving representations of individual data points
by treating the data set itself as a (self expressive) dictionary.
For creating subspace preserving representations, one resorts to
using sparse coding algorithms developed in sparse representations and 
compressive sensing literature. 

Two common algorithms are
*Sparse Subspace Clustering using $\ell_1$ regularization*
(SSC-$\ell_1$)[@elhamifar2009sparse; @elhamifar2013sparse]  
and *Sparse Subspace Clustering using Orthogonal
Matching Pursuit* (SSC-OMP) [@dyer2013greedy; @you2015sparse; @you2016scalable]. 
While SSC-$\ell_1$ is guaranteed to give correct clustering under
broad conditions (arbitrary subspaces and corrupted data), it
requires solving a large scale convex optimization problem. On
the other hand, SSC-OMP 
is computationally efficient but its clustering accuracy is
poor (especially at low density of data points per subspace).

The dataset $Y$ is modeled as being sampled from a collection
or arrangement $\mathcal{U}$ of linear (or affine) subspaces
$\mathcal{U}_k \subset \mathbb{R}^M$ : 
$\mathcal{U} = \{ \mathcal{U}_1  , \dots , \mathcal{U}_K \}$. 
The union of the subspaces
is denoted as
$Z_{\mathcal{U}} = \mathcal{U}_1 \cup \dots \cup \mathcal{U}_K$.

Let the data set be $\{ y_j  \in \mathbb{R}^M \}_{j=1}^S$
drawn from the union of subspaces under consideration.
$S$ is the total number of data points being analyzed
simultaneously.
We put the data points together in a *data matrix* as
$$
Y  \triangleq \begin{bmatrix}
y_1 & \dots & y_S
\end{bmatrix}.
$$
Let the vectors be drawn from a set of $K$ (linear or affine) subspaces, 
The subspaces are indexed by a variable $k$ with $1 \leq k \leq K$.
The $k$-th subspace is denoted by $\mathcal{U}_k$. 
Let the (linear or affine) dimension
of $k$-th subspace be $\dim(\mathcal{U}_k) = D_k$ with $D_k \leq D \ll M$.

The vectors in $Y$ can be grouped (or segmented or clustered) 
as submatrices 
$Y_1, Y_2, \dots, Y_K$ such 
that all vectors in $Y_k$ are drawn from the subspace $\mathcal{U}_k$. 
Thus, we can write
$$
Y^* = Y \Gamma = \begin{bmatrix} y_1 & \dots & y_S \end{bmatrix} 
\Gamma
= \begin{bmatrix} Y_1 & \dots & Y_K \end{bmatrix} 
$$
where $\Gamma$ is an $S \times S$ unknown permutation
matrix placing each vector to the right subspace. 

Let there be $S_k$ vectors in $Y_k$ with
$S = S_1 + \dots + S_K$. 
Let $Q_k$ be an orthonormal basis for subspace $\mathcal{U}_k$. Then,
the subspaces can be described as 
$$
\mathcal{U}_k = \{ y \in \mathbb{R}^M : y = \mu_k + Q_k \alpha \}, \quad 1 \leq k \leq K 
$$
For linear subspaces, $\mu_k = 0$.

A dataset where each point can be expressed as a linear combination
of other points in the dataset is said to satisfy 
*self-expressiveness property*. The self-expressive 
representation of a point $y_s$ in $Y$ is given by 
$$
y_s = Y c_s, \; c_{ss} = 0, \text{ or } Y = Y C, \quad \text{diag}(C) = 0
$$
where $C = \begin{bmatrix}c_1, \dots, c_S \end{bmatrix} \in \mathbb{R}^{S \times S}$ 
is the matrix of representation coefficients. 

Let $y_s$ belong to $k$-th subspace $\mathcal{U}_k$. 
Let $Y^{-s}$ denote the dataset $Y$ excluding the point $y_s$ 
and  $Y_k^{-s}$ denote the
set of points in $Y_k$ excluding $y_s$. If $Y_k^{-s}$ spans the subspace
$\mathcal{U}_k$, then a representation of $y_s$ can be constructed entirely
from the points in $Y_k^{-s}$. A representation is called 
*subspace preserving* if it consists of points within the same subspace.

If $c_i$ is a subspace preserving representation of $y_i$ and $y_j$
belongs to a different subspace, then $c_{ij} = 0$. Thus, if $C$ consists
entirely of subspace preserving representations, then $C_{ij} = 0$ whenever
$y_i$ and $y_j$ belong to different subspaces. 
In other words, if $Y_{-k}$ denotes the set of points from 
all subspaces excluding the subspace $Y_k$ corresponding
to the point $y_i$, then points in $Y_{-k}$ do not
participate in the representation $c_i$.

In the `cr.sparse.cluster.ssc` package, we provide a version of
OMP which can be used to construct the sparse self expressive representations 
$C$ of $Y$. Once the representation has been constructed, we compute an
affinity matrix $W = |C| + |C^T|$. 

We then apply spectral clustering on $W$ to complete SSC-OMP. 
For this, we have written a JAX version of spectral clustering
in `cr.sparse.cluster.spectral` package. In particular, it
uses our own Lanczos Bidiagonalization with Partial Orthogonalization (LANBPRO)
algorithm to compute the $K$ largest singular values of the
normalized affinity matrix
in as few iterations as possible. The intermediate variables 
$C$, $W$ are maintained in the experimental sparse matrices
stored in BCOO format.
The LANBPRO algorithm also works on sparse matrices directly. 
Thus, even though $C$ is of size $S \times S$, it can be stored 
efficiently in $O(DS)$ storage. This enables us to process 
hundreds of thousands of points efficiently. 


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
| HAAR DWT2, level=8 | input=(4000,4000) output=(4096, 4096) | 981 ms | 34.4 ms | 28.5x | 713 ms | 60.8 ms | 11.7x | 

\normalsize

Notebooks for reproducing these micro-benchmarks are also available in the 
[companion](https://github.com/carnotresearch/cr-sparse-companion/tree/main/comparison/pylops) repository.

We would like to mention that for small sizes, `pylops` on CPU runs much quicker. 
Benefits of `CR-Sparse` on GPU can be seen on large sizes.

# Acknowledgements

Shailesh would like to thank his Ph.D. supervisors Prof. Surendra 
Prasad and Prof. Brejesh Lall to inculcate his interest in this area
and support him over the years in his exploration. He would also
like to thank his employers, Interra Systems Inc., for allowing him
to pursue his research interests along with his day job.


# References


# Change Log
All notable changes to this project will be documented in this file.

* This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
* The format of this log is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

[Documentation](https://cr-sparse.readthedocs.io/en/latest/)

## [0.4.0] - 2023-10-15

[Documentation](https://cr-sparse.readthedocs.io/en/v0.4.0/)

This is a maintenance release to align the library with `JAX` 0.4.x
versions.

## [0.3.2] - 2022-10-08

[Documentation](https://cr-sparse.readthedocs.io/en/v0.3.2/)


### Added


Dictionaries

- Grassmannian frames
- 

Linear operators

- windowed_op

Sparse Recovery algorithms

- FOCUSS
- SPGL1 (Spectral Projected Gradient L1)

Optimization: Smooth functions

- smooth_quad_error 

Optimization algorithms

- Spectral projected gradient

Test Problems

- New test problems module introduced
- heavi-sine:fourier:heavi-side
- blocks:haar
- cosine-spikes:dirac-dct
- complex:sinusoid-spikes:dirac-fourier
- cosine-spikes:dirac-dct:gaussian
- piecewise-cubic-poly:daubechies:gaussian
- signed-spikes:dirac:gaussian
- complex:signed-spikes:dirac:gaussian
- blocks:heavi-side
- blocks:normalized-heavi-side
- gaussian-spikes:dirac:gaussian
- src-sep-1



Examples

- Matching pursuit demo
- Grassmannian frames demo
- Several examples based on the test problems

Documentation

- Thinking in JAX tutorial added
- Quick start expanded
- Test problems documentation linked with examples

### Changed

- Matching pursuit implementation revamped

### Fixed

- Handling of complex signals in Subspace Pursuit
- Handling of complex signals in Compressive Sampling Matching Pursuit

### Improved

- Support change condition added in convergence criteria for Subspace Pursuit
- order attribute in reshape linear operator


### Removed

- `cr.sparse.io` moved to `cr-nimble` project

## [0.3.1] - 2022-09-10

[Documentation](https://cr-sparse.readthedocs.io/en/v0.3.1/)

### Added

Block Sparse Bayesian Learning

- Expectation Maximization version
- Bound Optimization version

Dictionaries

- sparse_binary_mtx
- wavelet_basis


Linear operators

- sparse_real_matrix
- sparse_binary_dict


Test data

- sparse_normal_blocks

Miscellaneous

- Some plotting utilities


Examples

- ECG Data Compressive Sensing
- Block Sparse Bayesian Learning
- Sparse Binary Sensing Matrices


### Improvements

- Fixed some issues in the circulant linear operator
- Removed unnecessary `__init__` files
- Removed `bio` module which was empty
- Resolved some warnings related to incorrect static argument names or numbers
- Added `__str__` to several named tuples for debugging purposes
- ADMM tutorial updated to align with 0.3.x changes
- Added `length` and `x` properties in RecoverySolution
- 



### Others

- Changed copyright to CR-Suite Development Team





## [0.3.0] - 2022-08-27

[Documentation](https://cr-sparse.readthedocs.io/en/v0.3.0/)

### Added


Indicators

- zero
- singleton
- affine
- box
- box affine
- conic
- l2 ball
- l1 ball

Projectors

- zero
- identity
- singleton
- affine
- box
- conic
- l2 ball
- l1 ball



Proximal operators

- zero
- 1l
- l2
- l1 positive
- l1 ball
- ordered weighted l1

Smooth functions

- constant
- entropy
- huber
- linear
- logdet
- quad matrix


First order methods

- l1 regulated least square
- smooth conic dual solver
- lasso
- ordered weighted l1 regularized least squares
- dantzig smooth conic dual
- basis pursuit smooth conic dual


### Removed

- Common utility functions, linear algebra routines
  and basic signal processing functions have been refactored into a separate
  library [CR-Nimble](https://github.com/carnotresearch/cr-nimble)
- Wavelets related functionality has been refactored into a separate
  library [CR-Wavelets](https://github.com/carnotresearch/cr-wavelets)


### Misc

- Support for JAX 0.3.14


## [0.2.2] - 2021-12-02

[Documentation](https://cr-sparse.readthedocs.io/en/v0.2.2/)

### Improved

- Documentation
  - Introduction page revised
  - API docs improved
  - README revised
  - Algorithm page added
  - Quick start page revised

### Added

- JOSS Paper added and revised based on feedback from reviewers
- Linear Operators
  - input_shape, output_shape attributes introduced
  - fft, dot, norm estimate, reshape, scalar_mult, total variation, 


## [0.2.1] - 2021-11-01

[Documentation](https://cr-sparse.readthedocs.io/en/v0.2.1/)

### Added

- Compressive Sensing
  - 1 bit compressive sensing process
  - BIHT (Binary Iterative Hard Thresholding) algorithm for signal reconstruction from 1 bit measurements

## [0.2.0] - 2021-10-30

[Documentation](https://cr-sparse.readthedocs.io/en/v0.2.0/)

### Added

- Linear Operators
  - Convolution 1D, 2D, ND
  - Gram and Frame operators for a given linear operator
  - DWT 1D operator
  - DWT 2D operator
  - Block diagonal operator (by combining one or more operators)
- Sparse Linear Systems
  - Power iterations for computing the largest eigen value of a symmetric linear operator
  - LSQR solver for least squares problems with support for N-D data
  - ISTA: Iterative Shrinkage and Thresholding Algorithm
  - FISTA: Fast Iterative Shrinkage and Thresholding Algorithm
  - lanbpro, simple lansvd
- Geophysics
  - Ricker wavelet
  - Hard, soft and half thresholding operators for ND arrays (both absolute and percentile thresholds)
- Image Processing
  - Gaussian kernels
- Examples
  - Deconvolution
  - Image Deblurring
- Data generation
  - Random subspaces, uniform points on subspaces
  - two_subspaces_at_angle, three_subspaces_at_angle
  - multiple index_sets
  - sparse signals with bi-uniform non-zero values 
- Utilities
  - More functions for ND-arrays
  - Off diagonal elements in a matrix, min,max, mean
  - set_diagonal, abs_max_idx_cw, abs_max_idx_rw
- Linear Algebra
  - orth, row_space, null_space, left_null_space, effective_rank
  - subspaces: principal angles, is_in_subspace, project_to_subspace
  - mult_with_submatrix, solve_on_submatrix
  - lanbpro, simple lansvd
- Clustering
  - K-means clustering
  - Spectral clustering
  - Clustering error metrics
- Subspace clustering
  - OMP for sparse subspace clustering
  - Subspace preservation ratio metrics 

A paper is being prepared for JOSS.

### Improved

- Linear Operators
  - Ability to apply a 1D linear operator along a specific axis of input data
  - axis parameter added to various compressive sensing operators
- Code coverage
  - It is back to 90+% in the unit tests



## [0.1.6] - 2021-08-29

[Documentation](https://cr-sparse.readthedocs.io/en/v0.1.6/)

### Added

Wavelets
- CWT implementation based on PyWavelets: CMOR and MEXH
- integrate_wavelet, central_frequency, scale2frequency

Examples
- CoSaMP step by step
- Chirp CWT with Mexican Hat Wavelet
- Frequency Change Detection using DWT
- Cameraman Wavelet Decomposition


### Changed

Wavelets
- CWT API has been revised a bit.

### Updated

Examples
- Sparse recovery via ADMM

Signal Processing
- frequency_spectrum, power_spectrum

## [0.1.5] - 2021-08-22

[Documentation](https://cr-sparse.readthedocs.io/en/v0.1.5/)

### Added

Linear Operators
- Orthogonal basis operators: Cosine, Walsh Hadamard
- General Operators: FIR, Circulant, First Derivative, Second Derivative, Running average
- Operators: Partial Op
- DOT TEST for linear operators added.

Convex optimization algorithms
- Sparsifying basis support in yall1
- TNIPM (Truncated Newton Interior Points Method) implemented.

Wavelets
- Forward DWT
- Inverse DWT
- Padding modes: symmetric, reflect, constant, zero, periodic, periodization
- Wavelet families: HAAR, DB, MEYER, SYMMETRIC, COIFLET
- Full DWT/IDWT for periodization mode
- Filtering with Upsampling/Downsampling
- Quadrature Mirror Filters
- Forward and inverse DWT along a specific axis
- 2D Forward and inverse DWT for images
- Print wavelet info
- wavelist
- families
- build_wavelet
- wavefun
- CWT for Morlet and Ricker wavelets

Benchmarking
- Introduced airspeed velocity based benchmarks

General stuff
- Examples gallery introduced
- Unit test coverage is now back to 90%
- Documentation has been setup at ReadTheDocs.org also https://cr-sparse.readthedocs.io/en/latest/


## [0.1.4] - 2021-07-12

[Documentation](https://cr-sparse.readthedocs.io/en/v0.1.4/)

### Added

- A framework for linear operators
- ADMM based algorithms for l1 minimization
- Several greedy algorithms updated to support linear operators as well as plain matrices
- Hard thresholding pursuit added

## [0.1.3] - 2021-06-06
### Added
- Subspace Pursuit
- Iterative Hard Thresholding

## [0.1.0] - 2021-06-05

Initial release

[Unreleased]: https://github.com/carnotresearch/cr-sparse/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/carnotresearch/cr-sparse/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/carnotresearch/cr-sparse/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/carnotresearch/cr-sparse/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/carnotresearch/cr-sparse/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/carnotresearch/cr-sparse/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/carnotresearch/cr-sparse/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/carnotresearch/cr-sparse/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/carnotresearch/cr-sparse/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/carnotresearch/cr-sparse/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/carnotresearch/cr-sparse/compare/0.1.3...v0.1.4
[0.1.3]: https://github.com/carnotresearch/cr-sparse/compare/v0.1...0.1.3
[0.1.0]: https://github.com/carnotresearch/cr-sparse/releases/tag/v0.1
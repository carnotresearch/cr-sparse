# Change Log
All notable changes to this project will be documented in this file.

* This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
* The format of this log is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

[Documentation](https://cr-sparse.readthedocs.io/en/latest/)

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

# Version comparisons

[Unreleased]: https://github.com/carnotresearch/cr-sparse/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/carnotresearch/cr-sparse/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/carnotresearch/cr-sparse/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/carnotresearch/cr-sparse/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/carnotresearch/cr-sparse/compare/0.1.3...v0.1.4
[0.1.3]: https://github.com/carnotresearch/cr-sparse/compare/v0.1...0.1.3
[0.1.0]: https://github.com/carnotresearch/cr-sparse/releases/tag/v0.1
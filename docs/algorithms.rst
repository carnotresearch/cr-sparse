Algorithms
=======================================

This section lists and organizes available and planned algorithms in the `CR-Sparse` package.


Sparse recovery algorithms
--------------------------------------------

.. rubric:: Convex relaxation algorithms

.. list-table::
    :widths: 70 10 10 10
    :header-rows: 1

    * - Algorithm
      - Acronym
      - Status
      - Docs
    * - Truncated Newton Interior Points Method
      - L1LS
      - done
      - :ref:`... <api:l1min:tnipm>`
    * - Basis Pursuit using ADMM
      - BP
      - done
      - :ref:`... <api:l1min:admmm>`
    * - Basis Pursuit Denoising using ADMM
      - BPDN
      - done
      - :ref:`... <api:l1min:admmm>`
    * - Basis Pursuit with Inequality Constraints using ADMM
      - BPIC
      - done
      - :ref:`... <api:l1min:admmm>`

.. rubric:: Greedy pursuit algorithms

.. list-table::
    :widths: 70 10 10 10
    :header-rows: 1

    * - Algorithm
      - Acronym
      - Status
      - Docs
    * - Orthogonal Matching Pursuit
      - OMP
      - done
      - :ref:`... <api:pursuit:matching>`
    * - Compressive Sampling Matching Pursuit
      - CoSaMP
      - done
      - :ref:`... <api:pursuit:matching>`
    * - Subspace Pursuit
      - CoSaMP
      - done
      - :ref:`... <api:pursuit:matching>`

.. rubric:: Shrinkage and thresholding algorithms

.. list-table::
    :widths: 70 10 10 10
    :header-rows: 1

    * - Algorithm
      - Acronym
      - Status
      - Docs
    * - Iterative Hard Thresholding
      - IHT
      - done
      - :ref:`... <api:pursuit:ht>`
    * - Normalized Iterative Hard Thresholding
      - NIHT
      - done
      - :ref:`... <api:pursuit:ht>`
    * - Hard Thresholding Pursuit
      - HTP
      - done
      - :ref:`... <api:pursuit:ht>`
    * - Normalized Hard Thresholding Pursuit
      - NHTP
      - done
      - :ref:`... <api:pursuit:ht>`

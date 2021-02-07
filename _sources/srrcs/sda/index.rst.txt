Fully Convolutional Stacked Denoising Autoencoders
=======================================================

.. contents::
    :depth: 3
    :local:

:cite:`mousavi2015deep` presents a deep learning
based framework for sensing and recovering structured
signals. This article builds on the ideas developed
in it and presents a fully convolutional auto-encoder
architecture for the same.

Compressive Sensing Framework
---------------------------------

.. image:: ../../diagrams/cs/cs.png

We consider a set of signals :math:`x \in \RR^N` from 
a specific domain (e.g. images). 

In compressive sensing, a number of random 
measurements are taken over the signal mapping 
from the signal space :math:`\RR^N` to a 
measurement space :math:`\RR^M` via a mapping

.. math::
    
    y = \mathbf{\Gamma}(x)

In general, this mapping from signal space to measurement space
can be either linear or non-linear. A linear mapping
is typically represented via a sensing matrix :math:`\BPhi`
as 

.. math::

    y  = \BPhi x

Compressive sensing is a field that focuses on solving 
the inverse problem of recovering the signal :math:`x`
from the linear measurements :math:`y`.
This is generally possible if :math:`x` has a sparse
representation in some basis :math:`\BPsi`
such that

.. math::

    x = \BPsi \alpha

where :math:`\alpha` has only :math:`K \ll N` non-zero entries.

Under these conditions, a small number of linear measurements
:math:`M \ll N` is sufficient to recover the
original signal :math:`x`.

The basis :math:`\BPsi` in which the signal has a sparse (or compressive)
representation is domain specific. Some popular bases include:

* Wavelets
* Frames
* Dictionaries (like multiple orthonormal bases)
* Dictionaries learnt from data

Sparse recovery is the process of recovering the
sparse representation :math:`\alpha` from the measurements
:math:`y` given that the sparsifying basis
:math:`\BPsi` and the sensing matrix :math:`\BPhi` are
known. This is represented by the step:

.. math::

    \widehat{\alpha} = \Delta_r(\mathbf{\Phi} \mathbf{\Psi}, y )

in the diagram above. Typical recovery algorithms include:

* Convex optimization based routines like basis pursuit
* Greedy algorithms like OMP, CoSaMP, IHT




Stacked Denoising Autoencoder
-----------------------------------

:cite:`mousavi2015deep` considers how deep learning ideas can
be used to develop a recovery algorithm from compressed measurements
of  a signal. 

In particular, it is not necessary to choose a specific
sparsifying basis for the recovery of signals. It is enough
to know that the signals are compressible in some basis 
and a suitable recovery algorithm can be learnt directly 
from the data in the form of a neural network.

.. image:: ../../diagrams/cs/recovery_in_signal_space.png

The figure above represents the recovery directly from measurement
space to the signal space.

Deep learning architectures can be constructed for following
scenarios:

* Recovery of the signal from fixed linear measurements 
  (using random sensing matrices)
* Recovery of the signal from nonlinear adaptive compressive 
  measurements

While in the first scenario, the sensing matrix :math:`\BPhi`
is fixed and known apriori, in the second scenario, the
sensing mapping :math:`\mathbf{\Gamma}` is also learned during the
training process.

The neural network architecture ideally suited for solving
this kind of recovery problem is a stacked denoising autoencoder (SDA).


SDA + Linear Measurements
''''''''''''''''''''''''''

.. image:: ../../diagrams/cs/sda/sda_from_linear_measurements.png

The diagram above shows a four layer Stacked Denoising Autoencoder (SDA) 
for recovering signals from their linear measurements. The 
first layer is essentially a sensing matrix (no nonlinearity added).
The following three layers form a neural network for which:

* The input is the linear measurements :math:`y`.
* The output is the reconstruction of the  :math:`\hat{x}` of the 
  original signal.

In other words:

* The first layer is the encoder
* The following three layers are the decoder

Each layer in the decoder is a fully connected
layer that implements an affine transformation
followed by a nonlinearity.

The functions of three layers in the decoder are
described below.

.. rubric:: Layer 1 (input :math:`\RR^M`, output :math:`\RR^N`)

.. math::

    x_{h_1} = \mathcal{T}(\mathbf{W}_1 y + \mathbf{b}_1)

:math:`\mathbf{W}_1 \in \RR^{N \times M}` 
and :math:`\mathbf{b}_1 \in \RR^N` are the weight
matrix and bias vector for the first decoding layer.


.. rubric:: Layer 2 (input :math:`\RR^N`, output :math:`\RR^N`)

.. math::

    x_{h_2} = \mathcal{T}(\mathbf{W}_2 x_{h_1} + \mathbf{b}_2)

:math:`\mathbf{W}_2 \in \RR^{M \times N}` 
and :math:`\mathbf{b}_2 \in \RR^M` are the weight
matrix and bias vector for the second decoding layer.



.. rubric:: Layer 3 (input :math:`\RR^M`, output :math:`\RR^N`)

.. math::

    \widehat{x} = \mathcal{T}(\mathbf{W}_3 x_{h_2} + \mathbf{b}_3)

:math:`\mathbf{W}_3 \in \RR^{N \times M}` 
and :math:`\mathbf{b}_3 \in \RR^N` are the weight
matrix and bias vector for the third and final decoding layer.


The set of parameters to be trained in this network is given
by:

.. math::

    \Omega = \{\mathbf{W}_1, \mathbf{b}_1, 
    \mathbf{W}_2, \mathbf{b}_2, 
    \mathbf{W}_3, \mathbf{b}_3, \}


Fully Convolutional Stacked Denoising Autoencoder
----------------------------------------------------

1x1 Convolutions
'''''''''''''''''''''

.. image:: ../../diagrams/cnn/1x1/channel_reduction.png



The AutoEncoder architecture
''''''''''''''''''''''''''''''''''''

.. image:: ../../diagrams/cs/sda/cs_sda_cnn.png




References 
---------------

.. bibliography::
   :filter: docname in docnames
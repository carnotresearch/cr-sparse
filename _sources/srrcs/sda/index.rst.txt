Fully Convolutional Stacked Denoising Autoencoders
=======================================================

.. contents::
    :depth: 3
    :local:

:cite:`mousavi2015deep` presents a deep learning
based framework for sensing and recovering structured
signals. This work builds on the ideas developed
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


Working with Images
--------------------------

SDA layers are fully connected layers. Hence, the
input layer has to be connected to all pixels in
an image. This is computationally infeasible for
large images.

The standard practice is to divide image into 
small patches and vectorize each patch. Then,
the network can process one patch at a time
(for encoding and decoding).

:cite:`mousavi2015deep` trained their SDA
for :math:`32 \times 32` patches of 
grayscale images. Working with patches leads
to some blockiness artifact in the reconstruction.
The authors suggest using overlapped patches 
during sensing and averaging the reconstructions
to avoid blockiness.


In the following, we discuss how SDA can be
developed as a network consisting solely of
convolutional layers.

Fully Convolutional Stacked Denoising Autoencoder
----------------------------------------------------

.. rubric:: Input

We use Caltech-UCSD Birds-200-2011 dataset :cite:`wang2008subspace` for our training.

* We work with color images. 
* For training, we work with randomly selected subset of images.
* We pick the center crop of size :math:`256 \times 256` from 
  these images. 
* If an image has a smaller size, it is resized first preserving
  the aspect ratio and then the center part of :math:`256 \times 256`
  is cropped.
* Image pixels are mapped to the range :math:`[0, 255]`.
* During training, batches of 32 images are fed to the network.


.. rubric:: Linear measurements

It is possible to implement patch-wise compressive sampling
:math:`y = \BPhi x` using a convolutional layer. 

* Consider patches of size :math:`N = n \times n \times 3`.
* Use a convolutional kernel with kernel size :math:`n \times n`.
* Use a stride of :math:`n \times n`.
* Don't use any bias.
* Don't use any activation function (i.e. linear activation).
* Use :math:`M` such kernels.

What is happening? 

* Each kernel is a row of the sensing matrix :math:`\BPhi`
* Each kernel is applied on a volume of size :math:`N = n \times n \times 3` to generate a single value.
* In effect it is an inner product of one row of :math:`\BPhi`, with
  one (linearized) patch of the input image.
* The stride of :math:`n \times n` ensures that the kernel 
  is applied on non-overlapping patches of the input image.
* :math:`M` separate kernels are :math:`M` rows of the sensing
  matrix :math:`\BPhi`.
* Let :math:`b = 256 / n`.
* Then, the number of patches in the image is :math:`b \times b`.
* Each input patch gets mapped to a single pixel on each output channel.
* Thus, each depth vector (across all channels) is a measurement vector
  for each input patch.


.. rubric:: 1x1 Convolutions for decoder layer 1 and 2

Since, each image patch is represented by a depth vector
in the input tensor to the decoder, we need a way
to map such a vector to another vector as per the FC
layers in the SDA. This can be easily achieved by 1x1 convolutions.

.. image:: ../../diagrams/cnn/1x1/channel_reduction.png


.. rubric:: Transposed convolution for the final decoder layer

Final challenge is to take the depth vectors for individual 
image patches and map them back into regular image patches 
with 3 channels.

A transposed convolution layer with identical kernel size
and stride as the encoding layer can achieve this job.

The Fully Convolutional SDA architecture
'''''''''''''''''''''''''''''''''''''''''

The figure below presents the architecture of the fully 
convolutional stacked denoising autoencoder. 


.. image:: ../../diagrams/cs/sda/cs_sda_cnn.png


There are few differences from the approach taken in :cite:`mousavi2015deep`.

* We use ReLU activations in decoder layers 1 and 2.
* The final decoder layer uses sigmoid activation to ensure
  that the output remains clipped between 0 and 1.
* We have added batch normalization after layer 1 and 2 of the
  decoder. 

While this architecture doesn't address the blockiness issue,
it can probably be addressed easily by adding one more convolutional
layer after the decoder.

References 
---------------

.. bibliography::
   :filter: docname in docnames
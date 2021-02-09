Convolution Basics
===================

The discussion in this section is primary based on
:cite:`dumoulin2016guide`.

Some prior familiarity with convolutional neural networks (CNNs)
is assumed. 

Convolutional neural networks are generally used for
working with images although they have found in other
domains too. 

A CNN consists of different types of layers:

* convolutional layers
* pooling layers
* batch normalization layers
* concatenation and reshaping layers
* fully connected layers

Input to a CNN is typically a 4 dimensional
tensor with shape (number of images x image  height x image  width x image  channels).

The number of images is the size of a single batch of images fed
to a CNN. The number of channels in an RGB image is 3, in a grayscale 
image is 1.

After passing through a convolutional layer, the image becomes
abstracted to a feature map with shape 
(number of images x feature map height x feature map width x feature map channels).

Following layers map input feature maps to output feature maps.

Let's assume that the number of images is just 1 and
focus on a 3 dimensional tensor of shape (height x width x channels).
The input is spread along three axes (height, width and channels).
The channels axis is also known as the depth axis.
The ordering of values in width and height dimensions is
critical for data interpretation. 
Each channel is also called a *feature map*.


.. rubric:: Discrete Convolutions 

A *discrete convolution* is a linear transformation that
preserves the notion of ordering in the width and height
axes of an image. It is sparse in the sense that only a few units 
of input volume contribute to a given output unit. 
It reuses parameters as same weights are applied for the
computation of different output units from corresponding
input units.

.. rubric:: 2-D convolution

Let's start with the simplest case of an input feature
map with size :math:`(x_h \times x_w)` and a 2D kernel
of size :math:`(k_h \times k_w)`.

.. math::

    y(r, c) = \sum_{i=0}^{k_h -1} \sum_{j=0}^{k_w -1} w(i, j) x(r+i, c+j)

A convolutional kernel 

* Let the shape of input tensor be given by
 
  * height: :math:`i_0`
  * width: :math:`i_1`
  * channels: :math:`i_2`

* Let the shape of the output tensor be given by

  * height: :math:`o_0`
  * width: :math:`o_1`
  * channels: :math:`o_2`


* Let the shape of the convolutional kernel be given by


 
References 
---------------

.. bibliography::
   :filter: docname in docnames 

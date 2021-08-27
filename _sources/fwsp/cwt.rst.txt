Continuous Wavelet Transform
==================================


Complex Morlet Wavelets
-------------------------------------

There are several definitions of Complex Morlet Wavelets.

.. math::

    \psi(t) = \frac{1}{\sqrt[4]{\pi}} e^{j \omega_0t } e^{\frac{-t^2}{2}}


Its Fourier transform is:

.. math::

    \Psi(s \omega) = \frac{1}{\sqrt[4]{\pi}} H(\omega) e^{\frac{-(s\omega - \omega_0)^2}{2}}


where :math:`H(\omega)` is the Heaviside step function.

Second definition is more general and is based on two parameters:

- Central frequency: :math:`C`
- Bandwidth: :math:`B` 


.. math::

    \psi(t,B, C) = \frac{1}{\sqrt{\pi B}} \ e^{\frac{-t^2}{B}} \ e^{j2 \pi C t}


This is Gaussian modulated by a complex sinusoid with the standard deviation:

.. math::

    \sigma = \sqrt{\frac{T_p}{2}}

However, this definition doesn't have unit energy.
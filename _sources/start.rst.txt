Getting Started
======================

.. highlight:: shell

We assume that you have a working Python 3 installation on your system.
Also, you should have GIT  available on command line. 


Installing as a package
----------------------------

Directly from our GITHUB repository::

    python -m pip install git+https://github.com/carnotresearch/cr-vision.git


Working with the source code in development mode
-----------------------------------------------------


Clone the repository::

    git clone https://github.com/carnotresearch/cr-vision.git


Change into the code::

    cd cr-vision


Ensure that the dependencies are installed::

    python -m pip install -r requirements.txt


Install the package in development mode::

    python -m pip install -e .


Examples
-----------------


Explore the examples directory::

    cd examples/basic


Run an example::

    python ex_add_logo.py



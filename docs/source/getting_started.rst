.. _getting-started:

Getting Started with Meent
==========================

Welcome to Meent!

Meent uses RCWA for Electromagnetic (EM) simulation and automatic-differentiation for back-propagation.

Tutorials
---------
:doc:`tutorials` will help you go through the key features.

Installation
---------------
Meent can be installed via ``pip``.

.. code-block:: python

   pip install meent

How to use
----------
You can simply select backend.

.. code-block:: python

    import meent

    # backend 0 = NumPy
    # backend 1 = JAX
    # backend 2 = PyTorch

    backend = 1
    mee = meent.call_mee(backend=backend, ...)

All the methods in `mee` are same across backends.

Backends
---------
Meent provides three libraries as a backend:

.. image:: _static/backends.png

* ``NumPy`` https://github.com/numpy/numpy

  * The fundamental package for scientific computing with Python
  * Easy and lean to use

* ``JAX`` https://github.com/google/jax

  * Autograd and XLA, brought together for high-performance machine learning research.

* ``PyTorch`` https://github.com/pytorch/pytorch

  * A Python package that provides two high-level features: Tensor computation with strong GPU acceleration and Deep neural networks built on a tape-based autograd system


.. table:: Truth table for "not"
   :align: center
   :widths: auto

   =====  =====
     A    not A
   =====  =====
   False  True
   True   False
   =====  =====


When to use
~~~~~~~~~~~

.. list-table:: Backend features
   :header-rows: 1
   :widths: 10 10 10 10 60

   * -
     - .. centered:: NumPy
     - .. centered:: JAX
     - .. centered:: PyTorch
     - .. centered:: Description
   * - .. centered:: 64bit
     - .. centered:: O
     - .. centered:: O
     - .. centered:: O
     - .. centered:: Default for scientific computing
   * - .. centered:: 32bit
     - .. centered:: O
     - .. centered:: O
     - .. centered:: O
     - .. centered:: 32bit data type operation [*]_
   * - .. centered:: GPU
     - .. centered:: X
     - .. centered:: O
     - .. centered:: O
     - .. centered:: except Eigendecomposition [*]_
   * - .. centered:: TPU
     - .. centered:: X
     - .. centered:: X
     - .. centered:: X
     - .. centered:: Not supported [*]_
   * - .. centered:: AD
     - .. centered:: X
     - .. centered:: O
     - .. centered:: O
     - .. centered:: Automatic Differentiation (Back Propagation)
   * - .. centered:: ``pmap``
     - .. centered:: X
     - .. centered:: O
     - .. centered:: X
     - .. centered:: Parallelization function in JAX

.. [*] In 32bit operation, operations on numbers of 8>= digit difference fail without warning or error.
    Use only when you do understand what you are doing.

.. [*] As of now(2023.03.19), GPU-native eigendecomposition is not implemented in JAX and PyTorch.
    It's enforced to run on CPUs and send back to GPUs.

.. [*] Currently there is no workaround to run codes on TPU, that includes eigendecomposition.

Numpy is simple and light to use. Suggested as a baseline with small ~ medium scale optics problem.
JAX and PyTorch is recommended for cases having large scale or optimization part.
If you want parallelized computing with multiple devices(e.g., GPUs), JAX is ready for that.
But since JAX does jit compilation, it takes much time at the first run.




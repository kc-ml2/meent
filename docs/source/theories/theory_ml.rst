.. _theories_ml:

Machine Learning
===================================

This contains background theory that is needed to understand how Meent works.

.. toctree::
   :maxdepth: 3
   :caption: EM Theories

   geometry
   fourier_analysis
   eigenmodes
   connecting_layers
   enhanced_tmm

RCWA is the sequence of the following processes: solving the Maxwell's equations, finding the eigenmodes of a layer and connecting these layers including the superstrate and substrate to calculate the diffraction efficiencies.
Precisely, the electromagnetic field and permittivity geometry are transformed from the real space to the Fourier space (also called the reciprocal space or k-space) by Fourier analysis. Maxwell's equations are then solved per layer through convolution operation, and a general solution of the field in each direction can be obtained. This general solution can be represented in terms of eigenmodes (eigenvectors) and eigenvalues with eigendecomposition, and used to calculate diffraction efficiencies by applying boundary conditions and connecting to adjacent layers.

This chapter provides a comprehensive explanation of the theories, formulations and implementations of :math:`\texttt{meent}` in the following sections:

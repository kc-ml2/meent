Fourier Analysis
================

Discrete Fourier Series (DFS) and Continuous Fourier Series (CFS)

In RCWA, the device geometry needs to be mapped to the Fourier space using Fourier analysis. To achieve this, the device is sliced into multi-layers so that each layer has Z-invariant (layer stacking direction) permittivity distribution. In other words, the permittivity can be considered as a piecewise-constant function that varies in X and Y but not Z direction in each layer. Then the geometry in real space can be expressed as a weighted sum of Fourier basis:

.. math::

    \begin{align}\label{eqn:fourier_series}
    \varepsilon(x, y) = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} c_{n, m} \cdot \exp{\left[j \cdot 2\pi \left(\frac{x}{\Lambda_x}m + \frac{y}{\Lambda_y}n\right)\right]},
    \end{align}

where $\Lambda_x, \Lambda_y$ are the period of the unit cell and $c_{n,m}$ is the Fourier coefficients ($m^{th}$ in X and $n^{th}$ in Y).
However, due to the limitation of the digital computations, this has to be approximated with truncation:

.. math::

    \begin{align}\label{eqn:fourier_series_truncation}
        \varepsilon(x, y) \simeq \sum_{m=-M}^{M} \sum_{n=-N}^{N} c_{n,m} \cdot \exp{\left[j \cdot 2\pi \left(\frac{x}{\Lambda_x}m + \frac{y}{\Lambda_y}n\right)\right]},
    \end{align}

where $M, N$ are the Fourier Truncation Order (FTO, the number of harmonics in use) in the X and Y direction, and these can be considered as hyperparamters that affects the simulation accuracy.

Here, $c_{n,m}$ is the permittivity distribution in the Fourier space which is our interest and can be found by one of these two methods:
% Based on this, two types of Fourier series can be employed:
Discrete Fourier Series (DFS) or Continuous Fourier Series (CFS). To be clear, CFS is Fourier series on piecewise-constant function (permittivity distribution in our case). This name was given to emphasize the characteristics of each type by using opposing words. The output array of DFS and CFS have the same shape and can be substituted for each other.

In DFS, the function $\varepsilon(x, y)$ to be transformed is sampled at a finite number of points, and this means it's given in matrix form with rows and columns, $\varepsilon_{\mathtt r,\mathtt c}$. The coefficients of DFS are then given by this equation:

.. math::

    \begin{equation} \label{eqn:dfs-coeffs}
    c_{n,m} = \frac{1}{P_xP_y}\sum_{\mathtt c=0}^{P_x-1}\sum_{\mathtt r=0}^{P_y-1}\varepsilon_{\mathtt r,\mathtt c} \cdot \exp{\left[-j \cdot 2\pi \left(\frac{m}{P_x} \mathtt c + \frac{n}{P_y} \mathtt r \right)\right]},
    \end{equation}

where $P_x, P_y$ are the sampling frequency (the size of the array), $\varepsilon_{\mathtt r,\mathtt c}$ is the ${(\mathtt r,\mathtt c)}^{th}$ element of the permittivity array.

There is an essential but easily overlooked fact: the sampling frequency ($P_x, P_y$) is very important in DFS \cite{TheSciEngDSP, alias, kreyszig2011advanced}. If this is not enough, an aliasing occurs: DFS cannot correctly capture the original signal (you can easily see the wheels of a running car in movies rotating in the opposite direction; this is also an aliasing and called the wagon-wheel effect).
In RCWA, this may occur during the process of sampling the permittivity distribution.
To resolve this, \texttt{meent} provides a scaling function by default - that is simply to increase the size of the permittivity array by repeatedly replicating the elements while keeping the original shape of the pattern.
This option improves the representation of the geometry in the Fourier space and results in more accurate RCWA simulations.

CFS utilizes the entire function to find the coefficients while DFS uses only some of them.
This means that CFS prevents potential information loss coming from the intrinsic nature of DFS, thereby enables more accurate simulation.
The Fourier coefficients can be expressed as follow:

.. math::

    \begin{align} \label{equ:fourier_coeff_CFS}
    c_{n,m} = \frac{1}{\Lambda_x\Lambda_y}\int_{x_0}^{x_0+\Lambda_x}\int_{y_0}^{y_0+\Lambda_y}\varepsilon(x,y) \cdot \exp{\left[-j \cdot 2\pi \left(\frac{m}{\Lambda_x}x + \frac{n}{\Lambda_y} y \right)\right]} dydx.
    \end{align}

The information that CFS needs are the points of discontinuity and the permittivity value in each area sectioned by those points, whereas DFS needs the whole permittivity array as in Figure \ref{fig:raster and vector}.

DFS and CFS have its own advantages and one can be chosen according to the purpose of the simulation. Basically, DFS is proper for Raster modeling since its operations are mainly on the pixels (array) and the input of the Raster modeling is the array. This is naturally connected to the pixel-wise operation (cell flipping) in metasurface freeform design.
CFS is suitable for Vector modeling because it deals with the graph (discontinuous points and length) of the objects and Vector modeling takes that graph as an input. Hence it enables direct and precise optimization of the design parameters (such as the width of a rectangle) without grid that severely limits the resolution. We will address this in section \ref{sec:Derivatives}
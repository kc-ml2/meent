Program Sequence
================

In this section, we will provide a detailed explanation of the functions in \texttt{meent}\footnote{for version 0.9.x} and discuss the simulation program sequence with examples.
Specifically, we will demonstrate an example code for simulating and optimizing the optical response of a silicon rectangular pillar in a periodic structure.

.. _initialization:
Initialization
--------------

A simple way to use \texttt{meent} is using `call\_mee()' function which returns an instance of Python class that includes all the functionalities of \texttt{meent}. Simulation conditions can be set by passing parameters as arguements (args) or keyword arguements (kwargs) in this function. It is also possible to change conditions after calling instance by directly assigning desired value to the property of the instance.

Methods to set simulation conditions[ccc]

.. code-block:: python

    # method 1: thickness setting in instance call
    mee = meent.call_mee(backend=backend, thickness=thickness, ...)

    # method 2: direct assignment
    mee = meent.call_mee(backend=backend, ...)
    mee.thickness = thickness

Here are the descriptions of the input parameters in \texttt{meent} class:

``backend`` : **integer**
    supports three backends: NumPy, JAX, and PyTorch.

    * 0: NumPy (RCWA only; AD is not supported).
    * 1: JAX.
    * 2: PyTorch.


``grating\_type``  **integer**
    This parameter defines the simulation space.
    
    * 0: 1D grating without conical incidence $(\phi = 0)$.
    * 1: 1D grating with conical incidence.
    * 2: 2D grating.
    
``pol``: **integer** or **float**
    This parameter controls the linear polarization state of the incident wave by this definition: $\psi = \pi / 2 * (1 - {\textit{pol}})$.
    It can take values between 0 and 1. 0 represents fully transverse electric (TE) polarization, and 1 represents fully transverse magnetic (TM) polarization. Support for other polarization states such as the circular polarization state which involves the phase difference between TE and TM polarization will be added in the future updates.

``n\_I`` : **float**
    The refractive index of the superstrate.
``n\_II`` : **float**
    The refractive index of the substrate.

``theta`` : **float**
    The angle of the incidence in radians.

``phi`` : **float**
    The angle of rotation (or azimuth angle) in radians.

``wavelength`` : **float**
    The wavelength of the incident light in vacuum. Future versions may support complex type wavelength.

``fourier\_order`` : **integer** or **list of integers**
    Fourier truncation order (FTO). This represents the number of Fourier harmonics in use. If \textit{fourier\_order} = $N$, this is for 1D grating and \texttt{meent} utilizes $(2N+1)$ harmonics spanning from $-N$ to $N$:$-N, -(N-1), ..., N$. For 2D gratings, it takes a sequence $[M,N]$ as an input, where $M$ and $N$ become FTO in $X$ and $Y$ directions, respectively. Note that 1D grating can also be simulated in 2D grating system by setting $N$ as $0$.

``period`` : **list of floats**
    The period of a unit cell. For 1D grating, it is a sequence with one element which is a period in X-direction. For 2D gratings, it takes a sequence [period in $X$, period in $Y$] as an input.

``type\_complex`` : **integer**
    The datatype used in the simulation.
    \begin{itemize}
        \item 0: complex128 (64 bit).
        \item 1: complex64 (32 bit).
    \end{itemize}

``device`` : **integer**
    The selection of the device for the calculations: currently CPU and GPU are supported. At the time of writing this paper, the eigendecomposition, which is the most expensive step as $\mathcal{O}(M^3N^3)$ where $M\text{ and } N$ are FTO, is available only on CPU. This means GPU may not as powerful as we expect as in deep learning regime.

    \begin{itemize}
        \item 0: CPU.
        \item 1: GPU.
        % \item 2: TPU. As of now\footnotemark[\value{footnote}], TPU is not supported since data transfer between TPU and CPU for eigendecomposition is not enabled.
    \end{itemize}

``fft\_type`` : **integer**
    This variable selects the type of Fourier series implementation. 0 and 1 are options for raster modeling and 2 is for vector modeling. 0 uses discrete Fourier series (DFS) while 1 and 2 use continuous Fourier series (CFS). Note that the name `fft\_type' may change since it is not correct expression.

    \begin{itemize}
        \item 0: DFS for the raster modeling (pixel-based geometry). \textit{fft\_type} supports \textit{improve\_dft} option, which is True by default, that can prevent aliasing by increasing sampling frequency, and drives the result to approach to the result of CFS.
        \item 1: CFS for the raster modeling (pixel-based geometry). This doesn't support backpropagation. Use this option for debugging or in RCWA-only situation.
        \item 2: CFS for the vector modeling (object-based geometry).
    \end{itemize}

``thickness`` : **list of floats**
    The sequence of the thickness of each layer from top to bottom.

``ucell`` : **array of \{floats, complex numbers\}, shape is (i, j, k)**
    The input for the raster modeling. It takes a 3D array in ($Z$,$Y$,$X$) order, where $Z$ represents the direction of the layer stacking. In case of 1D grating, j is 1 (e.g., shape = (3,1,10) for a stack composed of 3 layers that are 1D grating).

Geometry Modeling
----------------
`meent` provides two types of geometry modeling methods: vector and raster.

Vector Modeling
~~~~~~~~~~~~~~~

Figure \ref{fig:rot_rect} shows rotated rectangles drawn on XY plane. \texttt{meent} decomposes the geometrical figures into the collection of sub-rectangles which of each side lies on the direction of either $\hat x$ or $\hat y$. Then CFS with the sinc function is used to find the Fourier coefficients. The degree of approximation can be determined by `n\_split' option in Code \ref{code:vector}.

To add primitives to the simulation space, users can utilize `rectangle()' or `rectangle_rotation()' functions
which allows the insertion of desired geometry. The `draw()' function is then employed to create the
final structure, taking into account any potential overlaps between the geometries. Code \ref{code:vector}
is the example creating a layer that has rotated rectangle.

% By controlling `n\_split' option in the code, user can decide the degree of approximation as in Figure \ref{fig:rot_rect}.

.. code-block:: python

    thickness = [300.]
    length_x = 100
    length_y = 300
    center = [300, 500]
    n_index_1 = 3.48
    n_index_2 = 1
    base_n_index_of_layer = n_index_2
    angle = 35 * torch.pi / 180
    n_split = [5, 5]  # degree of approximation

    length_x = torch.tensor(length_x, dtype=torch.float64, requires_grad=True)
    length_y = torch.tensor(length_y, dtype=torch.float64, requires_grad=True)
    thickness = torch.tensor(thickness, requires_grad=True)
    angle = torch.tensor(angle, requires_grad=True)

    obj_list = mee.rectangle_rotate(*center, length_x, length_y, *n_split, n_index_1, angle)
    layer_info_list = [[base_n_index_of_layer, obj_list]]
    mee.draw(layer_info_list)

|pic1| |pic2|

.. |pic1| image:: images/rot_rect_1_1.png
   :width: 49%

.. |pic2| image:: images/rot_rect_20_20.png
   :width: 49%

**Rotated rectangles with approximation.** Light blue is the ideal one and light red is approximated one.


Code \ref{code:overlap} and Figure \ref{fig:overlap} show how \texttt{meent} can handle the overlap of the shapes. Figure \ref{fig:overlap1} and \ref{fig:overlap2} have the same set of rectangles (red and blue) but they are placed in different order and this can be controlled by the function `layer\_info\_list' in Code \ref{code:overlap}. It is the list that contains the base refractive index of the layer and the primitive shapes to be placed on the layer. In case of Figure \ref{fig:overlap1}, red rectangle comes first in the list and blue does for Figure \ref{fig:overlap2}.

.. code-block:: python

    red_rect = mee.rectangle_rotate(*[400, 500], 400, 600, 20, 20, 3.5, 0)
    blue_rect = mee.rectangle_rotate(*[600, 500], 100, 600, 40, 40, 10, -20)

    layer_info_list = [[2.4, red_rect + blue_rect]]  # red bottom, blue top
    layer_info_list = [[2.4, blue_rect + red_rect]]  # blue bottom, red top

    mee.draw(layer_info_list)
    de_ri, de_ti = mee.conv_solve()

|pic3| |pic4|

.. |pic3| image:: images/vector_overlap1.png
   :width: 49%

.. |pic4| image:: images/vector_overlap2.png
   :width: 49%
**The overlap of 2 rectangles in vector modeling.** The hierarchy is determined by the index of the objects in the list.

Raster Modeling
~~~~~~~~~~~~~~~

|pic5| |pic6|

.. |pic5| image:: images/ucell_1d.png
   :width: 49%

.. |pic6| image:: images/ucell_2d.png
   :width: 49%
**Raster-type structure examples.** (a) 2 layers in 1D and (b) 1 layer in 2D grating.

We have 2 example structures of raster modeling as shown in Figure \ref{fig:ucell-grating} and Code \ref{code:raster}.
Figure \ref{fig:ucell_1d-grating} is a stack of 2 layers which has 1D grating. Note that 1D grating unit cell can be defined by setting the length of the second axis to 1 as (a) in Code \ref{code:raster}. Figure \ref{fig:ucell_2d-grating} is a stack of single 2D grating layer.


.. code-block:: python

    # (a): 1D grating with 2 layers
    ucell = np.array(
        [
            [[1, 1, 1, 3.48, 3.48, 3.48, 3.48, 1, 1, 1]],
            [[1, 3.48, 3.48, 1, 1, 1, 1, 3.48, 3.48, 1]],
        ])   # array shape: (2, 1, 10)

    # (b): 2D grating with 1 layers
    ucell = np.array(
        [[
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 3.48, 3.48, 3.48, 3.48, 1, 1, 1],
                [1, 1, 1, 3.48, 3.48, 3.48, 3.48, 1, 1, 1],
                [1, 1, 1, 3.48, 3.48, 3.48, 3.48, 1, 1, 1],
                [1, 1, 1, 3.48, 3.48, 3.48, 3.48, 1, 1, 1],
                [1, 1, 1, 3.48, 3.48, 3.48, 3.48, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]])  # array shape: (1, 8, 10)

    mee = meent.call_mee(backend=backend, ucell=ucell)


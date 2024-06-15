Enhanced Transmittance Matrix Method
====================================


As addressed in :cite:`Moharam:95-implementation, li1993multilayer, Popov:00`, solving Equation \ref{eqn:solve-multilayer} may suffer from the numerical instability coming from the inversion of almost singular matrix when :math:`\mathbb X_\ell` has a very small and possibly numerically zero value. \texttt{meent} adopted Enhanced Transmittance Matrix Method (ETM) :cite:`Moharam:95-implementation` to overcome this by avoiding the inversion of :math:`\mathbb X_\ell`.

The technique is sequentially applied from the last layer to the first layer. In Equation \ref{eqn:solve-multilayer}, the set of modes at the bottom interface of the last layer :math:`(\ell = L)` is

.. math::

    \begin{equation}
    \begin{split}
    \label{eqn:etm-lastlayer}
        &\begin{bmatrix}
            \mathbb W_L & \mathbb{W}_L \mathbb X_L \\
            \mathbb V_L & -\mathbb{V}_L \mathbb X_L
        \end{bmatrix}
        \begin{bmatrix}
            \mathbb W_L \mathbb X_L & \mathbb W_L\\
            \mathbb V_L \mathbb X_L & -\mathbb V_L
        \end{bmatrix}^{-1}
        \begin{bmatrix}
            \mathbb F_{L+1} \\
            \mathbb G_{L+1}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf T_s \\ \mathbf T_p
        \end{bmatrix}
        \\
        &=
        \begin{bmatrix}
            \mathbb W_L & \mathbb W_L \mathbb X_L \\
            \mathbb V_L & -\mathbb V_L \mathbb X_L
        \end{bmatrix}
        \begin{bmatrix}
            {\mathbb X_L}^{-1} & \mathbb{0} \\
            \mathbb{0} & {\mathbb I} \\
        \end{bmatrix}
        {
        \begin{bmatrix}
            \mathbb W_L & \mathbb W_L \\
            \mathbb V_L & -\mathbb V_L
        \end{bmatrix}
        }^{-1}
        \begin{bmatrix}
            \mathbb F_{L+1} \\ \mathbb G_{L+1}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf T_s \\ \mathbf T_p
        \end{bmatrix}.
        \\
    \end{split}
    \end{equation}

The matrix to be inverted can be decomposed into two matrices by isolating :math:`\mathbb X_L`, which is the potential source of the numerical instability. The right-hand side can be shortened with new variables :math:`\mathbb A_L, \mathbb B_L`:

.. math::

    \begin{equation}
    \begin{split}
        \begin{bmatrix}
            \mathbb A_L \\
            \mathbb B_L
        \end{bmatrix}
        =
        \begin{bmatrix}
            {\mathbb W_L} & \mathbb{W_L} \\
            \mathbb{V_L} & {-\mathbb V_L} \\
        \end{bmatrix}^{-1}
        \begin{bmatrix}
            \mathbb F_{L+1} \\ \mathbb G_{L+1}
        \end{bmatrix},
    \end{split}
    \end{equation}

then the right-hand side of Equation \ref{eqn:etm-lastlayer} becomes

.. math::

    \begin{equation}
    \label{eqn:etm-lastlayer1}
    \begin{split}
        \begin{bmatrix}
            \mathbb W_L & \mathbb W_L \mathbb X_L \\
            \mathbb V_L & -\mathbb V_L \mathbb X_L
        \end{bmatrix}
        \begin{bmatrix}
            {\mathbb X_L}^{-1} & \mathbb{0} \\
            \mathbb{0} & {\mathbb I} \\
        \end{bmatrix}
        \begin{bmatrix}
            \mathbb A_L \\ \mathbb B_L
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf T_s \\ \mathbf T_p
        \end{bmatrix}.
        \\
    \end{split}
    \end{equation}

We can avoid the inversion of :math:`\mathbb X_L` by introducing the substitution :math:`\mathbf T_s = {\mathbb A_L}^{-1} \mathbb X_L \mathbf T_{s,L}` and :math:`\mathbf T_p = {\mathbb A_L}^{-1} \mathbb X_L \mathbf T_{p,L}`. Equation \ref{eqn:etm-lastlayer1} then becomes

.. math::

   \begin{equation}
    \begin{split}
        &\begin{bmatrix}
            \mathbb W_L & \mathbb W_L \mathbb X_L \\
            \mathbb V_L & -\mathbb V_L \mathbb X_L
        \end{bmatrix}
        \begin{bmatrix}
            {\mathbb X_L}^{-1} & \mathbb{0} \\
            \mathbb{0} & {\mathbb I} \\
        \end{bmatrix}
        \begin{bmatrix}
            \mathbb A_L \\ \mathbb B_L
        \end{bmatrix}
        {\mathbb A_L}^{-1}{\mathbb X_L}
        \begin{bmatrix}
            \mathbf T_{s,L} \\ \mathbf T_{p,L}
        \end{bmatrix}
        \\
        &=
        \begin{bmatrix}
            \mathbb W_L & \mathbb W_L \mathbb X_L \\
            \mathbb V_L & -\mathbb V_L \mathbb X_L
        \end{bmatrix}
        \begin{bmatrix}
            {\mathbb X_L}^{-1} & \mathbb{0} \\
            \mathbb{0} & {\mathbb I} \\
        \end{bmatrix}
        \begin{bmatrix}
            \mathbb X_L \\ \mathbb B_L \mathbb A_L^{-1} \mathbb X_L
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf T_{s,L} \\ \mathbf T_{p,L}
        \end{bmatrix}
        \\
        &=
        \begin{bmatrix}
            \mathbb W_L & \mathbb W_L \mathbb X_L \\
            \mathbb V_L & -\mathbb V_L \mathbb X_L
        \end{bmatrix}
        \begin{bmatrix}
            \mathbb I \\ \mathbb B_L \mathbb A_L^{-1} \mathbb X_L
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf T_{s,L} \\ \mathbf T_{p,L}
        \end{bmatrix}
        \\
        &=
        \begin{bmatrix}
            \mathbb W_L(\mathbb I+\mathbb X_L \mathbb B_L \mathbb A_L^{-1} \mathbb X) \\
            \mathbb V_L(\mathbb I-\mathbb X_L \mathbb B_L \mathbb A_L^{-1} \mathbb X)
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf T_{s,L} \\ \mathbf T_{p,L}
        \end{bmatrix}
        \\
        &=
        \begin{bmatrix}
            \mathbb F_L \\ \mathbb G_L
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf T_{s,L} \\ \mathbf T_{p,L}
        \end{bmatrix}
        .
    \end{split}
    \end{equation}

These steps can be repeated until the iteration gets to the first layer :math:`(\ell = 1)`, then the form becomes

.. math::

    \begin{align}
    \begin{bmatrix}
    \sin\psi\ \boldsymbol\delta_{00} \\
    \cos\psi\ \cos\theta\ \boldsymbol\delta_{00}
     \\
    j\sin\psi\ n_{\text{I}}\ \cos\theta\ \boldsymbol\delta_{00} \\
    -j\cos\psi\ n_{\text{I}}\ \boldsymbol\delta_{00} \\
    \end{bmatrix}
    +
    \begin{bmatrix}
    \mathbf I & \mathbf 0 \\
    \mathbf 0 & -j\mathbf Z_I \\
    -j\mathbf Y_I & \mathbf 0 \\
    \mathbf 0 & \mathbf I
    \end{bmatrix}
    \begin{bmatrix}
    \mathbf R_s \\
    \mathbf R_p
    \end{bmatrix}
    =
    \begin{bmatrix}
    \mathbb F_1 \\ \mathbb G_1
    \end{bmatrix}
    \begin{bmatrix}
    \mathbf T_{s,1} \\ \mathbf T_{p,1}
    \end{bmatrix},
    \end{align}

where

.. math::

    \begin{bmatrix}
    \mathbf T_s \\ \mathbf T_p
    \end{bmatrix}
    =
    \mathbb A_L^{-1} \mathbb X_L \cdots
    \mathbb A_\ell^{-1} \mathbb X_\ell \cdots
    \mathbb A_1^{-1} \mathbb X_1
    \begin{bmatrix}
    \mathbf T_{s,1} \\ \mathbf T_{p,1}
    \end{bmatrix}.

----

.. bibliography::
   :filter: docname in docnames

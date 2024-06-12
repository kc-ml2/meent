Connecting Layers
=================

Once the eigenmodes of each grating layer are identified, the transfer matrix method (TMM) can be utilized to determine the Rayleigh coefficients (:math:`\mathbf R_s, \mathbf R_p, \mathbf T_s, \mathbf T_p`) and the diffraction efficiencies.
TMM effectively represents this process as a matrix multiplication, where the transfer matrix is constructed by considering the interaction between the eigenmodes of neighboring layers.
This matrix accounts for the energy transfer and phase shift between the eigenmodes, and it is used to propagate the electromagnetic fields through the entire periodic structure.

From the boundary conditions, the systems of equations consisting of the in-plane (tangential) field components :math:`(\mathbf E_\mathtt s, \mathbf E_\mathtt p, \mathbf H_\mathtt s, \mathbf H_\mathtt p)` can be described at each layer interface. We will first consider the case of a single grating layer cladded with the superstrate and substrate, then expand to multilayer structure. At the input boundary (:math:`z=0`):

.. math::
    :label: eqn:z0

    \begin{align}
        \begin{bmatrix}
            \sin\psi\ \boldsymbol{\delta}_{00}\\
            \cos\psi\ \cos\theta\ \boldsymbol\delta_{00} \\
            j\sin\psi\ \mathtt n_{\text{I}} \cos\theta\ \boldsymbol\delta_{00} \\
            -j\cos\psi\ \mathtt n_{\text{I}}\ \boldsymbol\delta_{00} \\
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
            {\mathbf W_{g,ss}} &  {\mathbf W_{g,sp}} & {\mathbf W_{g,ss}\mathbf X_{g,1}} & {\mathbf W_{g,sp}\mathbf X_{g,2}}
            \\
            {\mathbf W_{g,ps}} & {\mathbf W_{g,pp}} & {\mathbf W_{g,ps}\mathbf X_{g,1}} & {\mathbf W_{g,pp}\mathbf X_{g,2}}
            \\
            {\mathbf V_{g,ss}} & {\mathbf V_{g,sp}} & {-\mathbf V_{g,ss}\mathbf X_{g,1}} & {-\mathbf V_{g,sp}\mathbf X_{g,2}}
            \\
            {\mathbf V_{g,ps}} & {\mathbf V_{g,pp}} & {-\mathbf V_{g,ps}\mathbf X_{g,1}} & {-\mathbf V_{g,pp}\mathbf X_{g,2}}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf c_{g,1}^+ \\
            \mathbf c_{g,2}^+ \\
            \mathbf c_{g,1}^- \\
            \mathbf c_{g,2}^- \\
        \end{bmatrix},
    \end{align}

and at the output boundary (:math:`z=d`):

.. math::

    \begin{align}
    \label{eqn:zd}
        \begin{bmatrix}
            {\mathbf W_{g,ss}\mathbf X_{g,1}} & {\mathbf W_{g,sp}\mathbf X_{g,2}} & {\mathbf W_{g,ss}} & {\mathbf W_{g,sp}}
            \\
            {\mathbf W_{g,ps}\mathbf X_{g,1}} & {\mathbf W_{g,pp}\mathbf X_{g,2}} & {\mathbf W_{g,ps}} & {\mathbf W_{g,pp}}
            \\
            {\mathbf V_{g,ss}\mathbf X_{g,1}} & {\mathbf V_{g,sp}\mathbf X_{g,2}} & {-\mathbf V_{g,ss}} &  {-\mathbf V_{g,sp}}
            \\
            {\mathbf V_{g,ps}\mathbf X_{g,1}} & {\mathbf V_{g,pp}\mathbf X_{g,2}} & {-\mathbf V_{g,ps}} & {-\mathbf V_{g,pp}}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf c_{g,1}^+ \\
            \mathbf c_{g,2}^+ \\
            \mathbf c_{g,1}^- \\
            \mathbf c_{g,2}^- \\
        \end{bmatrix}
        =
        \begin{bmatrix}
            \mathbf I & \mathbf 0 \\
            \mathbf 0 & j\mathbf Z_{\text{II}} \\
            j\mathbf Y_{\text{II}} & \mathbf 0 \\
            \mathbf 0 & \mathbf I
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf T_s \\
            \mathbf T_p
        \end{bmatrix},
    \end{align}

Here, the variables used above are defined: :math:`\mathbf X_{g,1}, \mathbf X_{g,2}` are the diagonal matrices

.. math::

    \begin{align}
        \mathbf X_{g,1} =
        \begin{bmatrix}
            e^{-k_0 q_{g,1,1}d_g} & & 0 \\
             & \ddots &  \\
            0 & & e^{-k_0 q_{g,1,\xi}d_g}
        \end{bmatrix}, \quad
        \mathbf X_{g,2} =
        \begin{bmatrix}
            e^{-k_0 q_{g,2,1}d_g} & & 0 \\
             & \ddots &  \\
            0 & & e^{-k_0 q_{g,2,\xi}d_g}
        \end{bmatrix},
    \end{align}

where :math:`d_g` is the thickness of the grating layer, and
:math:`\mathbf Y_{\text{I}}` and :math:`\mathbf Z_{\text{I}}` are

.. math::

    \begin{align}
        \mathbf Y_{\text I} =
        \begin{bmatrix}
            \tilde k_{\text{I},z,(-N,-M)} & & 0 \\
             & \ddots &  \\
            0 & & \tilde k_{\text{I},z,(N,M)}
        \end{bmatrix}, \quad
        \mathbf Z_{\text{I}} =
        \frac{1}{(\mathtt n_{\text{I}})^2}
        \begin{bmatrix}
            \tilde k_{\text{I},z,(-N,-M)} & & 0 \\
             & \ddots &  \\
            0 & & \tilde k_{\text{I},z,(N,M)}
        \end{bmatrix},
    \end{align}

and :math:`\mathbf Y_{\text{II}}` and :math:`\mathbf Z_{\text{II}}` are

.. math::

    \begin{align}
        \mathbf Y_{\text {II}} =
        \begin{bmatrix}
            \tilde k_{\text{II},z,(-N,-M)} & & 0 \\
             & \ddots &  \\
            0 & & \tilde k_{\text{II},z,(N,M)}
        \end{bmatrix}, \quad
        \mathbf Z_{\text{II}} =
        \frac{1}{(\mathtt n_{\text{II}})^2}
        \begin{bmatrix}
            \tilde k_{\text{II},z,(-N,-M)} & & 0 \\
             & \ddots &  \\
            0 & & \tilde k_{\text{II},z,(N,M)}
        \end{bmatrix}.
    \end{align}

Here, new set of :math:`\mathbf W_g` and :math:`\mathbf V_g` on SP basis :math:`\{\hat s, \hat p\}` are introduced which are recombined from the set of :math:`\mathbf W_g` and :math:`\mathbf V_g` from XY basis :math:`\{\hat x, \hat y\}`:

.. math::

    \begin{align}
    \mathbf W_{g,ss}&=\mathbf F_c\mathbf W_{g,21}-\mathbf F_s\mathbf W_{g,11}, & \mathbf W_{g,sp}&=\mathbf F_c\mathbf W_{g,22}-\mathbf F_s\mathbf W_{g,12},
    \\
    \mathbf W_{g,ps}&=\mathbf F_c\mathbf W_{g,11}+\mathbf F_s\mathbf W_{g,21}, & \mathbf W_{g,pp}&=\mathbf F_c\mathbf W_{g,12}+\mathbf F_s\mathbf W_{g,22},
    \\
    \mathbf V_{g,ss}&=\mathbf F_c\mathbf V_{g,11}+\mathbf F_s\mathbf V_{g,21}, & \mathbf V_{g,sp}&=\mathbf F_c\mathbf V_{g,12}+\mathbf F_s\mathbf V_{g,22},
    \\
    \mathbf V_{g,ps}&=\mathbf F_c\mathbf V_{g,21}-\mathbf F_s\mathbf V_{g,11}, & \mathbf V_{g,pp}&=\mathbf F_c\mathbf V_{g,22}-\mathbf F_s\mathbf V_{g,12},
    \end{align}

with :math:`\mathbf F_c` and :math:`\mathbf F_s` being diagonal matrices with the diagonal elements :math:`\cos\varphi_{(n,m)}` and :math:`\sin\varphi_{(n,m)}`, respectively, where

.. math::

    \begin{align}
        \varphi_{(n,m)} = \tan^{-1}(k_{y, n}/k_{x, m}).
    \end{align}

Equations \ref{eqn:z0} and \ref{eqn:zd} can be reduced to one set of equations by eliminating :math:`\mathbf c^\pm_{1,2}`:

.. math::

    \begin{align}
        \begin{bmatrix}
            \sin\psi\ \boldsymbol\delta_{00} \\
            \cos\psi\ \cos\theta\ \boldsymbol\delta_{00}
             \\
            j\sin\psi\ \mathtt n_{\text{I}} \cos\theta\ \boldsymbol\delta_{00} \\
            -j\cos\psi\ \mathtt n_{\text{I}}\ \boldsymbol\delta_{00} \\
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
        % \\\quad\\
        % \begin{align*}
        =
        \begin{bmatrix}
        \mathbb W & \mathbb {W X} \\
        \mathbb V & -\mathbb {V X}
        \end{bmatrix}
        \begin{bmatrix}
        \mathbb {W X} & \mathbb W \\
        \mathbb {V X} & -\mathbb V
        \end{bmatrix}^{-1}
        \begin{bmatrix}
        \mathbb F \\
        \mathbb G \\
        \end{bmatrix}
        \begin{bmatrix}
        \mathbf T_s \\ \mathbf T_p
        \end{bmatrix},
        % \end{align*}
    \end{align}

where

.. math::

    \begin{align}
        \mathbb W
        =
        \begin{bmatrix}
        \mathbf W_{g,ss} & \mathbf W_{g,sp} \\
        \mathbf W_{g,ps} & \mathbf W_{g,pp}
        \end{bmatrix},
        \quad \mathbb V
        =
        \begin{bmatrix}
        \mathbf V_{g,ss} & \mathbf V_{g,sp} \\
        \mathbf V_{g,ps} & \mathbf V_{g,pp}
        \end{bmatrix},
        \quad \mathbb X
        =
        \begin{bmatrix}
        \mathbf X_{g,1} & \mathbf 0 \\
        \mathbf 0 & \mathbf X_{g,2}
        \end{bmatrix},
        \quad \mathbb F
        =
        \begin{bmatrix}
        \mathbf I & \mathbf 0 \\
        \mathbf 0 & j\mathbf Z_{\text{II}}
        \end{bmatrix},
        \quad \mathbb G
        =
        \begin{bmatrix}
        j\mathbf Y_{\text{II}} & \mathbf 0 \\
        \mathbf 0 & \mathbf I
        \end{bmatrix}.
    \end{align}

This equation for a single layer grating can be simply extended to a multi-layer system as the following:

.. math::

    \begin{align}
        \label{eqn:solve-multilayer}
        \begin{bmatrix}
            \sin\psi\ \boldsymbol\delta_{00} \\
            \cos\psi\ \cos\theta\ \boldsymbol\delta_{00}
             \\
            j\sin\psi\ \mathtt n_{\text{I}} \cos\theta\ \boldsymbol\delta_{00} \\
            -j\cos\psi\ \mathtt n_{\text{I}}\ \boldsymbol\delta_{00} \\
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
        % \\\quad\\
        % \begin{align*}
        =
        \prod_{\ell=1}^{L}
        \begin{bmatrix}
        \mathbb W_\ell & \mathbb {W_\ell X_\ell} \\
        \mathbb V_\ell & -\mathbb {V_\ell X_\ell}
        \end{bmatrix}
        \begin{bmatrix}
        \mathbb {W_\ell X_\ell} & \mathbb W_\ell \\
        \mathbb {V_\ell X_\ell} & -\mathbb V_\ell
        \end{bmatrix}^{-1}
        \begin{bmatrix}
        \mathbb F_{L+1} \\
        \mathbb G_{L+1} \\
        \end{bmatrix}
        \begin{bmatrix}
        \mathbf T_s \\ \mathbf T_p
        \end{bmatrix},
        % \end{align*}
    \end{align}
where :math:`L` is the number of layers and

.. math::

    \begin{align}
        \mathbb F_{L+1}
        =
        \begin{bmatrix}
        \mathbf I & \mathbf 0 \\
        \mathbf 0 & j\mathbf Z_{\text{II}}
        \end{bmatrix}, \quad
        \mathbb G_{L+1}
        =
        \begin{bmatrix}
        j\mathbf Y_{\text{II}} & \mathbf 0 \\
        \mathbf 0 & \mathbf I
        \end{bmatrix}.
    \end{align}

Since we have four matrix equations for four unknown coefficients (:math:`\mathbf R_s, \mathbf R_p, \mathbf T_s, \mathbf T_p`), they can be derived and used for calculating diffraction efficiencies (also called the reflectance and transmittance).

The diffraction efficiency is the ratio of the power flux in propagating direction between incidence and diffracted wave of interest. It can be calculated by time-averaged Poynting vector \cite{liu2012s4, hugonin2021reticolo, rumpf-dissertation}:

.. math::

    \begin{align}
        P &= \frac{1}{2} \operatorname{Re}{(E \times H^{*})},
    \end{align}

where :math:`^*` is the complex conjugate.
Now we can find the total power of the incident wave as a sum of the power of TE wave and TM wave:

.. math::

    \begin{equation}
    \begin{split}
        P^{inc} & = P_{s}^{inc} + P_{p}^{inc} \\
          & = \frac{1}{2} \operatorname{Re}\Bigg[{(E_{s} \times H_{s}^*) + (E_{p} \times H_{p}^*)}\Bigg] \\
          & = \frac{1}{2} \operatorname{Re}
          \Bigg[{
            (\sin\psi\ \cdot \sin\psi\ \mathtt n_{\text{I}}\ \cos\theta) +
            (\cos\psi\ \cos\theta\ \cdot \cos\psi\ \mathtt n_{\text{I}})
          }\Bigg] \\
          & = \frac{1}{2} \operatorname{Re}
          \Bigg[{
            (\sin^2\psi\ \mathtt n_{\text{I}} \cos\theta) +
            (\cos^2\psi\ \mathtt n_{\text{I}} \cos\theta)
          }\Bigg] \\
          & = \frac{1}{2} \operatorname{Re}
          \Bigg[{
            (\mathtt n_{\text{I}} \cos\theta)
          }\Bigg].
        \end{split}
    \end{equation}

The power in each reflected diffraction mode is

.. math::

    \begin{equation}
    \begin{split}
        P_{n,m}^{r} & = P_{nm, s}^{r} + P_{nm, p}^{r} \\
          & = \frac{1}{2} \operatorname{Re}\Bigg[{(E_{nm, s}^{r} \times (H_{nm, s}^{r})^*) + (E_{nm, p}^{r} \times (H_{nm, p}^{r})^*)}\Bigg] \\
          & = \frac{1}{2} \operatorname{Re}
          \Bigg[{
            R_{nm, s} \cdot \frac{k_{\text{I},z,(n,m)}}{k_0}R_{nm,s}^* +
            \frac{k_{\text{I},z,(n,m)}}{k_0 \mathtt{n}_{\text{I}}^2} R_{nm, p} \cdot R_{nm, p}^*
          }\Bigg] \\
          & = \frac{1}{2} \operatorname{Re}
          \Bigg[{
            R_{nm, s}R_{nm, s}^* \cdot \frac{k_{\text{I},z,(n,m)}}{k_0} +
            R_{nm, p}R_{nm, p}^* \cdot \frac{k_{\text{I},z,(n,m)}}{k_0 \mathtt{n}_{\text{I}}^2}
          }\Bigg],
    \end{split}
    \end{equation}

and the power in each transmitted diffraction mode is

.. math::

    \begin{equation}
    \begin{split}
        P_{n,m}^{t} & = P_{nm, s}^{t} + P_{nm, p}^{t} \\
          & = \frac{1}{2} \operatorname{Re}\Bigg[{(E_{nm, s}^{t} \times (H_{nm, s}^{t})^*) + (E_{nm, p}^{t} \times (H_{nm, p}^{t})^*)}\Bigg] \\
          & = \frac{1}{2} \operatorname{Re}
          \Bigg[{
            T_{nm, s} \cdot \frac{k_{\text{II},z,(n,m)}}{k_0}T_{nm,s}^* +
            \frac{k_{\text{II},z,(n,m)}}{k_0 \mathtt{n}_{\text{II}}^2} T_{nm, p} \cdot T_{nm, p}^*
          }\Bigg] \\
          & = \frac{1}{2} \operatorname{Re}
          \Bigg[{
            T_{nm, s}T_{nm, s}^* \cdot \frac{k_{\text{II},z,(n,m)}}{k_0} +
            T_{nm, p}T_{nm, p}^* \cdot \frac{k_{\text{II},z,(n,m)}}{k_0 \mathtt{n}_{\text{II}}^2}
          }\Bigg].
    \end{split}
    \end{equation}

Since the diffraction efficiency is the ratio between them :math:`(P_{out}/P_{inc})`, we can get the efficiencies of reflected and transmitted waves:

.. math::

    \begin{align}\label{}
        DE_{r,(n,m)} &= |R_{s,(n,m)}|^2 \operatorname{Re}{\bigg(\frac{k_{\text{I},z,(n,m)}}{k_0 \mathtt n_\text{I} \cos{\theta}}\bigg)} +  |R_{p,(n,m)}|^2 \operatorname{Re}{\bigg(\frac{k_{\text{I},z,(n,m)}/{\mathtt n_{\text{I}}}^2}{k_0 \mathtt n_\text{I} \cos{\theta}}\bigg)}, \\
        DE_{t,(n,m)} &= |T_{s,(n,m)}|^2 \operatorname{Re}{\bigg(\frac{k_{\text{II},z,(n,m)}}{k_0 \mathtt n_\text{I}\cos{\theta}}\bigg)} +  |T_{p,(n,m)}|^2  \operatorname{Re}{\bigg(\frac{k_{\text{II},z,(n,m)}/{\mathtt n_{\text{II}}}^2}{k_0 \mathtt n_\text{I} \cos{\theta}}\bigg)}.
    \end{align}


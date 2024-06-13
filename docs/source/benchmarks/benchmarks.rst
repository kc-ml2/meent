Benchmarks
==========

In this section, we will address the 1D metasurface problem covered in our previous work \cite{seo2021structural}
with \texttt{meent} so that we can benchmark and analyze its capability and functionality.


We focus on optimizing the diffraction efficiency of a 1D diffraction grating.

Case Application
----------------

This metagrating deflector is composed of silicon pillars placed on a silica substrate. The device period is divided into 64 cells, and each cell can be filled with either air or silicon. The Figure of Merit for this optimization is set to the deflection efficiency of the +1$^{st}$ order transmitted wave when TM polarized wave is normally incident from the silica substrate as in Figure \ref{fig:1d-meta}.

.. figure:: images/1d_deflector_scheme.png
    :figwidth: 800
    :align: center

    Figure 1: **The image of 1D diffraction metagrating on a silicon dioxide substrate.**

Fourier Series Implementation
-----------------------------
When the sampling frequency of permittivity distribution is not enough, Fourier coefficients from DFS is aliased.
It can be resolved by increasing the sampling rate that is implemented in the way of duplicating the elements
so the array is extended to have identical distribution but larger array size. We will call this Enhanced DFS,
and it's implemented in \texttt{meent} as a default option.

% It is also possible to use CFS to avoid this issue. But note that AD for CFS in Raster modeling is not supported since current algorithm has compression step (adopted from Reticolo \cite{reticolo}) which loses the calculation chain needed for backpropagation.

% Before we start to look optimization part, we will address an effect of sampling frequency in Fourier series. As explained, \texttt{meent} supports two Fourier series methods(DFS and CFS) and DFS is proper for Raster modeling. However, DFS suffers from information loss coming from discretizing the original function while CFS uses the original function so no loss. It can be resolved by increasing the sampling frequency as dealiasing in discrete Fourier analysis.

% We will examine the convergence of RCWA calculations for different Fourier series methods by increasing FTO while utilizing Raster modeling.


|pic1| |pic2|

.. |pic1| image:: images/convergence_bad.png
   :width: 49%

.. |pic2| image:: images/dfs_hist.png
   :width: 49%

Convergence test

Histogram of the difference compared to Reticolo

Figure \ref{fig:convergence_FTO_sweep} illustrates the convergence tests of a particular structure with four different RCWA implementations.
% The purpose is to find the sweet spot between accuracy and time.
Considering Reticolo as the reference, we can see CFS is well-matched but DFS shows different behavior. This is due to the insufficient sampling rate of permittivity distribution, which can be resolved by Enhanced DFS. Figure \ref{fig:eff_difference_hist} is the histogram of the discrepancies from Reticolo result. About 600k structures were evaluated with 4 implementations and the errors of 3 \texttt{meent} implementations were calculated based on Reticolo. CFS shows the smallest errors and this is because Reticolo too uses CFS (CFS algorithms in \texttt{meent} are adopted from Reticolo). Enhanced DFS decreases the error about three orders of magnitudes (e.g., the median of DFS is 4.3E-4 and this becomes 1.4E-7).

% \subsection{Python native}
% \texttt{meent} is written in Python and this is great advantage for researchers in Photonics to apply state-of-the-art ML techniques and for the people in ML have lower huddle since de-facto language in Photonics is MATLAB.
% \begin{figure} % picture
%     \centering
%     \includegraphics[scale=0.5]{images/benchmarks/cal_time.png}
%     \caption{Violin plot of simulation time records that for EM solver to return RCWA result to Python script of experiment. Identical structures were simulated with Reticolo and \texttt{meent} in same computational environment. The green line is the time difference between two platforms.}
%     \label{fig:cal_time_compare}
% \end{figure}

Python-native
-------------

.. figure:: images/cal_time.png
    :figwidth: 800
    :align: center

    Figure 2: Violin plot of simulation time records that for EM solver to return RCWA result to
    Python script of experiment. Identical structures were simulated with Reticolo and \texttt{meent} in
    same computational environment. The green line is the time difference between two platforms.

Computing Performance
---------------------

.. list-table:: Hardware specification
   :widths: 25 25 20 22 22
   :header-rows: 1

   * -
     - CPU
     - clock
     - # threads
     - GPU
   * - Alpha
     - Intel Xeon Gold 6138
     - 2.00GHz
     - 80
     -
   * - Beta
     - Intel Xeon E5-2650 v4
     - 2.20GHz
     - 48
     - GeForce RTX 2080ti
   * - Gamma
     - Intel Xeon Gold 6226R
     - 2.90GHz
     - 64
     - GeForce RTX 3090


% \begin{table}
% \caption{Hardware specification}
% \label{tab:hardware}
% \centering
% \begin{tabular}{ l||l|l|l|l }
% \hline
% & CPU & clock & \# threads & GPU \\
% \hline\hline
% Alpha & Intel Xeon Gold 6138 & 2.00GHz & 80 & NVIDIA TITAN RTX \\
% Beta & Intel Xeon E5-2650 v4 & 2.20GHz & 48 & GeForce RTX 2080ti \\
% Gamma & Intel Xeon Gold 6226R & 2.90GHz & 64 & GeForce RTX 3090 \\
% Softmax & Intel i9-13900K & 3.00GHz & 32 & GeForce RTX 4090 \\
% \hline
% \end{tabular}
% \end{table}

\begin{center}
\begin{table}
\begin{center}
\caption{Performance test condition}
\label{tab:performance_condition}
\begin{tabular}{ |c|c|c||c|c|c| }
\hline
backend & device & bit & alpha server & beta server & gamma server\\
\hline\hline
NumPy & CPU & 64 & (A1) & (B1) & (C1)\\
NumPy & CPU & 32  & (A2) & (B2) & (C2)\\
JAX & CPU & 64  & (A3) & (B3) & (C3)\\
JAX & CPU & 32  & (A4) & (B4) & (C4)\\
JAX & GPU & 64  & - & (B5) & (C5)\\
JAX & GPU & 32  & - & (B6) & (C6)\\
PyTorch & CPU & 64 & (A7) & (B7) & (C7)\\
PyTorch & CPU & 32 & (A8) & (B8) & (C8)\\
PyTorch & GPU & 64 & - & (B9) & (C9)\\
PyTorch & GPU & 32 & - & (B10) & (C10)\\
\hline
\end{tabular}
\end{center}
\end{table}
\end{center}

% \begin{center}
% \begin{table}
% \begin{center}
% \caption{Performance test condition}
% \label{tab:performance_condition}
% \begin{minipage}{0.3\linewidth}
% \begin{scriptsize}
% \caption{Alpha server}
% \begin{tabular}{ |c|c|c|c| }
% \hline
% & backend & device & bit \\
% \hline\hline
% (A1) & numpy & CPU & 64 \\
% (A2) & numpy & CPU & 32 \\
% (A3) & jax & CPU & 64 \\
% (A4) & jax & CPU & 32 \\
% - & jax & GPU & 64 \\
% - & jax & GPU & 32 \\
% (A7) & torch & CPU & 64 \\
% (A8) & torch & CPU & 32 \\
% - & torch & GPU & 64 \\
% - & torch & GPU & 32 \\
% \hline
% \end{tabular}
% \end{scriptsize}
% \end{minipage}
% \begin{minipage}{0.3\linewidth}
% \begin{scriptsize}
% \caption{Beta server}
% \begin{tabular}{ |c|c|c|c| }
% \hline
% & backend & device & bit \\
% \hline\hline
% (B1) & numpy & CPU & 64 \\
% (B2) & numpy & CPU & 32 \\
% (B3) & jax & CPU & 64 \\
% (B4) & jax & CPU & 32 \\
% (B5) & jax & GPU & 64 \\
% (B6) & jax & GPU & 32 \\
% (B7) & torch & CPU & 64 \\
% (B8) & torch & CPU & 32 \\
% (B9) & torch & GPU & 64 \\
% (B10) & torch & GPU & 32 \\
% \hline
% \end{tabular}
% \end{scriptsize}
% \end{minipage}
% \begin{minipage}{0.3\linewidth}
% \begin{scriptsize}
% \caption{Gamma server}
% \begin{tabular}{ |c|c|c|c| }
% \hline
% & backend & device & bit \\
% \hline\hline
% (C1) & numpy & CPU & 64 \\
% (C2) & numpy & CPU & 32 \\
% (C3) & jax & CPU & 64 \\
% (C4) & jax & CPU & 32 \\
% (C5) & jax & GPU & 64 \\
% (C6) & jax & GPU & 32 \\
% (C7) & torch & CPU & 64 \\
% (C8) & torch & CPU & 32 \\
% (C9) & torch & GPU & 64 \\
% (C10) & torch & GPU & 32 \\
% \hline
% \end{tabular}
% \end{scriptsize}
% \end{minipage}
% \end{center}
% \end{table}
% \end{center}
In this section, computing options to speed up the calculation - backend, device (CPU and GPU) and architecture (64bit and 32bit) - will be benchmarked. Table \ref{tab:hardware} is the hardware specification of the test server and Table \ref{tab:performance_condition} is the index of each test condition.

\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{sections/manual/images/benchmarks/performance/result_all.png}
    \caption{\textbf{Performance test: calculation time with respect to FTO.} Top row is the result from 64bit and bottom is from 32bit. The first column is the result from the test server alpha and the rest is beta and gamma in order.}
    \label{fig:benchmark/performance_all}
\end{figure}

The graphs in Figure \ref{fig:benchmark/performance_all} are calculation time vs FTO with all the data per machine and architecture. Before look into the details, we will briefly mention some notice in this figure. (1) JAX can't afford large FTO regardless of device. We suspect that this is related to JIT compilation which takes much time and memory for the compilation at the first run. (2) GPU with JAX and PyTorch can't accept large FTO even though GPU memory is more than needed for array upload. (3) if large amount of calculation is needed, Numpy or PyTorch on CPU is the option. (4) no golden option exists: it is recommended to find the best option for the test environment by doing benchmark tests.

We will visit these computing options one by one. The option C9 at FTO 1600 will be excluded in further analyses: this seems an optimization issue in PyTorch or CUDA.

\subsubsection{Backend: NumPy, JAX and PyTorch}
NumPy, JAX and PyTorch as a backend are benchmarked. NumPy is installed via PyPI which is compiled with OpenBLAS. There are many types of BLAS libraries and the most representative ones are OpenBLAS and MKL (Math Kernel Library).
As of now, PyPI provides NumPy with OpenBLAS while conda does one with MKL. This makes small discrepancy in terms of speed and precision hence pay attention when doing consistency test between machines.
\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{sections/manual/images/benchmarks/performance/backend.png}
    \caption{\textbf{Performance test: calculation time by FTO sweep.} The result is normalized by NumPy case from the same options to compare the behavior of other backends. In these plots, black dashed line is $y=1$ and the results of NumPy cases lie on this line since they are normalized by themselves.}
    \label{fig:benchmark/backend}
\end{figure}
Figure \ref{fig:benchmark/backend} is the relative simulation time per server and architecture normalized by the time of NumPy case in the same conditions to make comparison easy.
In small FTO regime, all the options were successfully operated and no champion exists. Hence it is strongly recommended to run benchmark test on your hardware and pick the most efficient one. In case of X7 (A7, B7 and C7), Alpha and Gamma show the same behavior - spike in 100 - while beta shows fluctuation around B1. One possible reason for this is the type of CPU. The CPUs of Alpha and Gamma belong to `Xeon Scalable Processors' group but Beta is `Xeon E Processors'. Currently we don't know if this actually makes difference or some other reason (such as the number of threads or BLAS implementation) does. This result may vary if MKL were used instead of OpenBLAS.
In large FTO, only two options are available: NumPy and PyTorch on CPU in 64 bit. In case of JAX, the tests were failed: we watched memory occupation surge during the simulation which seems unrelated to matrix calculation. This might be an issue of JIT (Just In Time) compilation in JAX. Between NumPy and PyTorch, PyTorch is about twice faster than NumPy in both architectures at Alpha and Gamma, but beta shows different behavior. This too, we don't know the root cause but one notable difference is the family of CPU type.

\subsubsection{Device: CPU and GPU}
\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{sections/manual/images/benchmarks/performance/device.png}
    \caption{\textbf{Performance test result.} The calculation time of GPU cases are normalized by CPU cases from the same options to see the efficiency of GPU utilization. In these plots, black dashed line is $y=1$ where the capability of both are the same.}
    \label{fig:benchmark/device}
\end{figure}

Figure \ref{fig:benchmark/device} shows the relative simulation time of GPU cases normalized by CPU cases on the same backend and architecture. Note that it is \textbf{relative} time, so the smaller time does not mean it is a good option for the simulation experiments: the relative time can be small even if the absolute time of CPU and GPU are very large compared to other options.

JAX shows good GPU utilization throughout the whole range (except one point in beta) regardless of the architecture. Considering the architecture, the data trend in beta is not clear while the gamma clearly shows that GPU utilization can be more effective in 32bit operation. PyTorch data is a bit noisier than of JAX, but has the similar behavior per server. The data in beta is hard to conclude as the JAX cases and the gamma too shows ambiguous trend but we can consider GPU option is efficient with wide range of FTOs.
% It is also effective in PyTorch. FTO range of $200\sim800$ in beta (except one point) and $100\sim800$ in gamma show smaller simulation time with GPU utilization. Above 1600 in beta or 3200 in gamma, GPU tests failed.

Up to date, eigendecomposition for non-hermitian matrix which is the most expensive step ($O(M^3N^3)$) in RCWA, is not implemented on GPU in JAX and PyTorch hence the calculations are done on CPU and the results are sent back to GPU. As a result, we cannot expect great performance enhancement in using GPUs.

\subsubsection{Architecture: 64 and 32 bit}
\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{sections/manual/images/benchmarks/performance/archi.png}
    \caption{\textbf{Performance test result.} The calculation time of 32bit cases are normalized by 64bit cases from the same options. In these plots, black dashed line is $y=1$ where the capability of both are the same.}
    \label{fig:benchmark/archi}
\end{figure}

In Figure \ref{fig:benchmark/archi}, calculation time of 32bit case is normalized by 64bit case with the same condition. With some exceptions, most points show that simulation in 32bit is faster than 64bit. Here are some important notes:
(1) From our understanding, the eigendecomposition (Eig) in NumPy operates in 64bit regardless of the input type - even though the input is 32bit data (float32 or complex64), the matrix operations inside Eig are done in 64bit but returns the results in 32bit data type. This is different from JAX and PyTorch - they provides Eig in 32bit as well as 64bit. Hence the 32bit NumPy cases in the figure approach to 1 as FTO increases because the calculation time for Eig is the same and it is the most time-consuming step.
(2) Keep in mind that 32bit data type can handle only 8 digits. This means that 1000 + 0.00001 becomes 1000 without any warnings or error raises. For such a reason, the accuracy of 32bit cases in the figures are not guaranteed - we only consider the calculation time.
% (3) Eig in PyTorch shows interesting behavior. As FTO increases, Eig time in 32bit overtakes 64bit and this causes the time of B8 and C8 also increases as FTO increases. This is counter-intuitive and we don't have good explanation but cautiously presume that this might be related to the accuracy and precision in Eig or an optimization issue of PyTorch.
(3) Eig in PyTorch shows interesting behavior: as FTO increases, calculation time in 32bit overtakes 64bit - see A8/A7, B8/B7 and C8/C7. This is counter-intuitive and we don't have good explanation but cautiously guess that this might be related to the accuracy and precision in Eig or an optimization issue of PyTorch.


% \begin{center}
% \begin{table}
% \caption{FTO sweep}
% \centering
% \begin{tabular}{ c|ccccccccccccc }
% \hline
% \multirow{2}{4em}{} & \multicolumn{9}{c}{FTO} \\
% & 50 & 100 & 200 & 400 & 800 & 1600 & 3200 & 6400 & 12800\\
% \hline\hline
% (A1)	&0.06	&0.25	&0.95	&2.32	&14.27	&70.15	&410.83	&2658.76&23975.42 \\
% (A2)	&0.06	&0.24	&0.80	&3.43	&13.14	&73.30	&382.21	&2344.89&22362.61 \\
% (A3)	&0.07	&0.22	&0.80	&2.55	&10.52	&70.78	&-		&-&- \\
% (A4)	&0.05	&0.18	&0.45	&3.07	&6.42	&35.45	&	-	&-&- \\
% (A7)	&0.06	&0.49	&0.44	&1.96	&7.63	&47.68	&245.34	&1215.31 &7367.29 \\
% (A8)	&0.03	&0.16	&0.33	&1.60	&6.45	&30.36	&165.83	&1038.41 &7128.23 \\
% (B1)&0.02&	0.09&	0.42&	2.28&	11.89&	47.23&	322.76&	1837.95 &12971.09 \\
% (B2)&0.03&	0.10&	0.30&	1.39&	7.08&	38.06&	256.32&	1590.72 &10563.10 \\
% (B3)&0.02&	0.09&	0.36&	1.35&	7.26&-&-&-&- \\
% (B4)&0.02&	0.05&	0.19&	1.02&	3.81&-&-&-&- \\
% (B5)&0.02&	0.08&	0.28&	1.50&	5.62&-&-&-&- \\
% (B6)&0.01&	0.05&	0.18&	0.75&	2.21&-&-&-&- \\
% (B7)&0.03&	0.08&	0.54&	2.40&	10.73&	46.05&	332.08&	1932.11&	12881.39 \\
% (B8)&0.03&	0.07&	0.23&	1.07&	5.13&	38.03&	332.35&	2460.85&	20003.21 \\
% (B9)&0.05&	0.10&	0.43&	1.29&	5.80&	71.03&-&-&-\\
% (B10)&0.03&	0.07&	0.27&	0.83&	3.53&	44.70&	364.67&-&-\\
% (C1)&0.05&	0.17&	1.07&	3.02&	11.68&	63.87&	395.96&	2362.02&	19574.71 \\
% (C2)&0.06&	0.35&	1.51&	2.64&	10.53&	59.03&	362.99&	2239.51&	18507.45 \\
% (C3)&0.05&	0.16&	0.79&	2.37&	9.51&	55.76&-&-&-\\
% (C4)&0.08&	0.18&	0.46&	2.15&	13.30&	45.07&-&-&-\\
% (C5)&0.03&	0.12&	0.55&	2.17&	8.98&-&-&-& \\
% (C6)&0.02&	0.09&	0.31&	1.60&	5.82&-&-&-& \\
% (C7)&0.03&	0.29&	0.33&	1.65&	7.67&	37.55&	211.85&	1056.77&	6585.03 \\
% (C8)&0.02&	0.10&	0.23&	1.16&	4.17&	25.24&	158.01&	1048.85&	6990.33 \\
% (C9)&0.04&	0.23&	0.30&	1.43&	5.93&	2928.81&-&-&-\\
% (C10)&0.02&	0.08&	0.20&	1.06&	3.45&	51.26&-&-&-\\
% \hline
% \end{tabular}
% \end{table}
% \end{center}



% \renewcommand{\arraystretch}{1.5}
% \begin{table}[h!]
%     \centering
%     \caption{Backend and supported functions in \texttt{meent}}
%     \label{tab:supported functions}
%     \begin{center}
%         \begin{tabular}[c]{|c|c|c|c|c|c|}
%         \hline
%         Backend & Modeling & Fourier Series & RCWA & Gradient & GPU \\
%         \hline

%         \multirow{3}{10em}{\centering Numpy} & \multirow{2}{5em}{\centering Raster} & Continuous & O & X & X \\ \cline{3-6}
%          & & Discrete & O & X & X \\ \cline{2-6}
%          & \multirow{1}{5em}{\centering Vector} & Continuous & O & X & X \\ \hline

%          \multirow{3}{10em}{\centering Jax} & \multirow{2}{5em}{\centering Raster} & Continuous & O & X & O \\ \cline{3-6}
%          & & Discrete & O & O & O \\ \cline{2-6}
%          & \multirow{1}{5em}{\centering Vector} & Continuous & O & O & O \\ \hline

%          \multirow{3}{10em}{\centering PyTorch} & \multirow{2}{5em}{\centering Raster} & Continuous & O & X & O \\ \cline{3-6}
%          & & Discrete & O & O & O \\ \cline{2-6}
%          & \multirow{1}{5em}{\centering Vector} & Continuous & O & O & O \\ \hline

%         \end{tabular}
%     \end{center}
% \end{table}

# Optical Critical Dimension (OCD) metrology

## Spectrum fitting
`arxiv_ocd_optimize.py` conducts spectrum fitting with chosen algorithm. It can be designated by addtional argument. 
`python arxiv_ocd_optimize.py 0` uses Momentum. Algorithm list is in the code, and users can add their own or
adopt from PyTorch modules.

## Hyperparameter sweep
`arxiv_hyperparameter_sweep.ipynb` is a notebook file plotting the result of hyperparameter sweep for 5 optimization
algorithms.

## Plot 5 algos with chosen hyperparameters
`arxiv_optimization_chosen5algos.ipynb` 
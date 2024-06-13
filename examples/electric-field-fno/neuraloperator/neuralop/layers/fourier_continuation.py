import torch
import torch.nn as nn
import numpy as np
from numpy.polynomial.legendre import Legendre


class FCLegendre(nn.Module):
    def __init__(self, n, d, dtype=torch.float32):
        super().__init__()

        self.dtype = dtype

        self.compute_extension_matrix(n, d)
    
    def compute_extension_matrix(self, n, d):
        self.n = n
        self.d = d 

        a = 0.0
        h = 0.1

        #Generate grid for the extension
        total_points = 2*n + d
        full_grid = a + h*np.arange(total_points, dtype=np.float64)
        fit_grid = np.concatenate((full_grid[0:self.n], full_grid[-self.n:]), 0)
        extension_grid = full_grid[self.n:-self.n]

        #Initialize orthogonal polynomial system
        I = np.eye(2*self.n, dtype=np.float64)
        polynomials = []
        for j in range(2*self.n):
            polynomials.append(Legendre(I[j,:], domain=[full_grid[0], full_grid[-1]]))

        #Compute data and evaluation matrices
        X = np.zeros((2*self.n,2*self.n), dtype=np.float64)
        Q = np.zeros((self.d, 2*self.n), dtype=np.float64)
        for j in range(2*self.n):
            Q[:,j] = polynomials[j](extension_grid)
            X[:,j] = polynomials[j](fit_grid)

        #Compute extension matrix
        ext_mat = np.matmul(Q, np.linalg.pinv(X, rcond=1e-31))
        self.register_buffer('ext_mat', torch.from_numpy(ext_mat).to(dtype=self.dtype))
        self.register_buffer('ext_mat_T', self.ext_mat.T.clone())

        return self.ext_mat

    def extend_left_right(self, x):
        right_bnd = x[...,-self.n:]
        left_bnd = x[...,0:self.n]

        y = torch.cat((right_bnd, left_bnd), dim=-1)
        ext = torch.matmul(y, self.ext_mat_T)

        return torch.cat((x, ext), dim=-1)
    
    def extend_top_bottom(self, x):
        bottom_bnd = x[...,-self.n:,:]
        top_bnd = x[...,0:self.n,:]

        y = torch.cat((bottom_bnd, top_bnd), dim=-2)
        ext = torch.matmul(self.ext_mat, y)

        return torch.cat((x, ext), dim=-2)
    
    def extend2d(self, x):
        x = self.extend_left_right(x)
        x = self.extend_top_bottom(x)

        return x
    
    def forward(self, x):
        return self.extend2d(x)

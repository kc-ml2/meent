#import sys
#sys.path.append("/home/seolho/paper/meent")

import torch
import torch.nn as nn
from meent.on_torch.convolution_matrix import to_conv_mat
from meent.rcwa import call_solver

import numpy as np
import copy

fourier_order = 2
mode_key = 2
dtype = 0 
device = 0

'''
class Optimizer:
    def __init__(self, optimizer, mee, target_name, lr = 0.001):
        self.mee = mee
        self.target_name = target_name
        self.target = getattr(self.mee, target_name)
        
        if not isinstance(target, torch.Tensor):
            self.target = torch.Tensor(self.target)
        self.target.requires_grad = True
        
        if isinstance(optimizer,str):
            optimizer = getattr(torch.optim, optimizer)
        self.optimizer = optimizer([self.target], lr = lr)
        
    def optimize(self, iterations = 1000):
        for iteration in range(iterations):
            E_conv_all = to_conv_mat(self.mee.ucell, fourier_order)
            o_E_conv_all = to_conv_mat(1 / self.mee.ucell, fourier_order)

            de_ri, de_ti, _, _, _ = self.mee.solve(self.mee.wavelength, E_conv_all, o_E_conv_all)

            self.optimizer.zero_grad()
            loss = self.loss(de_ti)
            loss.backward()
            self.optimizer.step()
            
            setattr(self.mee, self.target_name, self.target)
            
            print(loss)
            
    def loss(self, de_ti):
        return -de_ti[3, 2]
'''

class Optimizer:
    def __init__(self, optimizer, solver, target, lr = 0.001):
        self.solver = solver
        
        self.target = target
        if not isinstance(target, torch.Tensor):
            self.target = torch.Tensor(self.target)
        self.target.requires_grad = True
        
        if isinstance(optimizer,str):
            optimizer = getattr(torch.optim, optimizer)
        self.optimizer = optimizer([self.target], lr = lr)
        
    def optimize(self, iterations = 1000):
        for iteration in range(iterations):
            E_conv_all = to_conv_mat(self.solver.ucell, fourier_order)
            o_E_conv_all = to_conv_mat(1 / self.solver.ucell, fourier_order)

            de_ri, de_ti, _, _, _ = self.solver.solve(self.solver.wavelength, E_conv_all, o_E_conv_all)

            self.optimizer.zero_grad()
            loss = self.loss(de_ti)
            loss.backward()
            self.optimizer.step()
            
            print(loss)
            
    def loss(self, de_ti):
        return -de_ti[3, 2]
    
def load_setting(mode_key, dtype, device):
    grating_type = 2

    pol = 1  # 0: TE, 1: TM

    n_I = 1  # n_incidence
    n_II = 1  # n_transmission

    theta = 0
    phi = 0
    psi = 0 if pol else 90

    wavelength = 900

    ucell_materials = [1, 3.48]
    fourier_order = 2

    thickness, period = [1120.], [1000, 1000]

    ucell = np.array(
        [[
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
        ]]
    )

    if mode_key == 0:
        device = 0
        type_complex = np.complex128 if dtype == 0 else np.complex64
        ucell = ucell.astype(type_complex)

    elif mode_key == 1:  # JAX
        jax.config.update('jax_platform_name', 'cpu') if device == 0 else jax.config.update('jax_platform_name', 'gpu')

        if dtype == 0:
            jax.config.update("jax_enable_x64", True)
            type_complex = jnp.complex128
            ucell = ucell.astype(jnp.float64)
            ucell = jnp.array(ucell, dtype=jnp.float64)

        else:
            type_complex = jnp.complex64
            ucell = ucell.astype(jnp.float32)
            ucell = jnp.array(ucell, dtype=jnp.float32)


    else:  # Torch
        device = torch.device('cpu') if device == 0 else torch.device('cuda')
        type_complex = torch.complex128 if dtype == 0 else torch.complex64

        if dtype == 0:
            ucell = torch.tensor(ucell, dtype=torch.float64, device=device)
        else:
            ucell = torch.tensor(ucell, dtype=torch.float32, device=device)

    return grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order,\
           type_complex, device, ucell



grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order, \
type_complex, device, ucell = load_setting(mode_key, dtype, device)

ucell.requires_grad = True

solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                     psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                     ucell_materials=ucell_materials, thickness=thickness, device=device,
                     type_complex=type_complex, )


optim = Optimizer('Adam', solver, solver.ucell, lr = 0.001)
optim.optimize()
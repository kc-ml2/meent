"""
This code is not supported and no plan to modify as of now.
"""

import torch
import torcwa
import numpy as np
import matplotlib.pyplot as plt


class TORCWA:
    def __init__(self, n_I=1., n_II=1., theta=0., phi=0., fourier_order=40, period=(100,),
                 wavelength=900., pol=1, ucell=None, thickness=(325,), device=0, *args, **kwargs):

        self.fourier_order = [fourier_order, 0]
        self.fourier_order = [0, fourier_order]
        # self.fto = [fto, fto]

        self.pol = pol
        if type(period) in (int, float):
            self.period = [period, 0]
            self.period = [1000000, period]
        elif len(period) == 1:
            self.period = [period[0], 1]
            self.period = [1000, period[0]]
        else:
            self.period = period

        self.theta = theta
        self.phi = phi

        self.n_I = n_I  # dielectric for layer 1 (uniform)
        self.n_II = n_II  # dielectric for layer N (uniform)

        self.thickness = torch.tensor(thickness)  # thickness of patterned layer

        self.ucell = torch.tensor(ucell**2)
        self.Nx = ucell.shape[2]
        self.Ny = ucell.shape[2]

        self.wavelength = wavelength
        self.device = device

        if device == 0:
            self.device = torch.device('cpu')
        elif device == 1:
            self.device = torch.device('cuda')
        else:
            raise ValueError

    def run(self):

        sim = torcwa.rcwa(freq=1/self.wavelength, order=self.fourier_order, L=self.period, dtype=torch.complex128, device=self.device)
        sim.add_input_layer(eps=self.n_I)
        sim.set_incident_angle(inc_ang=self.theta, azi_ang=self.phi)

        for layer, thick in zip(self.ucell,self.thickness):
            sim.add_layer(thickness=thick, eps=layer)
        # sim.add_output_layer(eps=self.n_bot)  # This line makes error.
        sim.solve_global_smatrix()

        order =[
            [-1,0],
            [0,-1],
            [0,0],
            [0,1],
            [1,0],
        ]

        a = (abs(sim.S_parameters(orders=order, direction='forward', port='t', polarization='ss', ref_order=[0, 0]))**2)
        b = (abs(sim.S_parameters(orders=order, direction='forward', port='t', polarization='pp', ref_order=[0, 0]))**2)
        c = (abs(sim.S_parameters(orders=order, direction='forward', port='t', polarization='sp', ref_order=[0, 0]))**2)
        d = (abs(sim.S_parameters(orders=order, direction='forward', port='t', polarization='ps', ref_order=[0, 0]))**2)

        e = (abs(sim.S_parameters(orders=order, direction='forward', port='r', polarization='ss', ref_order=[0, 0]))**2)
        f = (abs(sim.S_parameters(orders=order, direction='forward', port='r', polarization='pp', ref_order=[0, 0]))**2)
        g = (abs(sim.S_parameters(orders=order, direction='forward', port='r', polarization='sp', ref_order=[0, 0]))**2)
        h = (abs(sim.S_parameters(orders=order, direction='forward', port='r', polarization='ps', ref_order=[0, 0]))**2)

        # a = (abs(sim.S_parameters(orders=order, direction='forward', port='t', pol='xx', ref_order=[0, 0]))**2)
        # b = (abs(sim.S_parameters(orders=order, direction='forward', port='t', pol='yy', ref_order=[0, 0]))**2)
        # c = (abs(sim.S_parameters(orders=order, direction='forward', port='t', pol='xy', ref_order=[0, 0]))**2)
        # d = (abs(sim.S_parameters(orders=order, direction='forward', port='t', pol='yx', ref_order=[0, 0]))**2)
        #
        # e = (abs(sim.S_parameters(orders=order, direction='forward', port='r', pol='xx', ref_order=[0, 0]))**2)
        # f = (abs(sim.S_parameters(orders=order, direction='forward', port='r', pol='yy', ref_order=[0, 0]))**2)
        # g = (abs(sim.S_parameters(orders=order, direction='forward', port='r', pol='xy', ref_order=[0, 0]))**2)
        # h = (abs(sim.S_parameters(orders=order, direction='forward', port='r', pol='yx', ref_order=[0, 0]))**2)

        return e+f+g+h, a+b+c+d


if __name__ == '__main__':

    option = {
        'grating_type': 1,
        'pol': 1,
        'n_top': 1,
        'n_bot': 1,
        'theta': 1,
        'phi': 1,
        'wavelength': 100,
        'fto': 1,
        'thickness': [1000, 300],
        'period': [1000],
        'fourier_type': 1,
        'ucell': np.array(
            [
                [[3.1, 1.1, 1.2, 1.6, 3.1]*10],
                [[3, 3, 1, 1, 1]*10],
            ]
        ),
    }
    res = TORCWA(**option)

    de_ri, de_ti = res.run()
    print(de_ri.sum() + de_ti.sum())
    print(de_ri)
    print(de_ti)

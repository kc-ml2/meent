import torch
import torcwa
import numpy as np
import matplotlib.pyplot as plt


class TORCWA:
    def __init__(self, n_I=1., n_II=1., theta=0., phi=0., fourier_order=40, period=(100,),
                 wavelength=900., pol=1, ucell=None, thickness=(325,), device=0, *args, **kwargs):

        # super().__init__(grating_type)

        # Truncation order (actual number might be smaller)
        self.fourier_order = [fourier_order, fourier_order]

        self.pol = pol
        if len(period) == 1:
            # Not sure about this. Didn't check.
            self.period = [period, 0]
        else:
            self.period = period

        self.theta = theta
        self.phi = phi

        self.n_I = n_I  # dielectric for layer 1 (uniform)
        self.n_II = n_II  # dielectric for layer N (uniform)

        self.thickness = np.array(thickness)  # thickness of patterned layer

        self.ucell = ucell**2
        self.Nx = ucell.shape[2]
        self.Ny = ucell.shape[1]

        self.wavelength = wavelength
        self.device = device

    def run(self):

        sim = torcwa.rcwa(freq=1/self.wavelength, order=self.fourier_order, L=self.period, dtype=torch.complex128, device=self.device)
        sim.add_input_layer(eps=self.n_I)
        sim.set_incident_angle(inc_ang=self.theta, azi_ang=self.phi)

        for layer, thick in zip(self.ucell,self.thickness):
            sim.add_layer(thickness=thick, eps=layer)
        # sim.add_output_layer(eps=self.n_II)  # This line makes error.
        sim.solve_global_smatrix()

        order =[
            [-1,0],
            [0,-1],
            [0,0],
            [0,1],
            [1,0],
        ]

        a = (abs(sim.S_parameters(orders=order, direction='forward', port='t', polarization='xx', ref_order=[0, 0]))**2)
        b = (abs(sim.S_parameters(orders=order, direction='forward', port='t', polarization='yy', ref_order=[0, 0]))**2)
        c = (abs(sim.S_parameters(orders=order, direction='forward', port='t', polarization='xy', ref_order=[0, 0]))**2)
        d = (abs(sim.S_parameters(orders=order, direction='forward', port='t', polarization='yx', ref_order=[0, 0]))**2)

        e = (abs(sim.S_parameters(orders=order, direction='forward', port='r', polarization='xx', ref_order=[0, 0]))**2)
        f = (abs(sim.S_parameters(orders=order, direction='forward', port='r', polarization='yy', ref_order=[0, 0]))**2)
        g = (abs(sim.S_parameters(orders=order, direction='forward', port='r', polarization='xy', ref_order=[0, 0]))**2)
        h = (abs(sim.S_parameters(orders=order, direction='forward', port='r', polarization='yx', ref_order=[0, 0]))**2)

        return e+f+g+h, a+b+c+d


if __name__ == '__main__':
    from meent.testcase import load_setting

    mode = 2
    dtype = 0
    device = 0
    grating_type = 2
    pre = load_setting(mode, dtype, device, grating_type)

    res = TORCWA(**pre)

    de_ri, de_ti = res.run()
    print(de_ri.sum() + de_ti.sum())
    print(de_ri)
    print(de_ti)

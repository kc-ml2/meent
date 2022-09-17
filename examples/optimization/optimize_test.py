import numpy as np

from meent.rcwa import RCWA


class RCWAOptimizer:

    def __init__(self, gt, model):
        self.gt = gt
        self.model = model
        pass

    def get_difference(self):
        spectrum_gt = np.hstack(self.gt.spectrum_R, self.gt.spectrum_T)
        spectrum_model = np.hstack(self.model.spectrum_R, self.model.spectrum_T)
        residue = spectrum_model - spectrum_gt
        loss = np.linalg.norm(residue)


if __name__ == '__main__':
    grating_type = 0

    gt = RCWA(grating_type)
    model = RCWA(grating_type)
    pass

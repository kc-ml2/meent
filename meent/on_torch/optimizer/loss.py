import torch


class LossDeflector:
    def __init__(self, x_order=0, y_order=0):
        self.x_order = x_order
        self.y_order = y_order

    def __call__(self, value, *args, **kwargs):
        de_ri, de_ti = value

        if len(de_ti.shape) == 1:
            c_x = de_ti.shape[0] // 2
            res = de_ti[c_x + self.x_order]
        elif len(de_ti.shape) == 2:
            c_x = de_ti.shape[0] // 2
            c_y = de_ti.shape[1] // 2
            res = de_ti[c_x + self.x_order, c_y + self.y_order]
        else:
            raise ValueError

        return res


class LossSpectrumL2:
    def __init__(self):
        pass

    def __call__(self, pred, target, *args, **kwargs):
        gap = torch.linalg.norm(pred, target)
        return gap

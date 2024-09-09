import numpy as np


def meeinv(x, use_pinv=False):
    if use_pinv:
        res = np.linalg.pinv(x)
    else:
        res = np.linalg.inv(x)

    return res

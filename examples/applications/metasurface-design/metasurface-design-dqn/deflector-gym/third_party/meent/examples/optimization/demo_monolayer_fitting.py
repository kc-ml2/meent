import numpy as np
import matplotlib.pyplot as plt

from reticolo_interface.ClassReticolo import ClassReticolo
from meent.rcwa import RCWA


if __name__ == '__main__':

    incident_angle = 0
    detector_angle = 90
    wl_range = np.linspace(100, 2000, 100)

    spectrum_gt_R = np.zeros(len(wl_range))
    spectrum_gt_T = np.zeros(len(wl_range))
    spectrum_model_R = np.zeros(len(wl_range))
    spectrum_model_T = np.zeros(len(wl_range))

    config = (0,  # incident_angle
              0,  # detector_angle
              10,  # nn
              0.7,  # period
              1,  # tetm
              )

    textures = tuple([1, 1.5, 3,
                      ([-4, 4], [1, 1.5]),
                      ([-1, -0.5, 0.5, 1], [2, 1.3, 1.5, 1.3]),
                      2.5
                      ],
                     )

    profile_gt = tuple([[0, 110, 0], [1, 2, 3]])
    AA = ClassReticolo()

    for i, wl in enumerate(wl_range):

        R, T = AA.cal_reflectance(config, textures, profile_gt, wl)
        spectrum_gt_R[i] = R
        spectrum_gt_T[i] = T

    # Parameter fitting

    param_to_fit = 100

    profile_model = tuple([[0, param_to_fit, 0], [1, 2, 3]])

    while True:
        for i, wl in enumerate(wl_range):

            R, T = AA.cal_reflectance(config, textures, profile_model, wl)
            spectrum_model_R[i] = R
            spectrum_model_T[i] = T

        loss = np.linalg.norm(spectrum_gt_R - spectrum_model_R)

        plt.plot(wl_range, spectrum_gt_R)
        plt.plot(wl_range, spectrum_model_R)
        plt.title(f'parameter to fit:{param_to_fit}, loss:{loss}')
        plt.show()

        # Gradient descent with f(x) unknown
        update_rate = 0.01
        param_to_fit_new = param_to_fit * (1 + update_rate)

        profile_model_new = tuple([[0, param_to_fit_new, 0], [1, 2, 3]])

        for i, wl in enumerate(wl_range):

            R, T = AA.cal_reflectance(config, textures, profile_model_new, wl)
            spectrum_model_R[i] = R
            spectrum_model_T[i] = T

        loss_new = np.linalg.norm(spectrum_gt_R - spectrum_model_R)

        if abs(loss_new) <= 0.05:

            param_to_fit = param_to_fit_new
            profile_model = profile_model_new
            break
        elif loss_new > loss:
            param_to_fit_new = param_to_fit * (1 - update_rate)
        elif loss_new < loss:
            param_to_fit_new = param_to_fit * (1 + update_rate)
        else:
            break

        param_to_fit = param_to_fit_new
        profile_model = profile_model_new

    plt.plot(wl_range, spectrum_gt_R)
    plt.plot(wl_range, spectrum_model_R)
    plt.title(f'Result parameter to fit:{param_to_fit}, loss:{loss}')
    plt.show()

    pass

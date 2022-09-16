import numpy as np
import matplotlib.pyplot as plt

from reticolo_interface.ClassReticolo import ClassReticolo


def fitting(param_list_to_fit):

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

    param_list_gt = [110, 85, 65]
    profile_gt = tuple([[0] + param_list_gt + [0], [1, 2, 3, 4, 5]])

    AA = ClassReticolo()

    for i, wl in enumerate(wl_range):

        R, T = AA.cal_reflectance(config, textures, profile_gt, wl)
        spectrum_gt_R[i] = R
        spectrum_gt_T[i] = T

    # Parameter fitting
    profile_model = tuple([[0] + param_list_to_fit + [0], [1, 2, 3, 4, 5]])

    for i in range(20):

        for j, wl in enumerate(wl_range):
            R, T = AA.cal_reflectance(config, textures, profile_model, wl)
            spectrum_model_R[j] = R
            spectrum_model_T[j] = T

        loss1 = np.linalg.norm(spectrum_model_R - spectrum_gt_R)
        loss2 = np.linalg.norm(spectrum_model_T - spectrum_gt_T)
        loss = np.linalg.norm((loss1, loss2))

        print(np.array(param_list_to_fit).round(4), loss)

        if loss <= 0.05:  # converged
            print('converged')
            break

        plt.plot(wl_range, spectrum_gt_R, marker='x')
        plt.plot(wl_range, spectrum_model_R, marker='x')
        plt.title(f'parameter to fit:{np.array(param_list_to_fit).round(4)}, loss:{round(loss, 4)}')
        plt.show()
        param_list_to_fit_new = param_list_to_fit.copy()

        # Gradient descent with f(x) unknown
        for j, param_to_fit in enumerate(param_list_to_fit_new):

            unit_step = 0.001
            learning_rate = 0.01

            param_to_fit_new = param_to_fit * (1 + unit_step)
            param_list_to_fit_new[j] = param_to_fit_new
            profile_model_new = tuple([[0] + param_list_to_fit_new + [0], [1, 2, 3, 4, 5]])

            spectrum_model_R_right = np.zeros(len(wl_range))
            spectrum_model_T_right = np.zeros(len(wl_range))

            for k, wl in enumerate(wl_range):
                R, T = AA.cal_reflectance(config, textures, profile_model_new, wl)
                spectrum_model_R_right[k] = R
                spectrum_model_T_right[k] = T

            loss1 = np.linalg.norm(spectrum_model_R_right - spectrum_gt_R)
            loss2 = np.linalg.norm(spectrum_model_T_right - spectrum_gt_T)
            loss_right = np.linalg.norm((loss1, loss2))

            param_to_fit_new = param_to_fit * (1 - unit_step)
            param_list_to_fit_new[j] = param_to_fit_new
            profile_model_new2 = tuple([[0] + param_list_to_fit_new + [0], [1, 2, 3, 4, 5]])

            spectrum_model_R_left = np.zeros(len(wl_range))
            spectrum_model_T_left = np.zeros(len(wl_range))

            for k, wl in enumerate(wl_range):
                R, T = AA.cal_reflectance(config, textures, profile_model_new2, wl)
                spectrum_model_R_left[k] = R
                spectrum_model_T_left[k] = T

            loss1 = np.linalg.norm(spectrum_model_R_left - spectrum_gt_R)
            loss2 = np.linalg.norm(spectrum_model_T_left - spectrum_gt_T)
            loss_left = np.linalg.norm((loss1, loss2))

            if loss < min(loss_right, loss_left):
                continue

            else:
                if loss_right <= loss_left:
                    param_list_to_fit[j] = param_to_fit * (1 + learning_rate * (loss - loss_right) / unit_step)
                else:
                    param_list_to_fit[j] = param_to_fit * (1 - learning_rate * (loss - loss_left) / unit_step)

        profile_model = tuple([[0] + param_list_to_fit + [0], [1, 2, 3, 4, 5]])

        if param_list_to_fit == param_list_to_fit_new:
            print('local minimum')
            break

    else:
        print('failed to converge')

    plt.plot(wl_range, spectrum_gt_R, marker='x')
    plt.plot(wl_range, spectrum_model_R, marker='x')
    plt.title(f'Result parameter to fit:{np.array(param_list_to_fit).round(4)}, loss:{round(loss, 4)}')

    plt.show()

    pass


if __name__ == '__main__':
    list_of_param_list_to_fit = [[100, 85, 65], [100, 85, 75]]
    for param_list in list_of_param_list_to_fit:
        fitting(param_list)

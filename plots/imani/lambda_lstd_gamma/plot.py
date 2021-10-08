import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


def plot(results, multiplier=10):
    # blue = np.array([0.4, 0.4, 1.0])
    # orange = np.array([1., 0.5, 0.])
    # color = blue * color_gradient + (1-color_gradient) * orange
    n_res = results.shape[0]
    n_it = results.shape[1]
    ret_sg_m = np.mean(results, axis=0)
    ret_sg_t = 1.96 * np.std(results, axis=0) / np.sqrt(n_res)
    x = np.array(range(n_it))*multiplier
    plt.plot(x, ret_sg_m)
    plt.fill_between(x, ret_sg_m + ret_sg_t, ret_sg_m - ret_sg_t, alpha=0.5)


lambdas = [0., 0.8, 0.85, 0.9, 1.0]
for _lambda in lambdas:
    res = np.load("result_%.2f.npy" % _lambda)
    plot(res[:, ::10])

tikzplotlib.save('lstd_gamma_learning_curve.tex', table_row_sep=r"\\")
plt.show()
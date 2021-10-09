import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def plot(ax, results, _lambda):
    # blue = np.array([0.4, 0.4, 1.0])
    # orange = np.array([1., 0.5, 0.])
    # color = blue * color_gradient + (1-color_gradient) * orange
    n_res = results.shape[0]
    n_it = results.shape[1]
    ret_sg_m = np.mean(results, axis=0)
    ret_sg_t = 1.96 * np.std(results, axis=0) / np.sqrt(n_res)
    x = np.array(range(n_it))
    ax.plot(x, ret_sg_m, label=_lambda)
    ax.fill_between(x, ret_sg_m + ret_sg_t, ret_sg_m - ret_sg_t, alpha=0.5)


fig, ax = plt.subplots(1, 1)
lambdas = [0., 0.25, 0.50, 0.75, 1.]
for _lambda in lambdas:
    res = np.load("returns%.2f-1.00.npy" % _lambda)
    plot(ax, res, _lambda)

plt.legend(loc=1)
# axins = zoomed_inset_axes(ax, 2.5, loc=3) # zoom = 6
# for _lambda in lambdas:
#     res = np.load("returns%.2f-1.00.npy" % _lambda)
#     plot(axins, res, _lambda)
#
# # sub region of the original image
# x1, x2, y1, y2 = 0., 200., 0.14, 0.17
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)

tikzplotlib.save('rc_gamma_learning_curve.tex', table_row_sep=r"\\")
plt.show()
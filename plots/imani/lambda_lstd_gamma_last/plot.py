import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


def plot(results):
    orange = np.array([1, 0.498039215686275,0.0549019607843137])
    blue = np.array([0.12156862745098, 0.466666666666667, 0.705882352941177])
    n_res = results.shape[1]
    resolution = results.shape[0]
    ret_sg_m = np.mean(results, axis=1)
    ret_sg_t = 1.96 * np.std(results, axis=1) / np.sqrt(n_res)
    x = np.array(range(resolution))/(resolution-1)
    for i in range(resolution - 1):
        color = orange*(1 - x[i]) + blue*x[i]
        color_alpha = color.tolist() + [0.1]
        plt.plot(x[i:i+2], ret_sg_m[i:i+2], color=color)
        plt.fill_between(x[i:i+2], (ret_sg_m + ret_sg_t)[i:i+2], (ret_sg_m - ret_sg_t)[i:i+2], alpha=0.5, color=color,
                         edgecolor=color_alpha, lw=0.01)


ret_gamma = np.load("ret_gamma.npy")

plot(np.array(ret_gamma))

plt.xlim(0., 1.2)
plt.ylabel(r"$J(\theta)$")
plt.xlabel("$\lambda$")

tikzplotlib.save("last_performance.tex", table_row_sep=r"\\")
plt.savefig("last_performance.pdf")

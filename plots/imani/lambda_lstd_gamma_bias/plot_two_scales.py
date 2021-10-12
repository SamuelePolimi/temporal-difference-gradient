import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


mse_m = np.load("mse_m.npy")
bias_m = np.load("bias_m.npy")
var_m = np.load("var_m.npy")
mse_conf = np.load("mse_conf.npy")
bias_conf = np.load("bias_conf.npy")
var_conf = np.load("var_conf.npy")

res = len(mse_m)

fig, ax1 = plt.subplots()


x = np.linspace(0, 1, res)
color = 'tab:red'
ax1.set_xlabel('Samples')
ax1.set_ylabel('Bias', color=color)
ax1.plot(x, bias_m, color=color)
plt.fill_between(x, bias_m + bias_conf, bias_m - bias_conf, color=color, alpha=0.5)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Variance', color=color)  # we already handled the x-label with ax1
ax2.plot(x, var_m, color=color)
plt.fill_between(x, var_m + var_conf, var_m - var_conf, color=color, alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0., 0.5E-5)


fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("bias_variance_two_scales.pdf")
tikzplotlib.save("bias_variance_two_scales.tex", table_row_sep=r"\\")
plt.show()
#
# plt.plot(x, mse_m, label="mse")
# plt.fill_between(x, mse_m + mse_conf/8, mse_m - mse_conf/8, alpha=0.5) # Due to error in the main
# plt.plot(x, bias_m, label="bias")
# plt.fill_between(x, bias_m + bias_conf, bias_m - bias_conf, alpha=0.5)
# plt.plot(x, var_m, label="var")
# plt.fill_between(x, var_m + var_conf, var_m - var_conf, alpha=0.5)
#
# plt.legend(loc="best")
# plt.savefig("bias_variance.pdf")
# tikzplotlib.save("bias_variance.tex", table_row_sep=r"\\")
# plt.show()
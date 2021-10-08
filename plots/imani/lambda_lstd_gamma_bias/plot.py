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

x = np.linspace(0, 1, res)
plt.plot(x, mse_m, label="mse")
plt.fill_between(x, mse_m + mse_conf/8, mse_m - mse_conf/8, alpha=0.5) # Due to error in the main
plt.plot(x, bias_m, label="bias")
plt.fill_between(x, bias_m + bias_conf, bias_m - bias_conf, alpha=0.5)
plt.plot(x, var_m, label="var")
plt.fill_between(x, var_m + var_conf, var_m - var_conf, alpha=0.5)

plt.legend(loc="best")
plt.savefig("bias_variance.pdf")
tikzplotlib.save("bias_variance.tex", table_row_sep=r"\\")
plt.show()
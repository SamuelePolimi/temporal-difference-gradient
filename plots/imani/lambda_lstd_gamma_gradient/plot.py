import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

def get_color(_lambda):
    orange = np.array([1, 0.498039215686275,0.0549019607843137])
    blue = np.array([0.12156862745098, 0.466666666666667, 0.705882352941177])
    return orange*(1-_lambda) + blue*_lambda

xs = np.load("xs.npy")
ys = np.load("ys.npy")
ground_truth = np.load("ground_truth.npy")


class WebPlot:

    def __init__(self, _min, _max, n_dims, ax):
        self._n_dims = n_dims
        self._min = _min
        self._max = _max
        self._delta = _max - _min
        self._theta_diff = 2*np.pi /self._n_dims

        self._ax = ax

    def draw_axis(self):
        for i in range(self._n_dims):
            theta = self._theta_diff*i
            x = np.cos(theta) * 1.05
            y = np.sin(theta) * 1.05
            self._ax.plot([0, x], [0, y], color='gray')

    def draw_coordinates(self, val, axis=0):
        radius = (val-self._min)/self._delta
        theta = self._theta_diff*axis
        return np.cos(theta)*radius, np.sin(theta)*radius

    def draw_data(self, values, **kwargs):
        xs = []
        ys = []
        for i in [x for x in range(self._n_dims)] + [0]:
            x, y = self.draw_coordinates(values[i], i)
            xs.append(x)
            ys.append(y)
        self._ax.plot(xs, ys, **kwargs)

fig, ax = plt.subplots(1, 1)
web_plot = WebPlot(np.min(ys), np.max(ys), 8, ax)
web_plot.draw_axis()
web_plot.draw_data(np.zeros(8), color='gray', lw=1)
web_plot.draw_data(np.ones(8)*web_plot._max, color='gray', lw=1)

ranges = [x for x in range(len(xs))]
np.random.shuffle(ranges)
for i in ranges:
    lam = xs[i]
    data = ys[i]
    web_plot.draw_data(data, color=get_color(lam), lw=1, alpha=.5)
web_plot.draw_data(ground_truth, color='green', lw=2, label="Ground Truth")

ax.legend()
ax.set_aspect('equal')
tikzplotlib.save("gradient_lstd_gamma.tex", table_row_sep=r"\\")
plt.savefig("gradient_lstd_gamma.pdf")
plt.show()


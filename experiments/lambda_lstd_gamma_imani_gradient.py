
import numpy as np
import torch.optim
import matplotlib.pyplot as plt
import tikzplotlib
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})


from herl.classic_envs import get_imani_mdp
from herl.actor import TabularPolicy
from herl.rl_interface import RLTask
from herl.rl_analysis import MDPAnalyzer, bias_variance_estimate
from herl.rl_visualizer import BiasVarianceVisualizer
from herl.utils import ProgressBar, Printable, _one_hot
from herl.solver import RLCollector

from algorithms import LambdaLSTDGamma, SemiGradient
from settings import ImaniCounterexample


print(r"""EMPIRICAL IMANI COUNTEREXAMPLE
=========================

We detail the gradient prediction of LSTD\Gamma on the counterexample of Imani et al, 2018.

The output of this experiment will be found in `/plots/imani/lambda_lstd_gamma_gradient/`

""")


actor_aliasing = True
critic_aliasing = False

setting = ImaniCounterexample()

if actor_aliasing:
    analyzer = setting.analyzer
else:
    # TODO: to define
    analyzer = MDPAnalyzer(mdp_task, policy, None)


def critic_features(s, a):
    state = s
    n_codes = setting.n_states * setting.n_actions
    if critic_aliasing:
        state = setting.actor_features.codify_state(s)
        n_codes = (setting.n_states - 1) * setting.n_actions

    s_a_t = _one_hot(state.int() * setting.n_actions + a.int(), n_codes)
    return s_a_t

# ------------------------
# Dataset Generation
# ------------------------


n_samples = 500


def get_dataset():
    dataset = setting.mdp_task.get_empty_dataset(n_samples)
    collector = RLCollector(dataset, setting.mdp_task, setting.behavior_policy, episode_length=4)
    collector.collect_samples(n_samples)
    return dataset


lambdas = np.linspace(0., 1., 20)
# Closed form versions of LSTDGamma and SemiGradient

lstd_gamma = lambda dataset, _lambda: LambdaLSTDGamma(setting.policy, critic_features, setting.actor_features, dataset,
                           setting.mdp_task.get_descriptor(), regularization=0., _lambda=_lambda)

# --------------------------------------------
# Train the policy with a given algorithm
# --------------------------------------------
idx = 2

setting.policy.set_parameters(setting.init_parameters)
ground_truth = analyzer.get_policy_gradient()

estimator = lambda _lambda: lambda: lstd_gamma(get_dataset(), _lambda).get_gradient()

x_scatter = []
y_scatter = []


np.save("../plots/imani/lambda_lstd_gamma_gradient/ground_truth.npy", ground_truth)
exit()
for _lambda in lambdas:
    for _ in range(20):
        x_scatter.append(_lambda)
        y_scatter.append(estimator(_lambda)())

np.save("../plots/imani/lambda_lstd_gamma_gradient/xs.npy", x_scatter)
np.save("../plots/imani/lambda_lstd_gamma_gradient/ys.npy", y_scatter)
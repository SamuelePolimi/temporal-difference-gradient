"""
In this experiment, we use the MDP introduced by Imani, Graves and White in 2018. They show that semi-gradient
fails to find an optimal policy.
We want to compare our boostrapped gradient with semi-gradient.
However, Imani et al. consider an off-policy objective, while we consider the on-policy objective (still relying on off-policy data).
"""
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

We test LSTD\Gamma on the counterexample of Imani et al, 2018.

Differently from the original paper, we consider the classic return as metric to be optimized.
We also have a slightly different (but equivalent) treatment of the terminal state. In our case, the terminal 
state is just absorbing with reward=0. This modifies the on-policy distribution, but yields same expected return and 
same policy gradient.

We test 4 different scenarios:

1. no critic aliasing, no actor aliasing
2. no critic aliasing, actor aliasing (similar to original paper - but estimated critic)
3. critic aliasing, no actor aliasing
4. both critic and actor aliasing

The output of this experiment will be found in `/plots/imani/lstd_gamma/`

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
                           setting.mdp_task.get_descriptor(), regularization=1E-16, _lambda=_lambda)

# --------------------------------------------
# Train the policy with a given algorithm
# --------------------------------------------
idx = 2

setting.policy.set_parameters(setting.init_parameters)
ground_truth = analyzer.get_policy_gradient()

estimator = lambda _lambda: lambda: lstd_gamma(get_dataset(), _lambda).get_gradient()


def compute_bias(values):
    return np.mean(np.mean([value - ground_truth for value in values], axis=0)**2)


def compute_var(values):
    return np.mean(np.std(values, axis=0)**2)


def compute_mse(values):
    return np.mean([(value - ground_truth)**2 for value in values])


mse_m = []
mse_conf = []
bias_m = []
bias_conf = []
var_m = []
var_conf = []

n_batches = 50
pb = ProgressBar(Printable("Bias Variance Mse"), max_iteration=len(lambdas)*n_batches*20)
for _lambda in lambdas:
    mses = []
    biases = []
    variances = []
    for _ in range(n_batches):
        values = [[estimator(_lambda)(), pb.notify()][0] for _ in range(20)]
        mses.append(compute_mse(values))
        biases.append(compute_bias(values))
        variances.append(compute_var(values))
        pb.notify()

    bias_m.append(np.mean(biases))
    var_m.append(np.mean(variances))
    mse_m.append(np.mean(mses))

    bias_conf.append(1.96*np.std(biases)/np.sqrt(n_batches))
    var_conf.append(1.96*np.std(variances)/np.sqrt(n_batches))
    mse_conf.append(1.96*np.std(mses)/np.sqrt(n_batches))


np.save("../plots/imani/lambda_lstd_gamma_bias/bias_m.npy", bias_m)
np.save("../plots/imani/lambda_lstd_gamma_bias/var_m.npy", var_m)
np.save("../plots/imani/lambda_lstd_gamma_bias/mse_m.npy", mse_m)
np.save("../plots/imani/lambda_lstd_gamma_bias/bias_conf.npy", bias_conf)
np.save("../plots/imani/lambda_lstd_gamma_bias/var_conf.npy", var_conf)
np.save("../plots/imani/lambda_lstd_gamma_bias/mse_conf.npy", mse_conf)

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
from herl.rl_analysis import MDPAnalyzer
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
if actor_aliasing:
    lstd_gamma = [LambdaLSTDGamma(setting.policy, critic_features, setting.actor_features, get_dataset(),
                           setting.mdp_task.get_descriptor(), regularization=0., _lambda=_lambda) for _lambda in lambdas]
else:
    lstd_gamma = [LambdaLSTDGamma(setting.policy, critic_features, None, get_dataset(),
                     setting.mdp_task.get_descriptor(), regularization=0., _lambda=_lambda) for _lambda in
     lambdas]

# --------------------------------------------
# Train the policy with a given algorithm
# --------------------------------------------

setting.policy.set_parameters(setting.init_parameters)
ground_truth = analyzer.get_policy_gradient()


def train(policy_gradient):
    error = np.mean((policy_gradient.get_gradient() - ground_truth)**2)
    print(policy_gradient.name, error)
    return error


# ---------------------------------------------
# Plot the Mean Squared Error
# ---------------------------------------------

def plot(results):
    color = np.array([1., 0.5, 0.])
    n_res = results.shape[1]
    resolution = results.shape[0]
    ret_sg_m = np.mean(results, axis=1)
    ret_sg_t = 1.96 * np.std(results, axis=1) / np.sqrt(n_res)
    x = np.array(range(resolution))/resolution
    plt.plot(x, ret_sg_m, label=r"$\lambda$-LSTD$\Gamma$", color=color)
    plt.fill_between(x, ret_sg_m + ret_sg_t, ret_sg_m - ret_sg_t, alpha=0.5, color=color)


ret_gamma = [[train(alg) for _ in range(50)] for alg in lstd_gamma]

plot(np.array(ret_gamma))

plt.ylabel(r"$RMSE J(\theta)$")
plt.xlabel("$\lambda$")
plt.legend(loc="best")

critic_name = "critic_aliasing" if critic_aliasing else "no_critic_aliasing"
actor_name = "actor_aliasing" if actor_aliasing else "no_actor_aliasing"
tikzplotlib.save("../plots/imani/lambda_lstd_gamma_rmse/%s_%s.tex" % (actor_name, critic_name))
plt.savefig("../plots/imani/lambda_lstd_gamma_rmse/%s_%s.pdf" % (actor_name, critic_name))



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
dataset = setting.mdp_task.get_empty_dataset(n_samples)
collector = RLCollector(dataset, setting.mdp_task, setting.behavior_policy, episode_length=4)
collector.collect_samples(n_samples)

lambdas = [0., 0.7, 0.8, 0.9, 1.]
# Closed form versions of LSTDGamma and SemiGradient
if actor_aliasing:
    lstd_gamma = [LambdaLSTDGamma(setting.policy, critic_features, setting.actor_features, dataset,
                           setting.mdp_task.get_descriptor(), regularization=0., _lambda=_lambda) for _lambda in lambdas]
else:
    lstd_gamma = [LambdaLSTDGamma(setting.policy, critic_features, None, dataset,
                     setting.mdp_task.get_descriptor(), regularization=0., _lambda=_lambda) for _lambda in
     lambdas]

# --------------------------------------------
# Train the policy with a given algorithm
# --------------------------------------------


def train(policy_gradient):
    n_iterations = 1000
    ret = []
    setting.policy.set_parameters(setting.init_parameters)
    adam = torch.optim.Adam(setting.policy.parameters(), lr=0.01)
    pb = ProgressBar(Printable("Policy Gradient"), max_iteration=n_iterations, prefix='Progress %s' % policy_gradient.name)
    for i in range(n_iterations):
        policy_gradient.update_policy(adam)
        ret.append((1-setting.mdp_task.gamma) * analyzer.get_return().detach().numpy())
        pb.notify()
    return ret, setting.policy.get_parameters()


# ---------------------------------------------
# Plot the learning curve
# ---------------------------------------------

def plot(results, algorithm, color_gradient):
    blue = np.array([0.4, 0.4, 1.0])
    orange = np.array([1., 0.5, 0.])
    color = blue * color_gradient + (1-color_gradient) * orange
    n_res = results.shape[0]
    n_it = results.shape[1]
    ret_sg_m = np.mean(results, axis=0)
    ret_sg_t = 1.96 * np.std(results, axis=0) / np.sqrt(n_res)
    plt.plot(ret_sg_m, label=algorithm, color=color)
    plt.fill_between(np.array(range(n_it)), ret_sg_m + ret_sg_t, ret_sg_m - ret_sg_t, alpha=0.5, color=color)


ret_gamma = [np.array([train(alg)[0] for _ in range(5)]) for alg in lstd_gamma]

for ret, alg in zip(ret_gamma, lstd_gamma):
    plot(ret, alg.name, alg._lambda)

plt.ylabel(r"$J(\theta)$")
plt.xlabel("Gradient Updates")
plt.legend(loc="best")

critic_name = "critic_aliasing" if critic_aliasing else "no_critic_aliasing"
actor_name = "actor_aliasing" if actor_aliasing else "no_actor_aliasing"
tikzplotlib.save("../plots/imani/lambda_lstd_gamma/%s_%s.tex" % (actor_name, critic_name))
plt.savefig("../plots/imani/lambda_lstd_gamma/%s_%s.pdf" % (actor_name, critic_name))



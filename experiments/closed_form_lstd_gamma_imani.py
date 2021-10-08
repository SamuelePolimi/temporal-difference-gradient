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


from herl.classic_envs import get_imani_mdp
from herl.actor import TabularPolicy
from herl.rl_interface import RLTask
from herl.rl_analysis import MDPAnalyzer
from herl.utils import ProgressBar, Printable, _one_hot

from algorithms import ClosedLSTDGamma, ClosedSemiGradient
from settings import ImaniCounterexample


print(r"""IMANI COUNTEREXAMPLE IN CLOSED FORM
=========================

We test LSTD\Gamma on the counterexample of Imani et al, 2018, 
showing that our method achieves higher performance than the Semi-Gradient.

Differently from the original paper, we consider the classic return as metric to be optimized.
We also have a slightly different (but equivalent) treatment of the terminal state. In our case, the terminal 
state is just absorbing with reward=0. This modifies the on-policy distribution, but yields same expected return and 
same policy gradient.

The output of this experiment will be found in `/plots/imani/closed_form_lstd_gamma/counterexample.pdf` and `.tikz`

""")


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


setting = ImaniCounterexample()


# --------------------------------
# Critic Features
# --------------------------------
def critic_features(s, a):
    s_a_t = _one_hot(s * setting.n_actions + a, setting.n_states * setting.n_actions)
    return s_a_t


# Closed form versions of LSTDGamma and SemiGradient
# P.S. The critic features are perfect, while the actor feature are aliased
cf_bg = ClosedLSTDGamma(setting.mdp_task, setting.policy, critic_features, setting.actor_features,
                        setting.mu, setting.beta)
cf_sg = ClosedSemiGradient(setting.mdp_task, setting.policy, critic_features, setting.actor_features,
                           setting.mu, setting.beta)


# --------------------------------
# Train the policy
# --------------------------------

# Train the policy with a given algorithm
def train(policy_gradient):
    ret = []
    setting.policy.set_parameters(setting.init_parameters)
    adam = torch.optim.Adam(setting.policy.parameters(), lr=0.01)
    pb = ProgressBar(Printable("Policy Gradient"), max_iteration=1000, prefix='Progress %s' % policy_gradient.name)
    for i in range(1000):
        policy_gradient.update_policy(adam)
        ret.append((1-setting.mdp_task.gamma) * setting.analyzer.get_return().detach().numpy())
        pb.notify()
    return ret, setting.policy.get_parameters()


# Get and plot the learning curve and the learned parameters
ret_sg, param_sg = train(cf_sg)
ret_bg, param_bg = train(cf_bg)


# --------------------------------
# Plot the learning curves
# --------------------------------
plt.plot(ret_sg, label=cf_sg.name)
plt.plot(ret_bg, label=cf_bg.name)
plt.ylabel(r"$J(\theta)$")
plt.xlabel("Gradient Updates")
plt.legend(loc="best")
tikzplotlib.save("../plots/imani/closed_form_lstd_gamma/counterexample.tex")
plt.savefig("../plots/imani/closed_form_lstd_gamma/counterexample.pdf")


# --------------------------------
# Plot on console the optimized
# policy
# --------------------------------

# def print_policy(policy_params):
#     policy.set_parameters(policy_params)
#     for i, s in enumerate(mdp.get_states()):
#         print("S%d %s" % (i, [policy.get_prob(s, a) for a in mdp.get_actions()]))


# print("-"*50)
# print("Semi-Gradient Final Policy:")
# # print_policy(param_sg)
#
# print("-"*50)
# print(r"LSTD\Gamma Gradient Final Policy:")
# # print_policy(param_bg)
# print("-"*50)

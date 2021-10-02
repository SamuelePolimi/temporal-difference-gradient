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

from algorithms import ClosedLSTDGamma, ClosedSemiGradient


print(r"""IMANI COUNTEREXAMPLE IN CLOSED FORM
=========================

We test LSTD\Gamma on the counterexample of Imani et al, 2018, 
showing that our method achieves higher performance than the Semi-Gradient.

Differently from the original paper, we consider the classic return as metric to be optimized.
We also have a slightly different (but equivalent) treatment of the terminal state. In our case, the terminal 
state is just absorbing with reward=0. This modifies the on-policy distribution, but yields same expected return and 
same policy gradient.

The output of this experiment will be found in `plots/imani-counterexample.pdf` and `.tikz`

""")

# MPD from Imani et al. 2018
mdp, actor_features = get_imani_mdp()
n_states = 4
n_actions = 2

# Tabular SoftMax policy
policy = TabularPolicy(mdp)
# TODO: fix the initial parameters!
init_parameters = policy.get_parameters()

mdp_task = RLTask(mdp, mdp.get_initial_state_sampler(), max_episode_length=500, gamma=0.9)
analyzer = MDPAnalyzer(mdp_task, policy, actor_features)

# Define off-policy behavior (as in the original paper)

beta = np.array([[0.25, 0.75],
                 [0.25, 0.75],
                 [0.25, 0.75],
                 [0.25, 0.75]]
                 )

# Define the state distribution (similarly to the original paper, but with a positive probability on the terminal state)
mu = np.array([0.5/2, 0.125/2, 0.375/2, 0.5])


def critic_features(s, a):
    s_a_t = _one_hot(s * n_actions + a, n_states * n_actions)
    return s_a_t

# Closed form versions of LSTDGamma and SemiGradient
# P.S. The critic features are perfect, while the actor feature are aliased
cf_bg = ClosedLSTDGamma(mdp_task, policy, critic_features, actor_features, mu, beta)
cf_sg = ClosedSemiGradient(mdp_task, policy, critic_features, actor_features, mu, beta)


# Train the policy with a given algorithm
def train(policy_gradient):
    ret = []
    policy.set_parameters(init_parameters)
    adam = torch.optim.Adam(policy.parameters(), lr=0.01)
    pb = ProgressBar(Printable("Policy Gradient"), max_iteration=1000, prefix='Progress %s' % policy_gradient.name)
    for i in range(1000):
        policy_gradient.update_policy(adam)
        ret.append((1-mdp_task.gamma) * analyzer.get_return().detach().numpy())
        pb.notify()
    return ret, policy.get_parameters()


# Get and plot the learning curve and the learned parameters
ret_sg, param_sg = train(cf_sg)
ret_bg, param_bg = train(cf_bg)

plt.plot(ret_sg, label=cf_sg.name)
plt.plot(ret_bg, label=cf_bg.name)
plt.ylabel(r"$J(\theta)$")
plt.xlabel("Gradient Updates")
plt.legend(loc="best")
tikzplotlib.save("../plots/imani-counterexample.tex")
plt.savefig("../plots/imani-counterexample.pdf")


# def print_policy(policy_params):
#     policy.set_parameters(policy_params)
#     for i, s in enumerate(mdp.get_states()):
#         print("S%d %s" % (i, [policy.get_prob(s, a) for a in mdp.get_actions()]))


print("-"*50)
print("Semi-Gradient Final Policy:")
# print_policy(param_sg)

print("-"*50)
print(r"LSTD\Gamma Gradient Final Policy:")
# print_policy(param_bg)
print("-"*50)

# print("Bootstrapped policy gradient: %s" % cf_bg.get_td_gradient(mu, beta))

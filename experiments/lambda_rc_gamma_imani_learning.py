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

from algorithms import LambdaRCGamma
from settings import ImaniCounterexample

print(r"""Learn the gradient of IMANI COUNTEREXAMPLE
=========================

We test GammaRC to see if it learns the correct gradient.

The output of this experiment will be found in `/plots/imani/gamma_rc_eval/`

""")


actor_aliasing = True
critic_aliasing = False

setting = ImaniCounterexample()

if actor_aliasing:
    analyzer = setting.analyzer
else:
    # TODO: fix it!!!
    analyzer = MDPAnalyzer(mdp_task, policy, None)

beta = 0.


def critic_features(s, a):
    state = s
    n_codes = setting.n_states * setting.n_actions
    if critic_aliasing:
        state = setting.actor_features.codify_state(s)
        n_codes = (setting.n_states - 1) * setting.n_actions

    s_a_t = _one_hot(state.int() * setting.n_actions + a.int(), n_codes)
    return s_a_t


actor_feature_parameters = None
if actor_aliasing:
    actor_feature_parameters = setting.actor_features

sa = np.array([[s, a] for s in setting.mdp.get_states()
               for a in setting.mdp.get_actions()
       ])

true_q = analyzer.get_Q(sa[:, 0:1], sa[:, 1:])
true_gamma = analyzer.get_Gamma(sa[:, 0:1], sa[:, 1:])

n_critic_features = 8 if not critic_aliasing else 6


def experiment(_lambda):

    gamma_rc = LambdaRCGamma(setting.policy, critic_features, actor_feature_parameters, setting.mdp_task.get_descriptor(), n_critic_features,
                   regularization=beta, learning_rate=0.01, decreasing_learning_rate=False, _lambda=_lambda)

    setting.policy.set_parameters(setting.init_parameters)
    adam = torch.optim.Adam(setting.policy.parameters(), lr=0.01)

    critic_error = []
    gradient_error = []

    setting.mdp_task.max_episode_length = 5
    n_iterations = 1000

    j_returns = []

    pb = ProgressBar(Printable("Policy Gradient"), max_iteration=n_iterations, prefix='Gamma-RC')
    for it in range(n_iterations):
        s = setting.mdp_task.reset()
        pb.notify()
        for i in range(5):
            a = setting.behavior_policy.get_action(s)
            ret = setting.mdp_task.step(a)
            r, s_n, t = [ret[k] for k in ["reward", "next_state", "terminal"]]
            gamma_rc.update_critic(s, a, r[0], s_n, t[0])
            gamma_rc.update_gradient(s, a, s_n, t[0])
            gamma_rc.update_policy(adam, s, i==4)
            # Critic Analisys
            # q_gamma = gamma_rc.get_Q(sa[:, 0:1], sa[:, 1:])
            # error = np.mean((q_gamma - true_q)**2)
            # # print("critic error", error)
            # critic_error.append(error)
            #
            # # Gradient Analysis
            # gamma_gradient = gamma_rc.get_Gamma(sa[:, 0:1], sa[:, 1:])
            # # print(np.mean(np.abs(gamma_gradient - true_gamma)))
            # error = np.mean((gamma_gradient - true_gamma)**2)
            # # print("gradient error", error)
            # gradient_error.append(error)
            # gamma_rc.update_time(s, a)

            s = s_n

        # j_filter = j_filter * 0.99 + 0.01 * (1 - setting.mdp_task.gamma) * setting.mdp_task.episode(setting.policy)
        j_ret = (1-setting.mdp_task.gamma) * analyzer.get_return()
        # print(it, j_filter)
        j_returns.append(np.asscalar(j_ret))

    return j_returns


for _lambda in [0., 0.25, 0.5, 0.75, 1.]:
    print("Testing %.2f" % _lambda)
    returns = [experiment(_lambda) for _ in range(20)]
    np.save("../plots/imani/lambda_rc_gamma_learning/returns%.2f-%.2f.npy" % (_lambda, beta), returns)


def plot(results, algorithm):
    n_res = results.shape[0]
    n_it = results.shape[1]
    ret_sg_m = np.mean(results, axis=0)
    ret_sg_t = 1.96 * np.std(results, axis=0) / np.sqrt(n_res)
    plt.plot(ret_sg_m, label=algorithm)
    plt.fill_between(np.array(range(n_it)), ret_sg_m + ret_sg_t, ret_sg_m - ret_sg_t, alpha=0.5)

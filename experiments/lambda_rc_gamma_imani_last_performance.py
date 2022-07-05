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
import time

print(r"""Learn the gradient of IMANI COUNTEREXAMPLE
=========================

We measure the last performance of TDRC\Gamma on Imani's MDP.

The output of this experiment will be found in `/plots/imani/lambda_rc_gamma_last/`

""")


actor_aliasing = True
critic_aliasing = False

setting = ImaniCounterexample()

if actor_aliasing:
    analyzer = setting.analyzer
else:
    # TODO: fix it!!!
    analyzer = MDPAnalyzer(mdp_task, policy, None)

beta = 1.0
pure_td = False


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
                   regularization=beta, learning_rate=0.1, decreasing_learning_rate=False, _lambda=_lambda, pure_td=pure_td)

    setting.policy.set_parameters(setting.init_parameters)
    adam = torch.optim.Adam(setting.policy.parameters(), lr=0.001) # Shall use 0.01 was 0.1

    critic_error = []
    gradient_error = []

    setting.mdp_task.max_episode_length = 5
    n_iterations = 10000

    j_returns = []

    start = time.time()
    pb = ProgressBar(Printable("Policy Gradient"), max_iteration=n_iterations, prefix='Gamma-RC')
    for it in range(n_iterations):
        s = setting.mdp_task.reset()
        gamma_rc.reset()
        pb.notify()
        for i in range(5):
            a = setting.behavior_policy.get_action(s)
            ret = setting.mdp_task.step(a)
            r, s_n, t = [ret[k] for k in ["reward", "next_state", "terminal"]]
            gamma_rc.update_critic(s, a, r[0], s_n, t[0])
            gamma_rc.update_gradient(s, a, s_n, t[0])
            gamma_rc.update_policy(adam, s)

            s = s_n

        j_ret = (1-setting.mdp_task.gamma) * analyzer.get_return()
        j_returns.append(np.asscalar(j_ret))

    print("time: ", time.time() - start)

    return j_returns


for _lambda in np.linspace(0, 1, 20):
    print("Testing %.2f" % _lambda)
    returns = [experiment(_lambda) for _ in range(0, 5)]
    if pure_td:
        np.save("../plots/imani/lambda_rc_gamma_last/returns%.2f-td.npy" % _lambda, returns)
    else:
        np.save("../plots/imani/lambda_rc_gamma_last/returns%.2f-%.2f.npy" % (_lambda, beta), returns)


def plot(results, algorithm):
    n_res = results.shape[0]
    n_it = results.shape[1]
    ret_sg_m = np.mean(results, axis=0)
    ret_sg_t = 1.96 * np.std(results, axis=0) / np.sqrt(n_res)
    plt.plot(ret_sg_m, label=algorithm)
    plt.fill_between(np.array(range(n_it)), ret_sg_m + ret_sg_t, ret_sg_m - ret_sg_t, alpha=0.5)

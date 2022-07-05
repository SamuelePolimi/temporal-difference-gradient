"""
In this experiment, we use the MDP introduced by Imani, Graves and White in 2018. They show that semi-gradient
fails to find an optimal policy.
We want to compare our boostrapped gradient with semi-gradient.
However, Imani et al. consider an off-policy objective, while we consider the on-policy objective (still relying on off-policy data).
"""
import numpy as np
import torch.optim

from herl.utils import ProgressBar, Printable, _one_hot

from algorithms import LambdaRCGamma
from settings import ImaniCounterexample
import time, json, random

print(r"""Learn the gradient of IMANI COUNTEREXAMPLE
=========================

We test GammaRC to see if it learns the correct gradient.

The output of this experiment will be found in `/plots/imani/gamma_rc_eval/`

""")

# Opening JSON file
f = open('../plots/imani/lambda_rc_gamma_learning/parameters.json')
parameters = json.load(f)
f.close()

seed = parameters["seed"]
random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)

actor_aliasing = parameters["actor_aliasing"]
critic_aliasing = parameters["critic_aliasing"]

setting = ImaniCounterexample()
analyzer = setting.analyzer

beta = parameters["beta"]
pure_td = parameters["pure_td"]


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

n_critic_features = 8 if not critic_aliasing else 6


def experiment(_lambda):

    gamma_rc = LambdaRCGamma(setting.policy,
                             critic_features,
                             actor_feature_parameters,
                             setting.mdp_task.get_descriptor(),
                             n_critic_features,
                             regularization=beta,
                             learning_rate=parameters["critic_lr"],
                             decreasing_learning_rate=parameters["decreasing_critic_lr"],
                             _lambda=_lambda,
                             pure_td=pure_td)

    setting.policy.set_parameters(setting.init_parameters)
    adam = torch.optim.Adam(setting.policy.parameters(), lr=parameters["actor_lr"])

    setting.mdp_task.max_episode_length = parameters["max_ep_length"]
    n_iterations = parameters["max_iterations"]

    j_returns = []

    start = time.time()
    pb = ProgressBar(Printable("Policy Gradient"), max_iteration=n_iterations, prefix='Gamma-RC')
    for it in range(n_iterations):
        s = setting.mdp_task.reset()
        gamma_rc.reset()
        pb.notify()
        for i in range(setting.mdp_task.max_episode_length):
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


for _lambda in parameters["lambdas"]:
    print("Testing %.2f" % _lambda)
    returns = [experiment(_lambda) for _ in range(0, parameters["n_runs"])]
    np.save("../plots/imani/lambda_rc_gamma_learning/returns%.2f.npy" % _lambda, returns)


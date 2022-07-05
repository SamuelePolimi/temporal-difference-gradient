import numpy as np
import torch.optim
import matplotlib.pyplot as plt

from herl.utils import ProgressBar, Printable, _one_hot

from algorithms import OffPAC
from settings import ImaniCounterexample
import time, json, random


print(r"""IMANI COUNTEREXAMPLE - ACE LEARNING CURVE
=========================

We provide the learning curve of OffPAC.
Setting file can be found at '../plots/imani/offpac/parameters.json',
and results '../plots/imani/offpac/'.

""")

f = open('../plots/imani/offpac/parameters.json')
parameters = json.load(f)
f.close()

seed = parameters["seed"]
random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)

actor_aliasing = parameters["actor_aliasing"]
critic_aliasing = parameters["critic_aliasing"]

setting = ImaniCounterexample()

analyzer = setting.analyzer


def critic_features(s):
    state = s
    n_codes = setting.n_states
    if critic_aliasing:
        state = setting.actor_features.codify_state(s)
        n_codes = (setting.n_states - 1)

    s_a_t = _one_hot(state.int(), n_codes)
    return s_a_t


actor_feature_parameters = None
if actor_aliasing:
    actor_feature_parameters = setting.actor_features

sa = np.array([[s, a] for s in setting.mdp.get_states()
               for a in setting.mdp.get_actions()
       ])

true_q = analyzer.get_Q(sa[:, 0:1], sa[:, 1:])
true_gamma = analyzer.get_Gamma(sa[:, 0:1], sa[:, 1:])

n_critic_features = 4 if not critic_aliasing else 3


def experiment():

    offpac = OffPAC(setting.policy,
                    setting.behavior_policy,
                    critic_features,
                    actor_feature_parameters,
                    setting.mdp_task.get_descriptor(),
                    n_critic_features,
                    trace=parameters["trace"],
                    critic_learning_rate=parameters["critic_lr"],
                    gtd_reg=parameters["gtd_regularization"])

    setting.policy.set_parameters(setting.init_parameters)
    adam = torch.optim.Adam(setting.policy.parameters(), lr=parameters["actor_lr"])  # Shall use 0.01 was 0.1

    setting.mdp_task.max_episode_length = parameters["max_ep_length"]
    n_iterations = parameters["max_iterations"]

    j_returns = []

    start = time.time()
    pb = ProgressBar(Printable("Policy Gradient"), max_iteration=n_iterations, prefix='OffPAC')
    for it in range(n_iterations):
        offpac.reset()
        s = setting.mdp_task.reset()
        pb.notify()
        for i in range(setting.mdp_task.max_episode_length):
            a = setting.behavior_policy.get_action(s)
            ret = setting.mdp_task.step(a)
            r, s_n, t = [ret[k] for k in ["reward", "next_state", "terminal"]]
            delta = offpac.update_critic(s, a, r[0], s_n, t[0])
            offpac.update_policy(adam, s, a, delta)

            s = s_n

        j_ret = (1-setting.mdp_task.gamma) * analyzer.get_return()
        j_returns.append(np.asscalar(j_ret))

    print("time: ", time.time() - start)

    return j_returns


returns = [experiment() for _ in range(0, parameters["n_runs"])]
np.save("../plots/imani/offpac/returns.npy", returns)


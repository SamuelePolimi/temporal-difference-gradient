import numpy as np
import torch.optim

from herl.utils import ProgressBar, Printable, _one_hot

from algorithms import ACE
from settings import ImaniCounterexample
import time, json, random

print(r"""IMANI COUNTEREXAMPLE - ACE LEARNING CURVE
=========================

We provide the learning curve of ACE.
Setting file can be found at '../plots/imani/lambda_rc_gamma_learning/parameters.json',
and results '../plots/imani/lambda_rc_gamma_learning/'.

""")

f = open('../plots/imani/ace/parameters.json')
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


def experiment(eta):

    offpac = ACE(setting.policy,
                 setting.behavior_policy,
                 critic_features,
                 actor_feature_parameters,
                 setting.mdp_task.get_descriptor(),
                 n_critic_features,
                 critic_trace=parameters["critic_trace"],  # 0.1
                 critic_learning_rate=parameters["critic_lr"],  # 0.1
                 gtd_reg=parameters["gtd_regularization"],  # 0.1
                 eta=eta)

    setting.policy.set_parameters(setting.init_parameters)
    adam = torch.optim.Adam(setting.policy.parameters(), lr=parameters["actor_rl"])  # 0.001

    setting.mdp_task.max_episode_length = parameters["max_ep_length"]             # 5
    n_iterations = parameters["max_iterations"]                                # 5000

    j_returns = []

    start = time.time()
    pb = ProgressBar(Printable("Policy Gradient"), max_iteration=n_iterations, prefix='ACE(%.2f)' % eta)
    for it in range(n_iterations):
        offpac.reset()
        s = setting.mdp_task.reset()
        pb.notify()
        for i in range(setting.mdp_task.max_episode_length):
            a = setting.behavior_policy.get_action(s)
            ret = setting.mdp_task.step(a)
            r, s_n, t = [ret[k] for k in ["reward", "next_state", "terminal"]]
            delta = offpac.update_critic(s, a, r[0], s_n, t[0])
            offpac.update_policy(adam, s, a, delta, t[0])

            s = s_n

        j_ret = (1-setting.mdp_task.gamma) * analyzer.get_return()
        j_returns.append(np.asscalar(j_ret))

    print("time: ", time.time() - start)

    return j_returns


for eta in parameters["etas"]:
    print("Testing %.2f" % eta)
    returns = [experiment(eta) for _ in range(0, parameters["n_runs"])]
    np.save("../plots/imani/ace/returns%.2f.npy" % eta, returns)


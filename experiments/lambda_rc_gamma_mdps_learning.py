"""
In this experiment, we use the MDP introduced by Imani, Graves and White in 2018. They show that semi-gradient
fails to find an optimal policy.
We want to compare our boostrapped gradient with semi-gradient.
However, Imani et al. consider an off-policy objective, while we consider the on-policy objective (still relying on off-policy data).
"""
import numpy as np
import torch.optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
import tikzplotlib
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
import time

from herl.classic_envs import get_imani_mdp, MDPFeatureInterface, get_random_mdp_core, MDP
from herl.actor import TabularPolicy
from herl.rl_interface import RLTask, RLDataset
from herl.rl_analysis import MDPAnalyzer
from herl.utils import ProgressBar, Printable, _one_hot
from herl.solver import RLCollector

from algorithms import LambdaRCGamma
from settings import ImaniCounterexample
import sys

print(r"""Learn the gradient of IMANI COUNTEREXAMPLE
=========================

We test GammaRC to see if it learns the correct gradient.

The output of this experiment will be found in `/plots/imani/gamma_rc_eval/`

""")

n_states = 10
n_actions = 2

critic_aliasing = True
beta = 1.0

id_start = 5
id_stop = 10
print(sys.argv)
exit()

class MyFeatures(MDPFeatureInterface):

    def __init__(self):
        MDPFeatureInterface.__init__(self, n_states, n_actions)
        self._state_matrix = np.array([[0], [1], [1]] + [[i] for i in range(2, n_states-1)])

    def codify_state(self, state):
        if hasattr(state, "shape") and len(state.shape) > 0:
            ret = [self._state_matrix[int(x)] for x in state]
        else:
            ret = self._state_matrix[int(state)]
        if type(state) is torch.Tensor:
            return torch.tensor(ret)
        return np.array(ret)

    def codify_action(self, action):
        return action


class CriticAliasing(MDPFeatureInterface):

    def __init__(self):
        MDPFeatureInterface.__init__(self, n_states, n_actions)
        self._state_matrix = np.array([[i] for i in range(3)] + [[i] for i in range(2, n_states-1)])

    def codify_state(self, state):
        if hasattr(state, "shape") and len(state.shape) > 0:
            ret = [self._state_matrix[int(x)] for x in state]
        else:
            ret = self._state_matrix[int(state)]
        if type(state) is torch.Tensor:
            return torch.tensor(ret)
        return np.array(ret)

    def codify_action(self, action):
        return action

actor_features = MyFeatures()
critic_features_base = CriticAliasing()

if critic_aliasing:
    n_codes = (n_states - 1) * n_actions
else:
    n_codes = n_states * n_actions


def critic_features(s, a):
    state = s
    if critic_aliasing:
        state = critic_features_base.codify_state(s)

    s_a_t = _one_hot(state.int() * n_actions + a.int(), n_codes)
    return s_a_t


@dataclass
class Setting:
    mdp_task: RLTask
    policy: TabularPolicy
    behavior_policy: TabularPolicy
    init_parameters: np.ndarray
    analyzer: MDPAnalyzer
    dataset: RLDataset
    id: int


actor_feature_parameters = actor_features

n_trajectories = 50000
length = 10                 # make sure the state-actions are visited enough
gamma = 0.95

settings = []
for d in range(id_start, id_stop):
    print("Create dataset %d" % d)
    core = get_random_mdp_core(n_states, n_actions)
    core._mu_0 = np.array([1.] + [0.]*(n_states-1))
    mdp = MDP(core)
    mdp_task = RLTask(mdp, mdp.get_initial_state_sampler(), gamma=gamma, max_episode_length=length)
    policy = TabularPolicy(mdp)
    analyzer = MDPAnalyzer(mdp_task, policy, actor_features)
    pi_opt, _, opt_ret = analyzer.get_opt()
    # very bad parameter initialization
    bad_parameters = np.log(np.ones((n_states, n_actions))*0.9)
    for s in range(n_states):
       bad_parameters[s, pi_opt[s]] = np.log(0.1)
    # policy.set_parameters(init_parameters)
    init_parameters = policy.get_parameters()
    behavior_policy = TabularPolicy(mdp)
    behavior_policy.set_parameters(bad_parameters)  # Let the two policy start equals and then diverge during learning
    dataset = mdp_task.get_empty_dataset(n_trajectories*length)
    rl_collector = RLCollector(dataset, mdp_task, behavior_policy, episode_length=length)
    rl_collector.collect_rollouts(n_rollouts=n_trajectories)
    settings.append(Setting(mdp_task, policy, behavior_policy, init_parameters, analyzer, dataset, d))


def experiment(_lambda, setting: Setting):

    alpha = 1/(n_states*n_actions)
    gamma_rc = LambdaRCGamma(setting.policy, critic_features, actor_feature_parameters, setting.mdp_task.get_descriptor(), n_codes,
                   regularization=beta, learning_rate=alpha, decreasing_learning_rate=False, _lambda=_lambda)

    setting.policy.set_parameters(setting.init_parameters)
    adam = torch.optim.Adam(setting.policy.parameters(), lr=0.001*alpha)     # 0.001 was good


    j_returns = []

    start = time.time()
    pb = ProgressBar(Printable("Policy Gradient"), max_iteration=n_trajectories, prefix='Gamma-RC')
    trajectories = setting.dataset.get_trajectory_list()[0]
    for i, trajectory in enumerate(trajectories):
        gamma_rc.reset()
        pb.notify()
        for s, a, r, s_n, t in zip(trajectory["state"], trajectory["action"], trajectory["reward"],
                                   trajectory["next_state"], trajectory["terminal"]):

            gamma_rc.update_critic(s, a, r[0], s_n, t[0])
            gamma_rc.update_gradient(s, a, s_n, t[0])
            gamma_rc.update_policy(adam, s)

        if i % 10 == 0:
            j_ret = (1-setting.mdp_task.gamma) * setting.analyzer.get_return()
            j_returns.append(np.asscalar(j_ret))
    end = time.time()
    print("Time %f m" % ((end-start)/60.))

    return j_returns


for _lambda in [0., 0.25, 0.5, 0.75, 1.]:
    print("Testing %.2f" % _lambda)
    for setting in settings:
        print("Testing %.2f, Setting %d" % (_lambda, setting.id))
        returns = [experiment(_lambda, setting) for _ in range(1)]
        np.save("../plots/mdps/lambda_rc_gamma_learning/returns%.2f-%.2f-%d.npy" % (_lambda, beta, setting.id), returns)


# def plot(results, algorithm):
#     n_res = results.shape[0]
#     n_it = results.shape[1]
#     ret_sg_m = np.mean(results, axis=0)
#     ret_sg_t = 1.96 * np.std(results, axis=0) / np.sqrt(n_res)
#     plt.plot(ret_sg_m, label=algorithm)
#     plt.fill_between(np.array(range(n_it)), ret_sg_m + ret_sg_t, ret_sg_m - ret_sg_t, alpha=0.5)

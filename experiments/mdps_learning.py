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
from herl.actor import TabularPolicy, SoftMaxNeuralNetworkPolicy
from herl.rl_interface import RLTask, RLDataset
from herl.rl_analysis import MDPAnalyzer
from herl.utils import ProgressBar, Printable, _one_hot
from herl.solver import RLCollector

from algorithms import LambdaRCGamma, OffPAC, ACE
from settings import ImaniCounterexample
import sys

print(r"""Learn the gradient of Random MDP
=========================

We test OffPAC, ACE(1), TDRCGamma.

The output of this experiment will be found in `/plots/mdp/learning`

""")

n_states = 30
n_actions = 2

temperature = 10.
learning_rate = 0.005
n_trajectories = 5000  #50000                   # 50000
length = 10                 # make sure the state-actions are visited enough

beta = 1.0

id = int(sys.argv[1])


class MyFeatures(MDPFeatureInterface):

    def __init__(self):
        MDPFeatureInterface.__init__(self, n_states, n_actions)

    def codify_state(self, state):
        return state

    def codify_action(self, action):
        return action


actor_features = MyFeatures()
critic_features_base = MyFeatures()

state_dim = 1
action_dim = 1

@dataclass
class Setting:
    mdp_task: RLTask
    policy: SoftMaxNeuralNetworkPolicy
    behavior_policy: SoftMaxNeuralNetworkPolicy
    init_parameters: np.ndarray
    analyzer: MDPAnalyzer
    dataset: RLDataset
    id: int


actor_feature_parameters = actor_features

gamma = 0.95

setting = None

print("Create dataset %d" % id)
core = get_random_mdp_core(n_states, n_actions, temperature=temperature)
core._mu_0 = np.array([1.] + [0.] * (n_states - 1))
mdp = MDP(core)
mdp_task = RLTask(mdp, mdp.get_initial_state_sampler(), gamma=gamma, max_episode_length=length)
policy = SoftMaxNeuralNetworkPolicy(mdp.get_descriptor(), [5], [torch.tanh, torch.tanh], n_actions)
analyzer = MDPAnalyzer(mdp_task, policy, actor_features)

head_set = ["hidden.1.weight", "hidden.1.bias"]
body_set = ["hidden.0.weight", "hidden.0.bias"]
head_parameters = list(policy.parameters())[2:]
body_parameters = list(policy.parameters())[:2]
numerical_position_head = [2, 3]
# pi_opt, _, opt_ret = analyzer.get_opt()

# # very bad parameter initialization
# bad_parameters = np.log(np.ones((n_states, n_actions))*0.9)
# for s in range(n_states):
#    bad_parameters[s, pi_opt[s]] = np.log(0.1)

init_parameters = policy.get_parameters()
behavior_policy = SoftMaxNeuralNetworkPolicy(mdp.get_descriptor(), [5], [torch.tanh, torch.tanh], n_actions)
# behavior_policy.set_parameters(bad_parameters)  # Let the two policy start equals and then diverge during learning
dataset = mdp_task.get_empty_dataset(n_trajectories*length)
rl_collector = RLCollector(dataset, mdp_task, behavior_policy, episode_length=length)
rl_collector.collect_rollouts(n_rollouts=n_trajectories)
setting = Setting(mdp_task, policy, behavior_policy, init_parameters, analyzer, dataset, id)


def experiment(_lambda, setting: Setting, algorithm_name: str):

    if algorithm_name == "rc_lambda_full" or algorithm_name == "rc_lambda_head":
        n_codes = n_states * n_actions

        def critic_features(s, a):
            state = critic_features_base.codify_state(s)
            s_a_t = _one_hot(state.int() * n_actions + a.int(), n_codes)
            return s_a_t
    else:
        n_codes = n_states

        def critic_features(s):
            state = critic_features_base.codify_state(s)
            s_a_t = _one_hot(state.int(), n_codes)
            return s_a_t


    alpha = 1/(n_states*n_actions)
    algorithm = None
    if algorithm_name == "rc_lambda_full":
        algorithm = LambdaRCGamma(setting.policy, critic_features, actor_feature_parameters, setting.mdp_task.get_descriptor(), n_codes,
                       regularization=beta, learning_rate=alpha, decreasing_learning_rate=False, _lambda=_lambda)
    elif algorithm_name == "rc_lambda_head":
        algorithm = LambdaRCGamma(setting.policy, critic_features, actor_feature_parameters, setting.mdp_task.get_descriptor(), n_codes,
                       regularization=beta, learning_rate=alpha, decreasing_learning_rate=False, _lambda=_lambda,
                                  head_parameters=head_set, body_parameters=body_set,
                                  numerical_position_head=numerical_position_head)
    elif algorithm_name == "offpac":
        algorithm = OffPAC(setting.policy, setting.behavior_policy, critic_features, actor_feature_parameters, setting.mdp_task.get_descriptor(), n_codes,
                           trace=0., critic_learning_rate=alpha, gtd_reg=0.1)
    elif algorithm_name == "ace":
        algorithm = ACE(setting.policy, setting.behavior_policy, critic_features, actor_feature_parameters, setting.mdp_task.get_descriptor(), n_codes,
                        critic_trace=0., critic_learning_rate=alpha, gtd_reg=0.1, eta=1.0)

    setting.policy.set_parameters(setting.init_parameters)
    adam = torch.optim.Adam(setting.policy.parameters(), lr=learning_rate*alpha)

    if algorithm_name == "rc_lambda_head":
        adam = torch.optim.Adam(head_parameters, lr=learning_rate*alpha)
        adam_body = torch.optim.Adam(body_parameters, lr=learning_rate*alpha)

    j_returns = []

    start = time.time()
    pb = ProgressBar(Printable("Policy Gradient"), max_iteration=n_trajectories, prefix=algorithm_name)
    trajectories = setting.dataset.get_trajectory_list()[0]

    j_ret = (1 - setting.mdp_task.gamma) * setting.analyzer.get_return()
    j_returns.append(np.asscalar(j_ret))

    for i, trajectory in enumerate(trajectories):
        algorithm.reset()
        pb.notify()
        for s, a, r, s_n, t in zip(trajectory["state"], trajectory["action"], trajectory["reward"],
                                   trajectory["next_state"], trajectory["terminal"]):

            delta = algorithm.update_critic(s, a, r[0], s_n, t[0])

            if algorithm_name == "rc_lambda_full" or algorithm_name == "rc_lambda_head":
                algorithm.update_gradient(s, a, s_n, t[0])
            if algorithm_name == "offpac":
                algorithm.update_policy(adam, s, a, delta)
            elif algorithm_name == "ace":
                algorithm.update_policy(adam, s, a, delta, t[0])
            else:
                algorithm.update_policy(adam, s=s)

            if algorithm_name == "rc_lambda_head":
                algorithm.update_body_policy(adam_body, s)


        if i % 10 == 0:
            j_ret = (1-setting.mdp_task.gamma) * setting.analyzer.get_return()
            j_returns.append(np.asscalar(j_ret))
    end = time.time()
    print("Time %f m" % ((end-start)/60.))

    return j_returns


for _lambda in [0., 0.25, 0.5, 0.75, 1.]:
    print("Testing RC-Lambda %.2f, Setting %d" % (_lambda, setting.id))
    returns = [experiment(_lambda, setting, "rc_lambda_full") for _ in range(1)]
    np.save("plots/mdps/learning/returns-rcl-full-%.2f-%d.npy" % (_lambda, setting.id), returns)
    returns = [experiment(_lambda, setting, "rc_lambda_head") for _ in range(1)]
    np.save("plots/mdps/learning/returns-rcl-head-%.2f-%d.npy" % (_lambda, setting.id), returns)
returns = [experiment(_lambda, setting, "offpac") for _ in range(1)]
np.save("plots/mdps/learning/returns-offpac-%d.npy" % setting.id, returns)
returns = [experiment(_lambda, setting, "ace") for _ in range(1)]
np.save("plots/mdps/learning/returns-ace-%d.npy" % setting.id, returns)
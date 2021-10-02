import numpy as np

from herl.classic_envs import get_imani_mdp
from herl.actor import TabularPolicy
from herl.rl_interface import RLTask
from herl.rl_analysis import MDPAnalyzer


class ImaniCounterexample:

    def __init__(self):
        # MPD from Imani et al. 2018
        self.mdp, self.actor_features = get_imani_mdp()

        self.n_states = 4
        self.n_actions = 2

        # Tabular SoftMax policy
        self.policy = TabularPolicy(self.mdp)
        self.policy.set_parameters(np.log([[0.9, 0.1],
                                      [0.9, 0.1],
                                      [0.9, 0.1],
                                      [0.9, 0.1]]
                                     ))
        self.behavior_policy = TabularPolicy(self.mdp)

        # Define off-policy behavior (as in the original paper)
        self.beta = np.array([[0.25, 0.75],
                              [0.25, 0.75],
                              [0.25, 0.75],
                              [0.25, 0.75]]
                             )
        self.behavior_policy.set_parameters(np.log(self.beta))

        # Define the state distribution (similarly to the original paper,
        # but with a positive probability on the terminal state)
        self.mu = np.array([0.5 / 2, 0.125 / 2, 0.375 / 2, 0.5])

        self.init_parameters = self.policy.get_parameters()

        self.mdp_task = RLTask(self.mdp, self.mdp.get_initial_state_sampler(), max_episode_length=500, gamma=0.9)
        self.analyzer = MDPAnalyzer(self.mdp_task, self.policy, self.actor_features)
import numpy as np
import torch

from typing import Callable

from herl.rl_interface import RLTask, RLAgent, PolicyGradient
from herl.classic_envs import MDP
from herl.rl_analysis import MDPAnalyzer

from algorithms.closed.lstd_closed import ClosedLSTD


class ClosedSemiGradient(PolicyGradient, ClosedLSTD):

    """
    This gradient computation assumes perfect features for the Critic (they are always possible to find in a Finite MDP)
    while might use imperfect state representation for the policy. (It seems that Imani et al. did a similar experiment)
    """

    def __init__(self, task: RLTask, policy: RLAgent, critic_features: Callable, actor_features, mu: np.ndarray, beta: np.ndarray):
        ClosedLSTD.__init__(self, task, policy, critic_features, actor_features, mu, beta)
        PolicyGradient.__init__(self, "ClosedSemiGradient", policy)

    def get_q_matrix(self):
        """
        Given an off-policy distribution mu and a behavioral policy, compute the LSTD solution of the Q-matrix
        :param mu:  off-policy state distribution
        :param beta: behavioral policy
        :return:
        """

        # compute the parameter vector \omega_TD. In semi-gradient, this is NOT differentiable.
        omega = self.get_omega_q().detach()

        # build a matrix of the Q-values
        return torch.stack([
            torch.stack([
                    torch.inner(self._critic_features(torch.tensor(s), torch.tensor(a)), omega)
                for a in self._mdp.get_actions()
            ])
            for s in self._mdp.get_states()
        ])

    def get_v_vector(self):
        """
        Compute the value function.
        :param mu:  off-policy state distribution
        :param beta: behavioral policy
        :return:
        """
        # Get Q-matrix
        Q = self.get_q_matrix()
        # Build policy matrix (optimization policy)
        pi = torch.stack([
                torch.stack([self.get_scalar_pi(s, a)
                for a in self._mdp.get_actions()
            ])
            for s in self._mdp.get_states()
        ])
        # Compute Q-function
        return torch.sum(Q*pi, dim=1)

    def surrogate(self):
        v = self.get_v_vector()
        mu_t = torch.tensor(self.mu, dtype=torch.float64)
        return torch.inner(mu_t, v)

    def get_gradient(self):
        j = -self.surrogate()
        j.backward()
        ret = self._policy.get_gradient()
        self._policy.zero_grad()
        return ret

    def update_policy(self, optimizer):
        optimizer.zero_grad()
        j = -self.surrogate()
        j.backward()
        optimizer.step()
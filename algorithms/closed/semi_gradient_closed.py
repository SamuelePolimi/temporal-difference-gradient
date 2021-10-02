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
        j = -self.get_return()
        j.backward()
        optimizer.step()
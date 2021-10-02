import numpy as np
import torch

from typing import Callable
from herl.rl_interface import RLTask, RLAgent, PolicyGradient
from herl.classic_envs import MDPFeatureInterface

from algorithms.closed.lstd_closed import ClosedLSTD


class ClosedLSTDGamma(PolicyGradient, ClosedLSTD):

    def __init__(self, task: RLTask, policy: RLAgent,
                 critic_features: Callable,
                 actor_features: MDPFeatureInterface,
                 mu: np.ndarray, beta: np.ndarray):
        """LSTDGamma as described in Algorithm 1; but with closed form computation of expectations.
        Works only with Finite MDPs.

        :param task: A Finite MDP
        :param policy: optimization policy
        :param critic_features: A function that returns a feature vector given state-action pairs
        :param mu: State distribution (in form on a n-dimensional vector)
        :param beta: Matrix n_states x n_actions describing conditional probability
        of an action given a state as behavior policy."""
        ClosedLSTD.__init__(self, task, policy, critic_features, actor_features, mu, beta)
        PolicyGradient.__init__(self, r"ClosedLSTD$\Gamma$", policy)
        self.beta = beta

    def get_return(self):
        # Compute value function using the off-policy distribution
        v = self.get_v_vector()
        # get the starting-state distribution
        mu_0 = torch.tensor(self._mdp_analyzer._mu_0, dtype=torch.float64)
        # compute the expected return.
        return (1 - self._task.gamma) * torch.inner(mu_0, v)

    def surrogate_loss(self):
        """
        Computation Based on Lemma 2 and Eq 10
        :param mu:
        :param beta:
        :return:
        """
        A, b = self.get_A_b()
        A_inv = torch.linalg.inv(A).detach()
        mu_0 = torch.tensor(self._mdp_analyzer._mu_0, dtype=torch.float64)
        omega = torch.linalg.inv(A) @ b
        d_omega = - (A_inv @ A @ A_inv @ b).view(-1)
        surrogate = 0.
        for s in self._mdp.get_states():
            for a in self._mdp.get_actions():
                phi = self._critic_features(torch.tensor([s]), torch.tensor([a]))
                surrogate += torch.inner(phi, omega.detach())\
                             * self.get_scalar_pi(s, a) * mu_0[s] + torch.inner(phi, d_omega)\
                             * self.get_scalar_pi(s, a).detach() * mu_0[s]
        return (1 - self._task.gamma) * surrogate

    def get_gradient(self):
        j = self.surrogate_loss()
        j.backward()
        ret = self._policy.get_gradient()
        self._policy.zero_grad()
        return ret

    def update_policy(self, optimizer):
        optimizer.zero_grad()
        j = -self.surrogate_loss()
        j.backward()
        optimizer.step()
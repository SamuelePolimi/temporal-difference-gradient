import numpy as np
import torch

from typing import Callable
from herl.rl_interface import RLTask, RLAgent, Critic, PolicyGradient
from herl.classic_envs import MDP, MDPFeatureInterface
from herl.rl_analysis import MDPAnalyzer
from herl.utils import _one_hot


class ClosedLSTDGamma(PolicyGradient, Critic):

    def __init__(self, task: RLTask, policy: RLAgent,
                 critic_features: Callable,
                 actor_features: MDPFeatureInterface,
                 mu: np.ndarray, beta: np.ndarray):
        """LSTDGamma as described in Algorithm 1; but with closed form computation of expectations.
        Works only with Finite MDPs.

        :param policy: A parametric policy
        :param task: A Finite MDP
        :param policy: usual policy"""
        super().__init__(r"ClosedLSTD$\Gamma$", policy)
        self._task = task
        if type(self._task.environment) is not MDP:
            raise("task.environemnt should be an MDP!")
        self._mdp = self._task.environment  # type: MDP
        self._policy = policy

        self._actor_features = actor_features
        self._critic_features = critic_features

        self._mdp_analyzer = MDPAnalyzer(task, policy)
        self._n_states = len(self._mdp.get_states())
        self._n_actions = len(self._mdp.get_actions())
        self.name = "Bootstrapped Gradient"

        self.mu = mu
        self.beta = beta

    def get_omega_q(self, mu, beta):

        gamma = self._task.gamma

        # Usual transition matrix and reward vector
        P = torch.tensor(self._mdp.get_transition_matrix())
        r = torch.tensor(self._mdp.get_reward_matrix())

        A = 0.
        for s in self._mdp.get_states():
            for a in self._mdp.get_actions():

                # According to the off-policy distributiun, this state, action pairs has probability p_s_a
                p_s_a = mu[s] * beta[s, a]

                # Compute \phi
                phi = self._critic_features(torch.tensor(s), torch.tensor(a))

                # Compute \phi'
                phi_next = 0.
                for s_n in self._mdp.get_states():
                    for a_n in self._mdp.get_actions():
                        p_s_n = P[a, s, s_n]
                        # Use coding from the MDP
                        s_n_t, a_n_t = torch.tensor(s_n), torch.tensor(a_n)
                        p_a_n = self._policy.get_prob(
                            self._actor_features.codify_state(s_n_t),
                            self._actor_features.codify_action(a_n_t),
                                                      differentiable=True).squeeze()

                        phi_next += p_s_n * p_a_n * self._critic_features(s_n_t, a_n_t)

                A += p_s_a * torch.outer(phi, phi - gamma * phi_next)

        # Compute b
        b = 0.
        for s in self._mdp.get_states():
            for a in self._mdp.get_actions():
                p_s_a = mu[s] * beta[s, a]
                rew = r[a, s]
                phi = self._critic_features(torch.tensor(s), torch.tensor(a))

                b += p_s_a * rew * phi

        # \omega_{TD} = A^{-1}b
        return torch.linalg.solve(A, b)

    def get_q_matrix(self, mu, beta):
        """
        Given an off-policy distribution mu and a behavioral policy, compute the LSTD solution of the Q-matrix
        :param mu:  off-policy state distribution
        :param beta: behavioral policy
        :return:
        """

        # compute the parameter vector \omega_TD
        omega = self.get_omega_q(mu, beta)

        # build a matrix of the Q-values
        return torch.stack([
            torch.stack([
                    torch.inner(self._critic_features(torch.tensor(s), torch.tensor(a)), omega)
                for a in self._mdp.get_actions()
            ])
            for s in self._mdp.get_states()
        ])

    def get_v_vector(self, mu, beta):
        """
        Compute the value function
        :param mu:  off-policy state distribution
        :param beta: behavioral policy
        :return:
        """
        # Get Q-matrix
        Q = self.get_q_matrix(mu, beta)
        # Build policy matrix (optimization policy)
        pi = torch.stack([
                torch.stack([self._policy.get_prob(
                        torch.tensor(self._actor_features.codify_state(s)),
                        torch.tensor(self._actor_features.codify_action(a)), differentiable=True)
                for a in self._mdp.get_actions()
            ])
            for s in self._mdp.get_states()
        ])
        # Compute Q-function
        return torch.sum(Q*pi, dim=1)

    def get_return(self):
        # Compute value function using the off-policy distribution
        v = self.get_v_vector(self.mu, self.beta)
        # get the starting-state distribution
        mu_0 = torch.tensor(self._mdp_analyzer._mu_0, dtype=torch.float64)
        # compute the expected return.
        return torch.inner(mu_0, v)

    def get_gradient(self):
        j = -self.get_return()
        j.backward()
        ret = self._policy.get_gradient()
        self._policy.zero_grad()
        return ret

    def update_policy(self, optimizer):
        optimizer.zero_grad()
        j = -self.get_return()
        j.backward()
        optimizer.step()
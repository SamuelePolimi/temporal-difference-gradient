import numpy as np
import torch

from typing import Callable
from herl.rl_interface import RLTask, RLAgent, Critic, PolicyGradient
from herl.classic_envs import MDP, MDPFeatureInterface
from herl.rl_analysis import MDPAnalyzer
from herl.utils import _one_hot


class ClosedLSTD(Critic):

    def __init__(self, task: RLTask, policy: RLAgent,
                 critic_features: Callable,
                 actor_features: MDPFeatureInterface,
                 mu: np.ndarray, beta: np.ndarray):
        """Closed form solution of LSTD.

        :param task: A Finite MDP
        :param policy: optimization policy
        :param critic_features: A function that returns a feature vector given state-action pairs
        :param mu: State distribution (in form on a n-dimensional vector)
        :param beta: Matrix n_states x n_actions describing conditional probability
        of an action given a state as behavior policy."""
        Critic.__init__(self, "LSTD")
        self._task = task
        if type(self._task.environment) is not MDP:
            raise("task.environemnt should be an MDP!")
        self._mdp = self._task.environment  # type: MDP
        self._policy = policy

        self._actor_features = actor_features
        self._private_critic_features = critic_features

        self._mdp_analyzer = MDPAnalyzer(task, policy, actor_features)
        self._n_states = len(self._mdp.get_states())
        self._n_actions = len(self._mdp.get_actions())
        self.name = "Bootstrapped Gradient"

        self.mu = mu
        self.beta = beta

    def _critic_features(self, s, a):
        """
        Returns the features for a state action pairs
        :param s: state
        :param a:action
        :return:
        """
        return self._private_critic_features(s, a).view(-1)

    def get_omega_q(self) -> torch.Tensor:
        """
        Compute omega_TD as in inline math after Assumption 2
        :param mu:
        :param beta:
        :return:
        """
        A, b = self.get_A_b()
        return torch.linalg.solve(A, b)

    def get_A_b(self):
        """
        Get matrix A and vector b as in inline math after Assumption 2
        :return:
        """

        mu = self.mu
        beta = self.beta

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
                        s_n_t, a_n_t = torch.tensor([[s_n]]), torch.tensor([[a_n]])
                        p_a_n = self.get_scalar_pi(s_n, a_n)

                        phi_next += p_s_n * p_a_n * self._critic_features(s_n_t.squeeze(), a_n_t.squeeze())

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
        return A, b

    def get_q_matrix(self):
        """
        Given an off-policy distribution mu and a behavioral policy, compute the LSTD solution of the Q-matrix
        :param mu:  off-policy state distribution
        :param beta: behavioral policy
        :return:
        """

        # compute the parameter vector \omega_TD
        omega = self.get_omega_q()

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
        Compute the value function
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

    def get_scalar_pi(self, s, a):
        return self._policy.get_prob(
                        self._actor_features.codify_state(torch.tensor([[s]])),
                        self._actor_features.codify_action(torch.tensor([[a]])), differentiable=True).squeeze()
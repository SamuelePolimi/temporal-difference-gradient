import numpy as np
import torch

from typing import Callable

from herl.rl_interface import RLTask, RLAgent, Critic, PolicyGradient
from herl.classic_envs import MDP
from herl.rl_analysis import MDPAnalyzer
from herl.utils import _one_hot


class ClosedSemiGradient(PolicyGradient, Critic):

    """
    This gradient computation assumes perfect features for the Critic (they are always possible to find in a Finite MDP)
    while might use imperfect state representation for the policy. (It seems that Imani et al. did a similar experiment)
    """

    def __init__(self, task: RLTask, policy: RLAgent, critic_features: Callable, actor_features, mu: np.ndarray, beta: np.ndarray):
        super().__init__("ClosedSemiGradient", policy)
        self._task = task
        if type(self._task.environment) is not MDP:
            raise("task.environemnt should be an MDP!")
        self._mdp = self._task.environment  # type: MDP
        self._policy = policy

        self._mdp_analyzer = MDPAnalyzer(task, policy)
        self._n_states = len(self._mdp.get_states())
        self._n_actions = len(self._mdp.get_actions())

        self._actor_features = actor_features
        self._critic_features = critic_features

        self.mu = mu
        self.beta = beta
        self.name = "Semi-Gradient"

    def get_omega_q(self, mu, beta):

        gamma = self._task.gamma
        P = torch.tensor(self._mdp.get_transition_matrix())
        r = torch.tensor(self._mdp.get_reward_matrix())

        A = 0.
        for s in self._mdp.get_states():
            for a in self._mdp.get_actions():
                phi = self._critic_features(torch.tensor(s), torch.tensor(a))
                p_s_a = mu[s] * beta[s, a]

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

        b = 0.
        for s in self._mdp.get_states():
            for a in self._mdp.get_actions():
                p_s_a = mu[s] * beta[s, a]
                rew = r[a, s]
                phi = self._critic_features(s, a)

                b += p_s_a * rew * phi

        return torch.linalg.solve(A, b)

    def get_A_b(self, mu, beta):

        gamma = self._task.gamma
        P = torch.tensor(self._mdp.get_transition_matrix())
        r = torch.tensor(self._mdp.get_reward_matrix())

        A = 0.
        for s in self._mdp.get_states():
            for a in self._mdp.get_actions():
                phi = self._critic_features(s, a)
                p_s_a = mu[s] * beta[s, a]

                phi_next = 0.
                for s_n in self._mdp.get_states():
                    for a_n in self._mdp.get_actions():
                        p_s_n = P[a, s, s_n]
                        # TODO: check
                        s_n_t = torch.tensor(self._actor_features.codify_state(s_n))
                        a_n_t = torch.tensor(self._actor_features.codify_action(a_n))
                        p_a_n = self._policy.get_prob(s_n_t, a_n_t, differentiable=True).squeeze()

                        phi_next += p_s_n * p_a_n * self._critic_features(s_n, a_n)

                A += p_s_a * torch.outer(phi, phi - gamma * phi_next)

        b = 0.
        for s in self._mdp.get_states():
            for a in self._mdp.get_actions():
                p_s_a = mu[s] * beta[s, a]
                rew = r[a, s]
                phi = self._critic_features(s, a)

                b += p_s_a * rew * phi

        return A, b

    def get_q_matrix(self, mu, beta):
        omega = self.get_omega_q(mu, beta)
        return torch.stack([
            torch.stack([
                    torch.inner(self._critic_features(torch.tensor(s), torch.tensor(a)), omega)
                for a in self._mdp.get_actions()
            ])
            for s in self._mdp.get_states()
        ])

    def get_v_vector(self, mu, beta):
        Q = self.get_q_matrix(mu, beta).detach()
        pi = torch.stack([
                torch.stack([self._policy.get_prob(
                            self._actor_features.codify_state(torch.tensor(s)),
                            self._actor_features.codify_action(torch.tensor(a)), differentiable=True)
                for a in self._mdp.get_actions()
            ])
            for s in self._mdp.get_states()
        ])
        return torch.sum(Q*pi, dim=1)

    def get_return(self):
        v = self.get_v_vector(self.mu, self.beta)
        mu_t = torch.tensor(self.mu, dtype=torch.float64)
        return torch.inner(mu_t, v)

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
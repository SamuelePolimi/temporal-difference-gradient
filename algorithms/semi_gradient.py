import numpy as np
import torch

from typing import Union
from herl.rl_interface import PolicyGradient, Critic, RLAgent, RLParametricModel, RLDataset, RLTaskDescriptor,\
    DeterministicState, StochasticState, StateDistribution


class SemiGradient(PolicyGradient, Critic):

    def __init__(self, policy: Union[RLAgent, RLParametricModel], critic_features, actor_features, dataset: RLDataset,
                 task_descriptor: RLTaskDescriptor, n_critic_approximation=1,
                 reparametrization=False, regularization=1E-12):
        """
        Semi-gradient
        :param policy: A parametric policy
        :param critic_features: The features of the critic used by LSTD
        :param task_descriptor: Descriptor of the task to solve
        :param n_starting_state: number of starting states to be sampled
        :param n_critic_approximation: number of actions to be sampled
        :param reparametrization: specify whether we use the reparametrization trick to compute the gradient
        """
        super().__init__(r"SG", policy)
        self._critic_features = critic_features
        self._actor_features = actor_features
        self._dataset = dataset
        self._gamma = task_descriptor.gamma

        self._n_critic_approximation = n_critic_approximation
        self._reparametrization = reparametrization
        self._regularization = regularization

    def get_omega_batch(self, batch):
        """
        Compute omega_TD (Defined in text right after Assumption 2)
        :param batch:
        :return:
        """
        s, a, r, s_n, t = [torch.tensor(b, dtype=torch.float64) for b in batch]

        n = s.shape[0]
        phi = torch.tensor(self._critic_features(s.int(), a.int()), dtype=torch.float64)
        if self._reparametrization:
            a_n = self.policy(s_n, differentiable=True)
            phi_next = self._critic_features(s_n, a_n)
        else:
            a_n = self.get_on_policy_action(s_n)
            phi_next = torch.tensor(self._critic_features(s_n, a_n), dtype=torch.float64)
        A = torch.transpose(phi, 0, 1) @ (phi - self._gamma * phi_next) / n     # Alternative way to compute A_\pi

        return torch.linalg.inv(A + self._regularization*torch.eye(A.shape[0])) @ torch.mean(phi * r, dim=0)

    def get_on_policy_action(self, s):
        state = s
        if self._actor_features is not None:
            state = self._actor_features.codify_state(s)
        return self.policy(state, differentiable=True)

    def get_log_prob(self, s, a):
        state, action = s, a
        if self._actor_features is not None:
            state, action = self._actor_features.codify_state(s), self._actor_features.codify_action(a)
        return torch.log(self.policy.get_prob(state, action, differentiable=True))

    def get_Q(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:

        data_dict = self._dataset.get_full()
        batch = [data_dict[k] for k in ["state", "action", "reward", "next_state", "terminal"]]
        omega = self.get_omega_batch(batch)

        return self._critic_features(state, action) @ omega.detach().numpy()

    def get_V(self, state: np.ndarray) -> np.ndarray:

        data_dict = self._dataset.get_full()
        batch = [data_dict[k] for k in ["state", "action", "reward", "next_state", "terminal"]]
        omega = self.get_omega_batch(batch)
        return self._critic_features(torch.tensor(state), torch.tensor(self.policy.get_actions(state)))\
               @ omega.detach().numpy()

    def get_surrogate_loss(self):

        data_dict = self._dataset.get_full()
        batch = [torch.tensor(data_dict[k]) for k in ["state", "action", "reward", "next_state", "terminal"]]
        s, _, _, _, _ = batch

        a_on = self.get_on_policy_action(s)
        phi = self._critic_features(s, a_on)

        omega = self.get_omega_batch(batch)

        pi_log = self.get_log_prob(s, a_on)

        if self._reparametrization:
            first_term = torch.inner(torch.mean(phi, dim=0), omega)
        else:
            first_term = torch.inner(torch.mean(phi.detach() * pi_log, dim=0), omega)

        return first_term

    def get_gradient(self):
        """
        Estimate the gradient.
        This code take full advantage of automatic differentiation, and on the fact that G_TD = \nabla_theta \omega_TD,
        as in Lemma 2. \omega_TD is fully differentiable with pytorch.
        :return:
        """
        loss = self.get_surrogate_loss()
        loss.backward()
        ret = self.policy.get_gradient()
        self.policy.zero_grad()
        return ret

    def update_policy(self, optimizer):
        optimizer.zero_grad()
        j = -self.get_surrogate_loss()
        j.backward()
        optimizer.step()
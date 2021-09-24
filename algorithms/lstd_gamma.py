import numpy as np
import torch

from typing import Union
from herl.rl_interface import PolicyGradient, Critic, RLAgent, RLParametricModel, RLDataset, RLTaskDescriptor,\
    DeterministicState, StochasticState, StateDistribution


class LSTDGamma(PolicyGradient, Critic):

    def __init__(self, policy: Union[RLAgent, RLParametricModel], critic_features, dataset: RLDataset,
                 task_descriptor: RLTaskDescriptor, n_starting_state=10, n_critic_approximation=1,
                 reparametrization=False):
        """
        LSTDGamma as described in Algorithm 1.
        :param policy: A parametric policy
        :param critic_features: The features of the critic used by LSTD
        :param task_descriptor: Descriptor of the task to solve
        :param n_starting_state: number of starting states to be sampled
        :param n_critic_approximation: number of actions to be sampled
        :param reparametrization: specify whether we use the reparametrization trick to compute the gradient
        """
        super().__init__(r"LSTD$\Gamma$name", policy)
        self._critic_features = critic_features
        self._dataset = dataset
        self._gamma = task_descriptor.gamma

        if task_descriptor.initial_state_distribution.is_deterministic():
            self._s_0 = np.array([task_descriptor.initial_state_distribution.sample()])
        else:
            self._s_0 = np.array([task_descriptor.initial_state_distribution.sample() for i in range(n_starting_state)])

        self._n_critic_approximation = n_critic_approximation
        self._reparametrization = reparametrization

    def get_omega_batch(self, batch):
        """
        Compute omega_TD (Defined in text right after Assumption 2)
        :param batch:
        :return:
        """
        s, a, r, s_n, t = [torch.tensor(b, dtype=torch.float64) for b in batch]

        n = s.shape[0]
        phi = self._critic_features(s, a, differentiable=True)
        if self._reparametrization:
            a_n = self.policy(s_n, differentiable=True)
            phi_next = self._critic_features(s_n, a_n, differentiable=True)
        else:
            a_n = self.policy(s_n, differentiable=True).detach()
            phi_next = self._critic_features(s_n, a_n, differentiable=True)\
                       * torch.unsqueeze(self.policy.get_log_prob(s_n, a_n, differentiable=True), 1)
        A = torch.transpose(phi, 0, 1) @ (phi - self._gamma * phi_next) / n     # Alternative way to compute A_\pi

        return torch.linalg.inv(A) @ torch.mean(phi * r, dim=0)

    def get_return(self):
        s_0 = torch.tensor(self._s_0, dtype=torch.float64)
        a_0 = self.policy(s_0, differentiable=True).detach()
        phi_0 = self._critic_features(s_0, a_0, differentiable=True) * \
                torch.unsqueeze(self.policy.get_log_prob(s_0, a_0, differentiable=True), dim=1)

        data_dict = self._dataset.get_full()
        batch = [data_dict[k] for k in ["state", "action", "reward", "next_state", "terminal"]]
        omega = self.get_omega_batch(batch)
        return torch.inner(torch.mean(phi_0, dim=0), omega)

    def get_gradient(self):
        """
        Estimate the gradient.
        This code take full advantage of automatic differentiation, and on the fact that G_TD = \nabla_theta \omega_TD,
        as in Lemma 2. \omega_TD is fully differentiable with pytorch.
        :return:
        """
        loss = -self.get_return()
        loss.backward()
        ret = self.policy.get_gradient()
        self.policy.zero_grad()
        return ret

    def update_policy(self, optimizer):
        optimizer.zero_grad()
        j = -self.get_return()
        j.backward()
        optimizer.step()
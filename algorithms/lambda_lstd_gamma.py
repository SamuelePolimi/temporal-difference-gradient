import numpy as np
import torch

from typing import Union
from herl.rl_interface import PolicyGradient, Critic, RLAgent, RLParametricModel, RLDataset, RLTaskDescriptor,\
    DeterministicState, StochasticState, StateDistribution

from algorithms.lstd_gamma import LSTDGamma


class LambdaLSTDGamma(LSTDGamma):

    def __init__(self, policy: Union[RLAgent, RLParametricModel], critic_features, actor_features, dataset: RLDataset,
                 task_descriptor: RLTaskDescriptor, n_starting_state=10, n_critic_approximation=1,
                 reparametrization=False, regularization=1E-12, _lambda=0.):
        """
        LSTDGamma as described in Algorithm 1, but with eligibility traces described in Equation 13.
        :param policy: A parametric policy
        :param critic_features: The features of the critic used by LSTD
        :param task_descriptor: Descriptor of the task to solve
        :param n_starting_state: number of starting states to be sampled
        :param n_critic_approximation: number of actions to be sampled
        :param reparametrization: specify whether we use the reparametrization trick to compute the gradient
        """
        super().__init__(policy, critic_features, actor_features, dataset,
                 task_descriptor, n_starting_state, n_critic_approximation,
                 reparametrization, regularization)
        self.name = r"LSTD\Gamma \lambda={:.2f}".format(_lambda)
        self._lambda = _lambda

    def get_surrogate_loss(self):

        data_dict = self._dataset.get_full()
        batch = [data_dict[k] for k in ["state", "action", "reward", "next_state", "terminal"]]

        A, b, A_diff = self.get_A_b(batch)
        A_inv = torch.linalg.inv(A + self._regularization * torch.eye(A.shape[0])).detach()
        g = - A_inv @ A_diff @ A_inv @ b

        omega = A_inv @ b

        surrogates = []
        for trajectory in self._dataset.get_trajectory_list()[0]:
            surrogate = 0.
            mu = 1.
            for s in trajectory["state"]:
                s = torch.tensor([s])
                a = self.get_on_policy_action(s)  # self.policy(s_0, differentiable=True)
                phi = self._critic_features(s, a)

                pi_log = self.get_log_prob(s, a)

                if self._reparametrization:
                    first_term = torch.inner(torch.mean(phi, dim=0), omega)
                else:
                    first_term = torch.inner(torch.mean(phi.detach() * pi_log, dim=0), omega)

                surrogate += mu * (first_term + (1 - self._lambda) * torch.inner(torch.mean(phi.detach(), dim=0), g))
                mu *= self._lambda * self._gamma
            surrogates.append(surrogate)

        return (1-self._gamma) * sum(surrogates)/len(surrogates)

import numpy as np
import torch

from typing import Union
from herl.rl_interface import PolicyGradient, Critic, RLAgent, RLParametricModel, RLDataset, RLTaskDescriptor,\
    DeterministicState, StochasticState, StateDistribution


class OffPAC(PolicyGradient, Critic):

    def __init__(self, policy: Union[RLAgent, RLParametricModel], behavior: RLAgent, critic_features, actor_features,
                 task_descriptor: RLTaskDescriptor, n_critic_features,
                 trace=0., critic_learning_rate=1., gtd_reg=0.):
        """
        LSTDGamma as described in Algorithm 1.
        :param policy: A parametric policy
        :param critic_features: The features of the critic used by LSTD
        :param task_descriptor: Descriptor of the task to solve
        :param n_starting_state: number of starting states to be sampled
        :param n_critic_approximation: number of actions to be sampled
        :param reparametrization: specify whether we use the reparametrization trick to compute the gradient
        """
        super().__init__(r"OffPAC$", policy)
        self._behavior = behavior
        self._critic_features = critic_features
        self._actor_features = actor_features
        self._gamma = task_descriptor.gamma
        self._trace = trace
        self._alpha_v = critic_learning_rate
        self._alpha_w = gtd_reg

        self.n_p = len(self.policy.get_parameters())
        self.n_f = n_critic_features

        self._e_v = np.zeros(self.n_f)
        self._e_u = np.zeros(self.n_p)

        self._w = np.zeros(self.n_f)

        self._v = np.zeros(self.n_f)
        self._u = np.zeros(self.n_p)

    def phi(self, s: np.ndarray):
        """
        Compute critic features per given state action pairs
        :param s: 1D vector representing a state
        :return:
        """
        return self._critic_features(torch.tensor(s)).detach().numpy()

    def get_on_policy_action(self, s: np.ndarray):
        """
        Sample an action from the policy taking in consideration actor aliasing
        :param s: 1D vector representing a state
        """

        state = s
        if self._actor_features is not None:
            state = self._actor_features.codify_state(s).ravel()
        return self.policy.get_action(state)

    def get_nabla_log(self, s: np.ndarray, a: np.ndarray):
        """
        Compute \nabla_\theta \log \pi(a|s) for a single state action pair
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        """
        state = s
        if self._actor_features is not None:
            state = self._actor_features.codify_state(s).ravel()
        try:
            prob = self.policy.get_prob(torch.tensor([state]), torch.tensor([a]), differentiable=True)
        except:
            prob = self.policy.get_prob(torch.tensor([state]), torch.tensor([a]))
        log_prob = torch.log(prob)
        log_prob = log_prob.squeeze()
        log_prob.backward()
        ret = self.policy.get_gradient()
        self.policy.zero_grad()
        return ret

    def get_rho(self, s: np.ndarray, a: np.ndarray):
        """
        Compute \nabla_\theta \log \pi(a|s) for a single state action pair
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        """
        state = s
        if self._actor_features is not None:
            state = self._actor_features.codify_state(s).ravel()
        try:
            prob_a = self.policy.get_prob(torch.tensor([state]), torch.tensor([a]), differentiable=True)
            prob_b = self._behavior.get_prob(torch.tensor([state]), torch.tensor([a]), differentiable=True)
        except:
            prob_a = self.policy.get_prob(torch.tensor([state]), torch.tensor([a]))
            prob_b = self._behavior.get_prob(torch.tensor([state]), torch.tensor([a]))
        return np.asscalar((prob_a/prob_b).detach().numpy())

    def update_critic(self,  s: np.ndarray, a: np.ndarray, r: np.ndarray, s_n: np.ndarray, t: bool):
        """
        Update the critic (omega parameters) for a s, a, r, s_n tuple
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        :param r: scalar reward
        :param s_n: 1D vector representing the next_state
        :param t: True if the transition is a terminal one
        :return:
        """

        gamma = self._gamma * (1-t)

        x = self.phi(s).ravel()
        x_n = self.phi(s_n).ravel()

        delta = r + gamma * np.dot(self._v, x_n) - np.dot(self._v, x)
        rho = self.get_rho(s, a)

        self._e_v = rho * x + self._gamma * self._trace * self._e_v
        self._v += self._alpha_v * (delta * self._e_v - gamma * (1 - self._trace) * (np.dot(self._w, self._e_v) * x))
        self._w += self._alpha_w * (delta * self._e_v - np.dot(self._w, x) * x)

        return delta

    def get_surrogate_loss(self, s, a, delta):
        """
        TD error
        :param delta:
        :return:
        """
        parameters = self.policy.get_torch_parameters()

        immediate_gradient = self.get_nabla_log(s, a)
        rho = self.get_rho(s, a)

        self._e_u = rho * (immediate_gradient + self._gamma * self._trace * self._e_u)
        gradient = self._e_u * delta
        # Equation 13 of the paper
        return torch.inner(torch.tensor(gradient), parameters)

    def update_policy(self, optimizer, s, a, delta):
        optimizer.zero_grad()
        j = -self.get_surrogate_loss(s, a, delta)
        j.backward()
        optimizer.step()
        self.policy.zero_grad()

    def reset(self):
        self._e_u *= 0.
        self._e_v *= 0.
        self._w *= 0.
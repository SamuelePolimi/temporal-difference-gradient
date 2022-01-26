import numpy as np
import torch

from typing import Union
from herl.rl_interface import PolicyGradient, Critic, RLAgent, RLParametricModel, RLDataset, RLTaskDescriptor,\
    DeterministicState, StochasticState, StateDistribution


class LambdaRCGamma(PolicyGradient, Critic):

    def __init__(self, policy: Union[RLAgent, RLParametricModel], critic_features, actor_features,
                 task_descriptor: RLTaskDescriptor, n_critic_features, n_starting_state=10, n_critic_approximation=1,
                 reparametrization=False, regularization=1., _lambda=0., learning_rate=1.,
                 decreasing_learning_rate=False, pure_td=False, less_computation=False, delayed=True,
                 head_parameters=None, body_parameters=None, numerical_position_head=None):
        """
        LSTDGamma as described in Algorithm 1.
        :param policy: A parametric policy
        :param critic_features: The features of the critic used by LSTD
        :param task_descriptor: Descriptor of the task to solve
        :param n_starting_state: number of starting states to be sampled
        :param n_critic_approximation: number of actions to be sampled
        :param reparametrization: specify whether we use the reparametrization trick to compute the gradient
        """
        super().__init__(r"LSTD$\Gamma$", policy)
        self._critic_features = critic_features
        self._actor_features = actor_features
        self._gamma = task_descriptor.gamma
        self._lambda = _lambda
        self._mu = 0
        self._pure_td = pure_td
        self._less_computation = less_computation
        self._delayed = delayed

        if task_descriptor.initial_state_distribution.is_deterministic():
            self._s_0 = np.array([task_descriptor.initial_state_distribution.sample()])
        else:
            self._s_0 = np.array([task_descriptor.initial_state_distribution.sample() for i in range(n_starting_state)])

        self._n_critic_approximation = n_critic_approximation
        self._reparametrization = reparametrization
        self._regularization = regularization

        self._full_training = head_parameters is None and body_parameters is None
        if self._full_training:
            self.n_p = len(self.policy.get_parameters())
        else:
            self.n_p = len(self.policy.get_parameters(head_parameters))

        self._head_parameters = head_parameters
        self._numerical_position_head = numerical_position_head

        self.n_f = n_critic_features

        self._G = np.zeros((self.n_f, self.n_p))
        self._H = np.zeros((self.n_f, self.n_p))
        self._beta = regularization
        self._omega = np.zeros((self.n_f, 1))
        self._sigma = np.zeros((self.n_f, 1))

        self._alpha = learning_rate
        self._decreasing_alpha = decreasing_learning_rate
        self._t = np.ones((8, 2))

    def phi(self, s: np.ndarray, a: np.ndarray):
        """
        Compute critic features per given state action pairs
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        :return:
        """
        return self._critic_features(torch.tensor(s), torch.tensor(a)).detach().numpy()

    def get_alpha(self, s: np.ndarray, a: np.ndarray):
        """
        Get a state-action based learning rate. (only if decreasing_learning_rate in the init was True)
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        :return:
        """
        if self._decreasing_alpha:
            return 1/self._t[s[0], a[0]]
        else:
            return self._alpha

    def get_scalar_Q(self, s: np.ndarray, a: np.ndarray):
        """
        Get a scalar Q-value for a state action pair
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        :return:
        """
        return self.get_Q(s, a)[0]

    def get_vector_Gamma(self, s, a):
        """
        Get a 1D vector representing Gamma for a single state-action pair
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        :return:
        """
        return self.get_Gamma(s, a).ravel()

    def delta_critic(self,  s: np.ndarray, a: np.ndarray, r: np.ndarray, s_n: np.ndarray, a_n: np.ndarray, t: bool):
        """
        TD error for the critic (omega parameters) for a s, a, r, s_n, a_n tuple
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        :param r: scalar reward
        :param s_n: 1D vector representing the next_state
        :param s_n: 1D vector representing the next_state
        :param a_n: 1D vector representing the next_action
        :param t: True if the transition is a terminal one
        :return:
        """
        return r + self._gamma * (1-t) * self.get_scalar_Q(s_n, a_n) - self.get_scalar_Q(s, a)

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
        alpha = self.get_alpha(s, a)

        a_n = self.get_on_policy_action(s_n)
        delta = self.delta_critic(s, a, r, s_n, a_n, t)

        x = self.phi(s, a).T
        x_n = self.phi(s_n, a_n).T
        #
        if self._pure_td:
            self._omega = self._omega + alpha * delta * x
        else:
            self._omega = self._omega + alpha * delta * x - alpha * self._gamma * (self._sigma.T @ x)[0, 0] * x_n
            self._sigma = self._sigma \
                          + alpha * (delta - (self._sigma.T @ x)[0, 0]) * x - alpha * self._beta * self._sigma

    def get_nabla_log(self, s: np.ndarray, a: np.ndarray):
        """
        Compute \nabla_\theta \log \pi(a|s) for a single state action pair
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        """
        state = s
        if self._actor_features is not None:
            state = self._actor_features.codify_state(s).ravel()
        # TODO: check if differentiable creates problem. Probably yes.
        try:
            prob = self.policy.get_prob(torch.tensor([state]), torch.tensor([a]), differentiable=True)
        except:
            prob = self.policy.get_prob(torch.tensor([state]), torch.tensor([a]))

        log_prob = torch.log(prob)
        log_prob = log_prob.squeeze()
        log_prob.backward()
        try:
            ret = self.policy.get_gradient(numerical_position=self._numerical_position_head)
        except:
            ret = self.policy.get_gradient()

        self.policy.zero_grad()
        return ret

    def get_delta_gradient(self, s: np.ndarray, a: np.ndarray, s_n: np.ndarray, a_n: np.ndarray, t: bool):
        """
        Update the critic (omega parameters) for s_n, a_n tuple
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        :param s_n: 1D vector representing the next_state
        :param a_n: 1D vector representing the next_action
        :param t: True if the transition is a terminal one
        :return:
        """
        if self._delayed:
            immediate_gradient = self._gamma * self.get_scalar_Q(s_n, a_n) * self.get_nabla_log(s_n, a_n)
        else:
            immediate_gradient = self.get_scalar_Q(s, a) * self.get_nabla_log(s, a)
        return (immediate_gradient +
                   self._gamma * (1-t) * self.get_vector_Gamma(s_n, a_n) - self.get_vector_Gamma(s, a)).reshape(-1, 1)

    def update_gradient(self, s: np.ndarray, a: np.ndarray, s_n: np.ndarray, t: bool):
        """
        Update the critic (omega parameters) for a s, a, s_n tuple
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        :param s_n: 1D vector representing the next_state
        :param t: True if the transition is a terminal one
        :return:
        """

        alpha = self.get_alpha(s, a)

        a_n = self.get_on_policy_action(s_n)
        epsilon = self.get_delta_gradient(s, a, s_n, a_n, t)

        x = self.phi(s, a).T
        x_n = self.phi(s_n, a_n).T
        if self._pure_td:
            self._G = self._G + alpha * x @ epsilon.T
        else:
            self._G = self._G + alpha * x @ epsilon.T - alpha * self._gamma * x_n @ x.T @ self._H
            self._H = self._H + alpha * x @ epsilon.T - alpha * x @ x.T @ self._H - alpha * self._beta * self._H

    def update_time(self, s: np.ndarray, a: np.ndarray):
        """
        Update state-action visitation. (used for continuous state actions)
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        """
        self._t[s[0], a[0]] += 1.

    def get_on_policy_action(self, s: np.ndarray):
        """
        Sample an action from the policy taking in consideration actor aliasing
        :param s: 1D vector representing a state
        """

        state = s
        if self._actor_features is not None:
            state = self._actor_features.codify_state(s).ravel()
        return self.policy.get_action(state)

    def get_Q(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Get the Q-function for many state action pairs
        :param state: n x state_dim vector
        :param action: n x action_dim vector
        :return: (n) vector
        """
        return self.phi(state, action) @ self._omega.ravel()

    def get_Gamma(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Get the Gamma-function for many state action pairs
        :param state: n x state_dim vector
        :param action: n x action_dim vector
        :return: n x gradient_dim vector
        """
        return self.phi(state, action) @ self._G

    def get_surrogate_loss(self, s):
        # TODO: implement
        if self._head_parameters is None:
            parameters = self.policy.get_torch_parameters()
        else:
            parameters = self.policy.get_torch_parameters(self._numerical_position_head)

        a = self.get_on_policy_action(s)

        if self._delayed:
            # Equation 13 of the paper
            gradient = torch.tensor(self.get_Q(s, a) * self.get_nabla_log(s, a)\
                                    + (1-self._lambda) * self.get_Gamma(s, a).reshape(-1)).reshape(-1)
        else:
            gradient = torch.tensor(self._lambda * self.get_Q(s, a) * self.get_nabla_log(s, a)\
                                    + (1-self._lambda) * self.get_Gamma(s, a).reshape(-1)).reshape(-1)

        # Equation 13 of the paper
        return self._mu * torch.inner(torch.tensor(gradient), parameters)

    def get_surrogate_body_loss(self, s):
        # TODO: implement
        parameters = self.policy.get_torch_parameters()
        a = self.get_on_policy_action(s)

        state = torch.tensor(s)
        action = torch.tensor(a)

        ret = self.get_Q(s, a).item() * torch.log(self.policy.get_prob(state, action, differentiable=True))

        # Equation 13 of the paper
        return torch.mean(ret)

    def get_gradient(self):
        """
        Estimate the gradient.
        This code take full advantage of automatic differentiation, and on the fact that G_TD = \nabla_theta \omega_TD,
        as in Lemma 2. \omega_TD is fully differentiable with pytorch.
        :return:
        """
        loss = -self.get_surrogate_loss()
        loss.backward()
        ret = self.policy.get_gradient()
        self.policy.zero_grad()
        return ret

    def update_body_policy(self, optimizer, s):
        optimizer.zero_grad()
        j = -self.get_surrogate_body_loss(s)
        j.backward()
        optimizer.step()
        self.policy.zero_grad()

    def update_policy(self, optimizer, s):
        if self._mu != 0. or not self._less_computation:
            optimizer.zero_grad()
            j = -self.get_surrogate_loss(s)
            j.backward()
            optimizer.step()
        self._mu *= self._gamma * self._lambda
        self.policy.zero_grad()

    def reset(self):
        self._mu = 1.

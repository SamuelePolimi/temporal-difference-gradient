import numpy as np
import torch

from typing import Union, Callable
from herl.rl_interface import PolicyGradient, Critic, RLAgent, RLParametricModel, RLDataset, RLTaskDescriptor,\
    DeterministicState, StochasticState, StateDistribution
from herl.actor import NeuralNetwork, FixedGaussianNeuralNetworkPolicy


class DeepLambdaTDGamma(PolicyGradient, Critic):

    def __init__(self, policy: Union[RLAgent, RLParametricModel], task_descriptor: RLTaskDescriptor,
                value_critic_constructor: Callable,
                 gradient_critic_constructor: Callable,
                 value_optimizer,
                 gradient_optimizer,
                 actor_optimizer,
                 tau=1E-2,
                 _lambda=0., delayed=True):
        """
        LSTDGamma as described in Algorithm 1.
        :param policy: A parametric policy
        :param critic_features: The features of the critic used by LSTD
        :param task_descriptor: Descriptor of the task to solve
        :param n_starting_state: number of starting states to be sampled
        :param n_critic_approximation: number of actions to be sampled
        :param reparametrization: specify whether we use the reparametrization trick to compute the gradient
        """
        super().__init__(r"DeepLambda", policy)
        state_dim = task_descriptor.environment_descriptor.state_dim
        action_dim = task_descriptor.environment_descriptor.action_dim

        self._value_critic = value_critic_constructor()     # type: NeuralNetwork
        self._value_critic_t = value_critic_constructor()   # type: NeuralNetwork
        self._value_critic_t.set_parameters(self._value_critic.get_parameters())

        self._gradient_critic = gradient_critic_constructor()   # type: NeuralNetwork
        self._gradient_critic_t = gradient_critic_constructor() # type: NeuralNetwork
        self._gradient_critic_t.set_parameters(self._gradient_critic.get_parameters())

        self._value_optimizer = value_optimizer(self._value_critic.parameters())
        self._gradient_optimizer = gradient_optimizer(self._gradient_critic.parameters())
        self._actor_optimizer = actor_optimizer(self.policy.parameters())

        self._gamma = task_descriptor.gamma
        self._lambda = _lambda
        self._tau = tau
        self._mu = 1.
        self._delayed = delayed

        self._loss = torch.nn.HuberLoss()

    def get_nabla_log(self, s, a):
        """
        Compute \nabla_\theta \log \pi(a|s) for a single state action pair
        :param s: 1D vector representing a state
        :param a: 1D vector representing an action
        """
        self.policy.zero_grad()
        prob = self.policy.get_prob(s, a, differentiable=True)
        log_prob = torch.log(prob)
        log_prob = log_prob.squeeze()
        log_prob.backward()
        ret = self.policy.get_gradient()
        self.policy.zero_grad()
        return torch.tensor(ret)

    def get_gradient(self, s, a, likelihood=True):
        if likelihood:
            return self.Q_t(s, a) * self.get_nabla_log(s, a).unsqueeze(0)
        else:
            self.zero_all_grads()
            loss = self.Q_t(s, self.policy(s))
            loss.backward()
            ret = self.policy.get_gradient()
            self.zero_all_grads()
            return torch.tensor(ret)

    def Q(self, s, a):
        return self._value_critic(s, a, differentiable=True)

    def get_Q(self, s, a):
        return self._value_critic(s, a)

    def get_V(self, s):
        return self._value_critic(s, self.policy.get_action(s))

    def Q_t(self, s, a):
        return self._value_critic_t(s, a, differentiable=True)

    def Gamma(self, s, a):
        return self._gradient_critic(s, a, differentiable=True)

    def Gamma_t(self, s, a):
        return self._gradient_critic_t(s, a, differentiable=True)

    def get_value_critic_loss(self, s, a, r, s_n, t):
        s_t, a_t, s_n_t = [torch.tensor(x).unsqueeze(0) for x in [s, a, s_n]]
        a_n = self.policy(s_n_t)
        y = r + self._gamma * (1-t) * self.Q_t(s_n_t, a_n)
        return self._loss(y.detach(), self.Q(s_t, a_t))

    def get_gradient_critic_loss(self, s, a, r, s_n, t):
        s_t, a_t, s_n_t = [torch.tensor(x).unsqueeze(0) for x in [s, a, s_n]]
        a_n = self.policy(s_n_t)
        if not self._delayed:
            g = self.get_gradient(s_t, a_t)
        else:
            g = self._gamma * self.get_gradient(s_t, a_n, likelihood=False)
        y = g + self._gamma * (1-t) * self.Gamma_t(s_n_t, a_n)
        return (y.detach() - self.Gamma(s_t, a_t))**2

    def soft_update(self, model_a, model_b):
        parameters_a = model_a.get_parameters()
        parameters_b = model_b.get_parameters()

        new_parameters = (1 - self._tau) * parameters_b + self._tau * parameters_a

        model_b.set_parameters(new_parameters)

    def update_value_critic(self, s, a, r, s_n, t):
        self.zero_all_grads()
        loss = torch.mean(self.get_value_critic_loss(s, a, r, s_n, t))
        loss.backward()
        # print("value loss", loss)
        self._value_optimizer.step()
        self.zero_all_grads()
        self.soft_update(self._value_critic, self._value_critic_t)
        return loss

    def zero_all_grads(self):
        self._value_optimizer.zero_grad()
        self._gradient_critic.zero_grad()
        self._actor_optimizer.zero_grad()

    def update_gradient_critic(self, s, a, r, s_n, t):
        self.zero_all_grads()
        loss = torch.mean(self.get_gradient_critic_loss(s, a, r, s_n, t))
        loss.backward()
        # print("vgradient loss", loss)
        self._gradient_optimizer.step()
        self.zero_all_grads()
        self.soft_update(self._gradient_critic, self._gradient_critic_t)
        return loss

    def update_actor(self, s):
        self.zero_all_grads()
        s_0 = torch.tensor(s).unsqueeze(0)
        a_0 = self.policy(s_0)
        parameters = self.policy.get_torch_parameters()
        g = self.Gamma_t(s_0, a_0).detach().squeeze()
        loss = - self._mu * ((1 - self._lambda) * torch.inner(parameters, g) - self.Q_t(s_0, a_0))
        loss.backward()
        self._actor_optimizer.step()
        self._mu *= self._lambda    # * self._gamma
        self.zero_all_grads()

    def reset(self):
        self._mu = 1.

    def update_actor_classic(self, s):
        s_t = torch.tensor(s).unsqueeze(0)
        loss = - (self.Q(s_t, self.policy(s_t)))
        loss.backward()
        self._actor_optimizer.step()
        self.zero_all_grads()





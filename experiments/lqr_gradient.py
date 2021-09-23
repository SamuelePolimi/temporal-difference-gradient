import torch.nn
import numpy as np

from herl.classic_envs import LQR
from herl.analysis_center.lqr_solver import LQRAnalyzer
from herl.actor import LinearGaussianPolicy
from herl.rl_interface import RLTask
from herl.rl_interface import DeterministicState
from herl.utils import ProgressBar, Printable

from algorithms import LSTDGamma

print(r"""UNBIASEDNESS OF LSTD\Gamma on LQR
=========================

This experiment aims to prove empirically that the gradient estimation of LSTD\Gamma is unbiased on LQR.
In fact, LQR allows us to easily construct *perfect features* (see Theorem 2), since we know that the Q-function is 
quadratic w.r.t. states and actions.

To prove the unbiasedness, we first compute the gradient of LQR in closed form. Then, we sample off-policy datasets, and we collect
a high number of indipendent gradient estimates with different instantiations of LSTD\Gamma.

If the average gradient estimate is *close enough* to the true gradient, we can think that the gradient estimator is unbiased.


""")


"""
DEFINITION OF THE LQR SYSTEM
"""

A = 0.01 * np.eye(2)
B = 0.001 * np.eye(2)
Q = np.eye(2)
R = np.eye(2)

Cov = 0.001*np.eye(2)

s_0 = np.ones(2)*0.5
state_box = 100*np.ones(2)
action_box = 100*np.ones(2)
lqr = LQR(A, B, Q, R, s_0, state_box, action_box)
policy = LinearGaussianPolicy(2, 2, Cov, diagonal=True)

lqr_task = RLTask(lqr, DeterministicState(s_0), max_episode_length=100, gamma=0.9)
analyzer = LQRAnalyzer(lqr_task, policy)

"""
DEFINITION OF THE CRITIC FEATURES
"""


def features(s, a, differentiable=False):
    s_q, a_q = s**2, a**2
    n = len(s_q.shape)
    if differentiable:
        return torch.cat([s_q, a_q, s, a, torch.ones_like(s[..., 0:1])], dim=n-1)
    else:
        return np.concatenate([s_q, a_q, s, a, np.ones_like(s[..., 0:1])], axis=n-1)


"""
DEFINITION OF THE OFF-POLICY SAMPLING PROCEDURE
"""


def collect_offpolicy_data(n_samples):
    s = np.random.normal(0., 1., size=(n_samples, 2))
    a = np.random.normal(0., 1., size=(n_samples, 2))
    s_n = s @ A.T + a @ B.T
    r = -np.sum((s @ R.T) * s + (a @ Q.T) * a, axis=1, keepdims=True)

    dataset = lqr_task.get_empty_dataset(n_samples)
    dataset.notify_batch(state=s, action=a, next_state=s_n, reward=r, terminal=np.zeros_like(r))
    return dataset


"""
TAKING THE AVERAGE OF MANY LSTD-Gamma ESTIMATES
"""


def get_avg_lstd_gamma_gradient(n_estimates=10000, n_samples=100000, reparametrization=True):

    g = []

    progress = ProgressBar(Printable(r"Average LSTD\Gamma"), max_iteration=n_estimates)
    for _ in range(n_estimates):
        dataset = collect_offpolicy_data(n_samples)

        lstd_gamma = LSTDGamma(policy, features, dataset, lqr_task.get_descriptor(), reparametrization=reparametrization)
        g.append(lstd_gamma.get_gradient())
        progress.notify()

    return np.mean(g, axis=0)


print("-"*50)
print("The true gradient is %s" % analyzer.get_gradient())
print("-"*50)
print(r"The average LSTD\Gamma estimate is %s" % get_avg_lstd_gamma_gradient(n_estimates=100000))
print("-"*50)
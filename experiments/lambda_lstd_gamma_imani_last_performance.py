import numpy as np
import torch.optim
import matplotlib.pyplot as plt
import tikzplotlib
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})


from herl.classic_envs import get_imani_mdp
from herl.actor import TabularPolicy
from herl.rl_interface import RLTask
from herl.rl_analysis import MDPAnalyzer
from herl.utils import ProgressBar, Printable, _one_hot
from herl.solver import RLCollector

from algorithms import LambdaLSTDGamma, SemiGradient
from settings import ImaniCounterexample


print(r"""EMPIRICAL IMANI COUNTEREXAMPLE
=========================

We save the last performance of LSTD\Gamma on the counterexample of Imani et al, 2018 for many values of \lambda.


The output of this experiment will be found in `/plots/imani/lambda_lstd_gamma_last/`

""")


actor_aliasing = True
critic_aliasing = False

setting = ImaniCounterexample()

analyzer = setting.analyzer



def critic_features(s, a):
    state = s
    n_codes = setting.n_states * setting.n_actions
    if critic_aliasing:
        state = setting.actor_features.codify_state(s)
        n_codes = (setting.n_states - 1) * setting.n_actions

    s_a_t = _one_hot(state.int() * setting.n_actions + a.int(), n_codes)
    return s_a_t

# ------------------------
# Dataset Generation
# ------------------------


n_samples = 500


def get_dataset():
    dataset = setting.mdp_task.get_empty_dataset(n_samples)
    collector = RLCollector(dataset, setting.mdp_task, setting.behavior_policy, episode_length=4)
    collector.collect_samples(n_samples)
    return dataset


lambdas = np.linspace(0., 1., 20)
# Closed form versions of LSTDGamma and SemiGradient
# if actor_aliasing:
#     lstd_gamma = [LambdaLSTDGamma(setting.policy, critic_features, setting.actor_features, dataset,
#                            setting.mdp_task.get_descriptor(), regularization=0., _lambda=_lambda) for _lambda in lambdas]
# else:
#     lstd_gamma = [lambda dataset: LambdaLSTDGamma(setting.policy, critic_features, None, dataset,
#                      setting.mdp_task.get_descriptor(), regularization=0., _lambda=_lambda) for _lambda in
#      lambdas]

# --------------------------------------------
# Train the policy with a given algorithm
# --------------------------------------------


def train(policy_gradient):
    n_iterations = 1000
    setting.policy.set_parameters(setting.init_parameters)
    adam = torch.optim.Adam(setting.policy.parameters(), lr=0.01)
    pb = ProgressBar(Printable("Policy Gradient"), max_iteration=n_iterations, prefix='Progress %s' % policy_gradient.name)
    for i in range(n_iterations):
        policy_gradient.update_policy(adam)
        pb.notify()
    return (1-setting.mdp_task.gamma) * analyzer.get_return().detach().numpy()


# ---------------------------------------------
# Plot the learning curve
# ---------------------------------------------

def plot(results):
    color = np.array([1., 0.5, 0.])
    n_res = results.shape[1]
    resolution = results.shape[0]
    ret_sg_m = np.mean(results, axis=1)
    ret_sg_t = 1.96 * np.std(results, axis=1) / np.sqrt(n_res)
    x = np.array(range(resolution))/resolution
    plt.plot(x, ret_sg_m, label=r"$\lambda$-LSTD$\Gamma$")
    plt.fill_between(x, ret_sg_m + ret_sg_t, ret_sg_m - ret_sg_t, alpha=0.5, color=color)


ret_gamma = [np.array([train(
    LambdaLSTDGamma(setting.policy, critic_features, setting.actor_features, get_dataset(),
                               setting.mdp_task.get_descriptor(), regularization=0., _lambda=_lambda)
) for _ in range(5)]) for _lambda in lambdas]

plot(np.array(ret_gamma))

plt.ylabel(r"$J(\theta)$")
plt.xlabel("$\lambda$")
plt.legend(loc="best")

np.save("../plots/imani/lambda_lstd_gamma_last/ret_gamma.npy", ret_gamma)
critic_name = "critic_aliasing" if critic_aliasing else "no_critic_aliasing"
actor_name = "actor_aliasing" if actor_aliasing else "no_actor_aliasing"
tikzplotlib.save("../plots/imani/lambda_lstd_gamma_last/%s_%s.tex" % (actor_name, critic_name))
plt.savefig("../plots/imani/lambda_lstd_gamma_last/%s_%s.pdf" % (actor_name, critic_name))



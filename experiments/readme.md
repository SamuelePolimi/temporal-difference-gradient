Experiments
==

Imani Counterexample Closed Form (`closed_form_imani.py`)
--

We test LSTD$Gamma on the counterexample of Imani et al, 2018, 
showing that our method achieves higher performance than the Semi-Gradient.

Differently from the original paper, we consider the classic return as metric to be optimized.
We also have a slightly different (but equivalent) treatment of the terminal state. In our case, the terminal 
state is just absorbing with reward=0. This modifies the on-policy distribution, but yields same expected return and 
same policy gradient.

The output of this experiment will be found in `plots/imani-counterexample.pdf` and `.tikz`.

Unbiasedness of LSTDGamma in LQR (`lqr_gradient.py`)
--

This experiment aims to prove empirically that the gradient estimation of LSTD\Gamma is unbiased on LQR.
In fact, LQR allows us to easily construct *perfect features* (see Theorem 2), since we know that the Q-function is 
quadratic w.r.t. states and actions.

To prove the unbiasedness, we first compute the gradient of LQR in closed form. Then, we sample off-policy datasets, and we collect
a high number of indipendent gradient estimates with different instantiations of LSTD\Gamma.

If the average gradient estimate is *close enough* to the true gradient, we can think that the gradient estimator is unbiased.

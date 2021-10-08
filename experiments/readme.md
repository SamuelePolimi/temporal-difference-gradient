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

The output of this experiment will be found in `plots/imani/closed_form_lstd_gamma`

Lambda-LSTDGamma Learning Curves (`lambda_lstd_gamma_imani.py`)
--

__Figure 3b__

Produces the learning curves of different values of lambda for `Imani MPD`. 
Output in `plots/imani/lambda_lstd_gamma`.

Lambda-LSTDGamma Final Performance. (`lambda_lstd_gamma_imani_last_performance.py`)
--

__Figure 3c__

Visualizes the final performance of \lambda-LSTD\Gamma for many different values of \lambda.
Output in `plots/imani/lambda_lstd_gamma_last`.

Lambda-LSTDGamma Gradient (`lambda_lstd_gamma_imani_gradient.py`)
--


__Figure 4a__

Visualizes different gradient estimates for different values of Lambda.
Output in `plots/imani/lambda_lstd_gamma_gradient`

Lambda-LSTDGamma Bias and Variance (`lambda_lstd_gamma_imani_bias.py`)
--

__Figure 4b__

Visualizes the MSE of the gradient estimation decomposed in bias and variance.
Output in `plots/imani/lambda_lstd_gamma_bias`




Figure 3a of the paper. 
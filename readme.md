A Temporal Difference Approach to Policy Gradient Estimation
===

__[Samuele Tosatto](https://samueletoatto.com), [Andrew Patterson](https://andnp.github.io/), [Martha White](https://webdocs.cs.ualberta.ca/~whitem/), A. [Rupam Mahmood](https://armahmood.github.io/)__
*The international Conference of Machine Learning 2022 (Baltimore)*


This repository contains the code to run the experiments contained in our submission to AISTATS 2022.
The code runs with `python3.6` and relies on [HeRL](https://github.com/SamuelePolimi/HeRL).

To install `HeRL` simply download it, and run `pip install -e .`.

The repository is organized as follow: 
- `algorithms` contain a simple version of the algorithms listed in the paper
- `algorithm/closed` contains a closed-form version of the algorithms, that runs with Finite MDPs
- `environments` contains the environments used in these experiments
- `experiments` contains the scripts of the experiments
- `plots` contains tikz figures ready to be inserted in the paper

Suggestion: to render progress bars, enable `emulate in console` if using `pycharm`.

TODO
--

- [x] Write a version of Imani's counterexample with empirical estimation of the gradient
- [x] Implement semigradient with given critic
- [x] Implement RCGamma
- [ ] Test LSTDGamma on ContinuousMountainCar (RBF Features?)
- [ ] Test TDCGamma on ContinuousMountainCar (RBF Features?)

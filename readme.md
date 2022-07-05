A Temporal Difference Approach to Policy Gradient Estimation
===

__[Samuele Tosatto](https://samueletoatto.com), [Andrew Patterson](https://andnp.github.io/), [Martha White](https://webdocs.cs.ualberta.ca/~whitem/), [A. Rupam Mahmood](https://armahmood.github.io/)__
*The international Conference of Machine Learning 2022 (Baltimore)*

This repository contains the code to run the experiments contained in our submission to ICML 2022.
The code runs with `python3.6`.

We suggest creating a conda environment, and to install HeRL with `cd herl` and `pip install -e .`.

The repository is organized as follow: 
- `algorithms` contain a simple version of the algorithms listed in the paper
- `environments` contains the environments used in these experiments
- `experiments` contains the scripts of the experiments
- `plots` contains tikz figures (almost) ready to be inserted in the paper, and numpy files used to generate them
- `readme.md` contains a description of this project.

How to replicate results in the paper
---

- Figure 1b
  
```shell
cd experiments
python3 lambda_lstd_gamma_imani_bias.py
cd ../plots/imani/lambda_lstd_gamma_bias/plot.py
```

- Figure 1c, 1b, (and Figure 5 in Appendix)

```shell
cd experiments
python3 lambda_lstd_gamma_imani_gradient.py
cd ../plots/imani/lambda_lstd_gamma_gradient/plot.py
cd ../plots/imani/lambda_lstd_gamma_gradient/scatter.py
```

- Figure 2a
  
```shell
cd experiments
python3 lambda_lstd_gamma_imani_last_performance.py
cd ../plots/imani/lambda_lstd_gamma_last/plot.py
```

- Figure 2b

```shell
cd experiments
python3 lambda_rc_gamma_learning.py
cd ../plots/imani/lambda_lstd_gamma_learning/plot.py
```

- Figure 2c and d

```shell
cd parallel_experiments
sh mdps_learning.sh
cd ../plots/mdps/learning/plot.py
```


Suggestion: to render progress bars, enable `emulate in console` if using `pycharm`.

To cite us:
```json
@inproceedings{tosatto_temporal_2022,
        title = {A {Temporal}-{Difference} {Approach} to {Policy} {Gradient} {Estimation}},
        author = {Tosatto, Samuele and Patterson, Andrew and White, Martha and Mahmood, Rupam},
        booktitle = {Proceedings of the 39th {International} {Conference} on {Machine} {Learning}}
        year = {2022},
        note = {event-place: Baltimore, Meryland, United States of America},
        organization = {PMLR}
}
```




# Data Privacy Optimization

Code for implementation of my undergraduate thesis - _"Value-oriented Privacy Optimization in Model Based Data Marketplace"_.

### Prerequisites

- Python, NumPy, Scikit-learn, Tqdm, PyTorch

### Basic Usage

Including two parts:

* DPRL (**D**ata **P**rivacy optimization using **R**einforcement **L**earning)

* FDPRL (**F**ederated **D**ata **P**rivacy optimization using **R**einforcement **L**earning)

Attention: FDPRL is the federated version of DPRL, although there are some difference in framework design.

DPRL offers these abilities:

1. Given an epsilon budget list, optimize the allocation in value-oriented. (discrete)
2. Only the whole epsilon budget is given, optimize the budget distribution. (continuous)


#### Run Example Experiments

```
$ python3 examples.py
```

If you have browser env, jupyter notebook is recommended.

```
$ jupyter_notebook examples.ipynb
```

### Documents

More detailed usages and code implementation can refer to the documents.

```
$ make doc
```

(\* Documents are powered by [Sphinx](https://github.com/sphinx-doc/sphinx).)

### License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Reference

[1] [Collecting and Analyzing Multidimensional Data with Local Differential Privacy (ICDE '19)](https://arxiv.org/pdf/1907.00782.pdf)

[2] [Data Valuation using Reinforcement Learning (ICML '20)](http://proceedings.mlr.press/v119/yoon20a/yoon20a.pdf)

[3] [Differentially Private Federated Learning: A Client Level Perspective (ICLR '19)](https://arxiv.org/pdf/1712.07557.pdf)

[4] [Learning Differentially Private Recurrent Language Models (ICLR '18)](https://arxiv.org/abs/1710.06963)

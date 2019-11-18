# Overview

Group project for Columbia course COMS4995-004 Optimization for Machine Learning ([webpage](https://www.satyenkale.com/optml-f19/)). The goal of the project is to implement and understand the optimization methods from the paper "Stochastic Variance Reduction for Nonconvex Optimization".

## Scripts

### `svrg.py`

Runs SVRG optimization using MLP against the MNIST dataset. Performs several epochs of SGD during warmup phase before SVRG updates. Optionally outputs metrics of the run, and plots the loss curve.

Example Usage:

`python svrg.py --num_outer_epochs 10 --num_inner_epochs 2 --device cuda --metrics_path "runs/svrg_output.json" --plot`

## Group Members

* Ryan Goldenberg
* Liyi (Leo) Zhang
* Chengkuan (Kuan) Chen

## Experiment Setting
### Reproducible work
**MNIST SVRG**
  - learning rate: To be tuned
  - num_warmup_epoch 10
  - num_outer_epoch 242 (solve from (sn + 2sm / n) = 290 with m = n/10)
  - inner_epoch_fraction 0.1 (from m = n / 10)
  - batch_size 10
**MNIST SGD**
  - learning rate: To be tuned
  - num_epoch 300
  - batch_size 10
  - l2-regularization 1e-3

**CIFAR SVRG**
  - learning rate: To be tuned
  - num_warmup_epoch: 10
  - num_outer_epoch: 408 (solve from (sn + 2sm / n) = 490 with m = n/10, we assume the rightmost value of x-axis is 500)
  - inner_epoch_fraction 0.1
  - batch_size: 10

**CIFAR SGD**
  - learning rate: To be tuned
  - num_epoch 500
  - batch_size 10
  - l2-regularization 1e-3

**STL SVRG**
  - learning rate: To be tuned
  - num_warmup_epoch: 20
  - num_outer_epoch 242 (solve from (sn + 2sm / n) = 290 with m = n/10)
  - inner_epoch_fraction 0.1 (from m = n / 10)
  - batch_size 10


**STL SGD**
  - learning rate: To be tuned
  - num_epoch 300
  - batch_size 10
  - l2-regularization 1e-4

### Further evaluation work
## References	
**Deeper Neural Net**
  - Compare method: SGD/SVRG
  - dataset: MNIST
  - network: 3 layer MLP with hidden unit = 100 and relu as activate function
  
**Rosenbrock function**
  - Goal: Whether SVRG is better than SGD when the objective function has many saddle points
  - Compare mathod: SGD/SVRG
  - dataset: Rosenbrock data (size = 60000)

**Eggholder function**
  - Goal: Whether SVRG is better than SGD when the objective function has many local minima
  - Compare mathod: SGD/SVRG
  - dataset: Rosenbrock data (size = 60000)

## References

* Stochastic Variance Reduction for Nonconvex Optimization ([arxiv](https://arxiv.org/pdf/1603.06160))
* Accelerating Stochastic Gradient Descent using Predictive Variance Reduction ([arxiv](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf))

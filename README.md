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


## References

* Stochastic Variance Reduction for Nonconvex Optimization ([arxiv](https://arxiv.org/pdf/1603.06160))
* Accelerating Stochastic Gradient Descent using Predictive Variance Reduction ([arxiv](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf))

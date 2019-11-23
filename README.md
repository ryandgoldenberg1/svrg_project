# Overview

Group project for Columbia course COMS4995-004 Optimization for Machine Learning ([webpage](https://www.satyenkale.com/optml-f19/)). The goal of the project is to implement and understand the optimization methods from the paper "Stochastic Variance Reduction for Nonconvex Optimization".

[Final report](https://drive.google.com/file/d/1erhDAdLBGDkpXfznAalONYDWySlnkY3v/view?usp=sharing)

## Scripts


* `train.py`: Trains an MLP using SGD or SVRG. See `train.py -h` for options including dataset and hyperparameters.
* `plot.py`: Plots the results of several runs of `train.py` in a single figure.

### Example Usage

```bash
# SGD Run
python train.py \
  --seed 63 \
  --optimizer SGD \
  --dataset MNIST \
  --dataset_size 1000 \
  --run_name sgd \
  --output_path sgd.json \
  --layer_sizes 784 10 \
  --batch_size 1 \
  --learning_rate 0.1 \
  --weight_decay 0.001 \
  --num_epochs 20

# SVRG Run
python train.py \
  --seed 63 \
  --optimizer SVRG \
  --dataset MNIST \
  --dataset_size 1000 \
  --run_name svrg \
  --output_path svrg.json \
  --layer_sizes 784 10 \
  --batch_size 1 \
  --learning_rate 0.025 \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.1 \
  --num_warmup_epochs 1 \
  --num_outer_epochs 7 \
  --num_inner_epochs 2

# Plot Results
python plot.py --key train_loss --run_paths sgd.json svrg.json --save_path sgd_svrg.png
```

## Group Members

* Ryan Goldenberg
* Liyi (Leo) Zhang
* Chengkuan (Kuan) Chen

## Experiment Setting

### Reproducible work

Each experiment has a README including results and commands to reproduce.

#### [MNIST](experiments/nonconvex_mnist/README.md)
**MNIST SVRG**
  - learning rate: To be tuned
  - num_warmup_epoch 1
  - num_outer_epoch 242 (solve from (sn + 2sm / n) = 290 with m = n/10)
  - inner_epoch_fraction 0.1 (from m = n / 10)
  - batch_size 10

**MNIST SGD**
  - learning rate: To be tuned
  - num_epoch 300
  - batch_size 10
  - l2-regularization 1e-3

#### [CIFAR10](experiments/nonconvex_cifar10/README.md)
**CIFAR SVRG**
  - learning rate: To be tuned
  - num_warmup_epoch: 1
  - num_outer_epoch: 408 (solve from (sn + 2sm / n) = 490 with m = n/10, we assume the rightmost value of x-axis is 500)
  - inner_epoch_fraction 0.1
  - batch_size: 10

**CIFAR SGD**
  - learning rate: To be tuned
  - num_epoch 500
  - batch_size 10
  - l2-regularization 1e-3

#### [STL10](experiments/nonconvex_stl10/README.md)
**STL SVRG**
  - learning rate: To be tuned
  - num_warmup_epoch: 2
  - num_outer_epoch 242 (solve from (sn + 2sm / n) = 290 with m = n/10)
  - inner_epoch_fraction 0.1 (from m = n / 10)
  - batch_size 10


**STL SGD**
  - learning rate: To be tuned
  - num_epoch 300
  - batch_size 10
  - l2-regularization 1e-4

### Further evaluation work

**Deeper Neural Net**
  - Compare method: SGD/SVRG
  - dataset: MNIST
  - network: MLP with hidden 600, 300 and relu as activate function

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

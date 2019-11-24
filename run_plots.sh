#!/bin/bash

# CIFAR
python plot.py \
  --run_paths experiments/nonconvex_cifar10/cifar10_sgd_fixed001.json experiments/nonconvex_cifar10/cifar10_svrg.json \
  --key grad_norm \
  --save_path experiments/nonconvex_cifar10/compare-cifar10-gradnorm-fixed001.png

python plot.py \
  --run_paths experiments/nonconvex_cifar10/cifar10_sgd_fixed001.json experiments/nonconvex_cifar10/cifar10_svrg.json \
  --key train_loss \
  --save_path experiments/nonconvex_cifar10/compare-cifar10-trainloss-fixed001.png

python plot.py \
    --run_paths experiments/nonconvex_cifar10/warmup-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_cifar10/warmup_results.png

python plot.py \
    --run_paths experiments/nonconvex_cifar10/svrg-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_cifar10/svrg_results.png

# MNIST
python plot.py \
  --run_paths experiments/nonconvex_mnist/mnist_sgd_fixed001.json experiments/nonconvex_mnist/mnist_svrg.json \
  --key grad_norm \
  --save_path experiments/nonconvex_mnist/compare-mnist-gradnorm-fixed001.png

python plot.py \
  --run_paths experiments/nonconvex_mnist/mnist_sgd_fixed001.json experiments/nonconvex_mnist/mnist_svrg.json \
  --key train_loss \
  --save_path experiments/nonconvex_mnist/compare-mnist-trainloss-fixed001.png
python plot.py \
    --run_paths experiments/nonconvex_mnist/warmup-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_mnist/warmup_results.png

python plot.py \
    --run_paths experiments/nonconvex_mnist/svrg-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_mnist/svrg_results.png \
    --y_top 0.1 \
    --y_bottom 0.05
# STL
python plot.py \
  --run_paths experiments/nonconvex_stl10/stl10_sgd_fixed001.json experiments/nonconvex_stl10/stl10_svrg.json \
  --key grad_norm \
  --save_path experiments/nonconvex_stl10/compare-stl10-gradnorm-fixed001.png

python plot.py \
  --run_paths experiments/nonconvex_stl10/stl10_sgd_fixed001.json experiments/nonconvex_stl10/stl10_svrg.json \
  --key train_loss \
  --save_path experiments/nonconvex_stl10/compare-stl10-trainloss-fixed001.png
python plot.py \
    --run_paths experiments/nonconvex_stl10/warmup-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_stl10/warmup_results.png

python plot.py \
    --run_paths experiments/nonconvex_stl10/svrg-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_stl10/svrg_results.png \
    --y_top 2.5

# Deep MNIST

python plot.py \
  --run_paths experiments/nonconvex_mnist_deep/sgd-0.01.json experiments/nonconvex_mnist_deep/svrg-0.025.json \
  --key grad_norm \
  --save_path experiments/nonconvex_mnist_deep/grad_norm_sgd_svrg.png

python plot.py \
  --run_paths experiments/nonconvex_mnist_deep/sgd-0.01.json experiments/nonconvex_mnist_deep/svrg-0.025.json \
  --key train_loss \
  --save_path experiments/nonconvex_mnist_deep/train_loss_sgd_svrg.png

python plot.py \
    --run_paths experiments/nonconvex_mnist_deep/warmup-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_mnist_deep/warmup_results.png

python plot.py \
    --run_paths experiments/nonconvex_mnist_deep/svrg-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_mnist_deep/svrg_results.png \
    --y_top 0.5 \
    --y_bottom 0.001
# FMNIST
python plot.py \
  --run_paths experiments/nonconvex_fmnist/sgd-0.01.json experiments/nonconvex_fmnist/svrg-0.005.json \
  --key grad_norm \
  --save_path experiments/nonconvex_fmnist/grad_norm_sgd_svrg.png

python plot.py \
  --run_paths experiments/nonconvex_fmnist/sgd-0.01.json experiments/nonconvex_fmnist/svrg-0.005.json \
  --key train_loss \
  --save_path experiments/nonconvex_fmnist/train_loss_sgd_svrg.png

python plot.py \
    --run_paths experiments/nonconvex_fmnist/warmup-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_fmnist/warmup_results.png

python plot.py \
    --run_paths experiments/nonconvex_fmnist/svrg-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_fmnist/svrg_results.png

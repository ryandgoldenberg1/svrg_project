#!/bin/bash

# CIFAR
python plot.py \
  --run_paths experiments/nonconvex_cifar10/sgd-0.01.json experiments/nonconvex_cifar10/svrg-0.001.json \
  --key grad_norm \
  --save_path experiments/nonconvex_cifar10/gradnorm_sgd_lr001_svrg_lr0001.png

python plot.py \
  --run_paths experiments/nonconvex_cifar10/sgd-0.01.json experiments/nonconvex_cifar10/svrg-0.001.json \
  --key train_loss \
  --save_path experiments/nonconvex_cifar10/trainloss_sgd_lr001_svrg_lr0001.png

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
  --run_paths experiments/nonconvex_mnist/sgd-0.01.json experiments/nonconvex_mnist/svrg-0.05.json \
  --key grad_norm \
  --save_path experiments/nonconvex_mnist/gradnorm_sgd_lr001_svrg_lr005.png
python plot.py \
  --run_paths experiments/nonconvex_mnist/sgd-0.01.json experiments/nonconvex_mnist/svrg-0.05.json \
  --key train_loss \
  --save_path experiments/nonconvex_mnist/trainloss_sgd_lr001_svrg_lr005.png
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
  --run_paths experiments/nonconvex_stl10/sgd-0.01.json experiments/nonconvex_stl10/svrg-0.001.json \
  --key grad_norm \
  --save_path experiments/nonconvex_stl10/gradnorm_sgd_lr001_svrg_lr0001.png

python plot.py \
  --run_paths experiments/nonconvex_stl10/sgd-0.01.json experiments/nonconvex_stl10/svrg-0.001.json \
  --key train_loss \
  --save_path experiments/nonconvex_stl10/trainloss_sgd_lr001_svrg_lr0001.png
python plot.py \
    --run_paths experiments/nonconvex_stl10/warmup-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_stl10/warmup_results.png

python plot.py \
    --run_paths experiments/nonconvex_stl10/svrg-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_stl10/svrg_results.png \
    --y_top 2.5 \
    --y_bottom 0.0
# Deep MNIST

python plot.py \
  --run_paths experiments/nonconvex_mnist_deep/sgd-0.01.json experiments/nonconvex_mnist_deep/svrg-0.025.json \
  --key grad_norm \
  --save_path experiments/nonconvex_mnist_deep/gradnorm_sgd_lr001_svrg_lr0025.png

python plot.py \
  --run_paths experiments/nonconvex_mnist_deep/sgd-0.01.json experiments/nonconvex_mnist_deep/svrg-0.025.json \
  --key train_loss \
  --save_path experiments/nonconvex_mnist_deep/trainloss_sgd_lr001_svrg_lr0025.png

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
  --save_path experiments/nonconvex_fmnist/gradnorm_sgd_lr001_svrg_lr0005.png

python plot.py \
  --run_paths experiments/nonconvex_fmnist/sgd-0.01.json experiments/nonconvex_fmnist/svrg-0.005.json \
  --key train_loss \
  --save_path experiments/nonconvex_fmnist/trainloss_sgd_lr001_svrg_lr0005.png

python plot.py \
    --run_paths experiments/nonconvex_fmnist/warmup-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_fmnist/warmup_results.png

python plot.py \
    --run_paths experiments/nonconvex_fmnist/svrg-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_fmnist/svrg_results.png

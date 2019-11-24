#!/bin/bash

# For rproducibility result
echo "Running reproducibility experiment on MNIST, CIFAR10 and STL"

echo "Running SGD"
python train.py \
    --seed 79 \
    --optimizer SGD \
    --run_name sgd-0.01.json \
    --output_path experiments/nonconvex_mnist/sgd-0.01.json \
    --dataset MNIST \
    --layer_sizes 784 100 10 \
    --batch_size 10 \
    --learning_rate 0.01 \
    --weight_decay 0.001 \
    --num_epochs 260

python train.py \
    --seed 79 \
    --optimizer SGD \
    --run_name sgd-0.01.json \
    --output_path experiments/nonconvex_cifar10/sgd-0.01.json \
    --dataset CIFAR10 \
    --layer_sizes 3072 100 10 \
    --batch_size 10 \
    --learning_rate 0.01 \
    --weight_decay 0.001 \
    --num_epochs 260

python train.py \
    --seed 79 \
    --optimizer SGD \
    --run_name sgd-0.01.json \
    --output_path experiments/nonconvex_stl10/sgd-0.01.json \
    --dataset STL10 \
    --layer_sizes 27648 100 10 \
    --batch_size 10 \
    --learning_rate 0.01 \
    --weight_decay 0.001 \
    --num_epochs 260

for d in 'MNIST' 'CIFAR10' 'STL'; do
  echo "Running SVRG"
  python train.py \
    --seed 79 \
    --optimizer SVRG \
    --run_name svrg_0.05.json \
    --output_path experiments/nonconvex_$d/svrg-0.05.json \
    --dataset $d \
    --layer_sizes 784 100 10 \
    --batch_size 10 \
    --learning_rate 0.05 \
    --weight_decay 0.001 \
    --warmup_learning_rate 0.03 \
    --num_warmup_epochs 10 \
    --num_outer_epochs 250 \
    --num_inner_epochs 1 \
    --device cuda

  echo "Plotting the result of SGD and SVRG"
done
  
# For Critical evaluation result
echo "Running reproducibility experiment on MNIST, CIFAR10 and STL"


# For hyperparameter tuning result
echo "Running SVRG hyperparameter tuning of reproducibility experiment"
for d in 'MNIST' 'CIFAR10' 'STL'; do
  # Warmup Learning Rate Search
  for lr in 0.5 0.25 0.1 0.03 0.01 0.001; do python train.py \
    --seed 12 \
    --optimizer SGD \
    --run_name sgd_$lr.json \
    --output_path experiments/nonconvex_$d/warmup-$lr.json \
    --dataset $d \
    --layer_sizes 3072 100 10 \
    --batch_size 10 \
    --learning_rate $lr \
    --weight_decay 0.001 \
    --num_epochs 10
  done
  python plot.py \
    --run_paths experiments/nonconvex_$d/warmup-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_$d/warmup_results.png

  # SVRG Learning Rate Search
  for lr in 0.5 0.25 0.1 0.05 0.025 0.01 0.005 0.001; do python train.py \
    --seed 83 \
    --optimizer SVRG \
    --run_name svrg_$lr.json \
    --output_path experiments/nonconvex_$d/svrg-$lr.json \
    --dataset $d \
    --layer_sizes 3072 100 10 \
    --batch_size 10 \
    --learning_rate $lr \
    --weight_decay 0.001 \
    --warmup_learning_rate 0.01 \
    --num_warmup_epochs 10 \
    --num_outer_epochs 250 \
    --num_inner_epochs 1 \
    --device cuda
  done
  python plot.py \
    --run_paths experiments/nonconvex_$d/svrg-*.json \
    --key train_loss \
    --save_path experiments/nonconvex_$d/svrg_results.png
done
echo "Running SGD hyperparameter tuning of reproducibility experiment"

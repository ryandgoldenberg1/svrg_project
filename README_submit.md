Following command shown how to run the experiment. For generating plot used in paper, please refer to run_plot.sh.

# Requrement
  - Python3
  - pytorch >=1.3
  
Note that if you don't have cuda device, you should turn off the `--device cuda` flag.

# Reproducibility of paper result
## CIFAR 10
```bash
python train.py \
  --seed 83 \
  --optimizer SVRG \
  --run_name svrg_0.001.json \
  --output_path experiments/nonconvex_cifar10/svrg-0.001.json \
  --dataset CIFAR10 \
  --layer_sizes 3072 100 10 \
  --batch_size 10 \
  --learning_rate 0.001 \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.01 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda \
  --download
 ```
 
 ```bash
python train.py \
  --optimizer SGD \
  --run_name svrg_0.001.json \
  --output_path experiments/nonconvex_cifar10/sgd-0.001.json \
  --dataset CIFAR10 \
  --layer_sizes 3072 100 10 \
  --batch_size 10 \
  --learning_rate 0.001 \
  --weight_decay 0.001 \
  --num_epochs 260
  --device cuda \
  --download
 ```
## MNIST
```bash
python train.py \
  --seed 79 \
  --optimizer SVRG \
  --run_name svrg_0.05.json \
  --output_path experiments/nonconvex_mnist/svrg-0.05.json \
  --dataset MNIST \
  --layer_sizes 784 100 10 \
  --batch_size 10 \
  --learning_rate 0.05 \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.03 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda \
  --download
```

## STL
```
python train.py \
  --seed 77 \
  --optimizer SVRG \
  --run_name svrg_0.001.json \
  --output_path experiments/nonconvex_stl10/svrg-0.001.json \
  --dataset STL10 \
  --layer_sizes 27648 100 10 \
  --batch_size 10 \
  --learning_rate 0.001 \
  --weight_decay 0.01 \
  --warmup_learning_rate 0.001 \
  --num_warmup_epochs 20 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda \
  --download
```
# Critical evaluatin result
## MNIST with deeper network
```
python train.py \
  --seed 79 \
  --optimizer SVRG \
  --run_name svrg_0.025.json \
  --output_path experiments/nonconvex_mnist_deep/svrg-0.025.json \
  --dataset MNIST \
  --layer_sizes 784 600 300 10 \
  --batch_size 10 \
  --learning_rate 0.025 \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.03 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda \
  --download
```
## FMNIST
```bash
python train.py \
  --seed 79 \
  --optimizer SVRG \
  --run_name svrg_0.001.json \
  --output_path experiments/nonconvex_fmnist/svrg-0.001.json \
  --dataset FMNIST \
  --layer_sizes 784 100 10 \
  --batch_size 10 \
  --learning_rate 0.001 \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.03 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda \
  --download
``` 
# Running Hyperparameter search
Pleae open any new `.sh` file and paste the corresponding code to run

## SVRG CIFAR 10
```bash
# Warmup Learning Rate Search
for lr in 0.5 0.25 0.1 0.03 0.01 0.001; do python train.py \
  --seed 12 \
  --optimizer SGD \
  --run_name sgd_$lr.json \
  --output_path experiments/nonconvex_cifar10/warmup-$lr.json \
  --dataset MNIST \
  --layer_sizes 784 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.001 \
  --num_epochs 10 \
  --download
done

# SVRG Learning Rate Search
for lr in 0.5 0.25 0.1 0.05 0.025 0.01 0.005 0.001; do python train.py \
  --seed 79 \
  --optimizer SVRG \
  --run_name svrg_$lr.json \
  --output_path experiments/nonconvex_cifar10/svrg-$lr.json \
  --dataset MNIST \
  --layer_sizes 784 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.03 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda \
  --download
done
```

## SVRG MNIST
```bash
for lr in 0.5 0.25 0.1 0.03 0.01 0.001; do 
python train.py \
  --seed 12 \
  --optimizer SGD \
  --run_name sgd_$lr.json \
  --output_path experiments/nonconvex_mnist/warmup-$lr.json \
  --dataset MNIST \
  --layer_sizes 784 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.001 \
  --num_epochs 10 \
  --download
done

for lr in 0.5 0.25 0.1 0.05 0.025 0.01 0.005 0.001; do 
python train.py \
  --seed 79 \
  --optimizer SVRG \
  --run_name svrg_$lr.json \
  --output_path experiments/nonconvex_mnist/svrg-$lr.json \
  --dataset MNIST \
  --layer_sizes 784 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.03 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda \
  --download
done
```

## SVRG STL
```bash
for lr in 0.5 0.25 0.1 0.03 0.01 0.001; do python train.py \
  --seed 12 \
  --optimizer SGD \
  --run_name sgd_$lr.json \
  --output_path experiments/nonconvex_stl10/warmup-$lr.json \
  --dataset STL10 \
  --layer_sizes 27648 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.01 \
  --num_epochs 20 \
  --download
done

for lr in 0.5 0.25 0.1 0.05 0.025 0.01 0.005 0.001; do python train.py \
  --seed 77 \
  --optimizer SVRG \
  --run_name svrg_$lr.json \
  --output_path experiments/nonconvex_stl10/svrg-$lr.json \
  --dataset STL10 \
  --layer_sizes 27648 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.01 \
  --warmup_learning_rate 0.001 \
  --num_warmup_epochs 20 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda \
  --download
done
```
## SVRG MNIST with deeper network
```bash
for lr in 0.5 0.25 0.1 0.05 0.025 0.01 0.005 0.001; do python train.py \
  --seed 79 \
  --optimizer SVRG \
  --run_name svrg_$lr.json \
  --output_path experiments/nonconvex_mnist_deep/svrg-$lr.json \
  --dataset MNIST \
  --layer_sizes 784 600 300 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.03 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda \
  --download
done
```

## SVRG FMNIST
```bash
for lr in 0.5 0.25 0.1 0.05 0.025 0.01 0.005 0.001; do python train.py \
  --seed 79 \
  --optimizer SVRG \
  --run_name svrg_$lr.json \
  --output_path experiments/nonconvex_fmnist/svrg-$lr.json \
  --dataset FMNIST \
  --layer_sizes 784 100 10 \
  --batch_size 10 \
  --learning_rate $lr \
  --weight_decay 0.001 \
  --warmup_learning_rate 0.03 \
  --num_warmup_epochs 10 \
  --num_outer_epochs 250 \
  --num_inner_epochs 1 \
  --device cuda \
  --download
done
```

